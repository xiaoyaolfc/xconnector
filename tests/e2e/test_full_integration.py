# tests/e2e/test_full_integration.py
"""
完整的端到端集成测试

测试 XConnector + vLLM + LMCache + Dynamo 的完整集成流程
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List
import torch

from xconnector.core.connector import XConnector, AdapterConfig, AdapterType
from xconnector.adapters.inference.vllm_adapter import VLLMAdapter
from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter
from xconnector.adapters.distributed.dynamo_adapter import DynamoAdapter, WorkerStatus, WorkerInfo
from xconnector.interfaces.cache_manager import CacheResult, CacheStatus
from xconnector.utils.config import ConnectorConfig


class MockInferenceRequest:
    """模拟完整的推理请求"""

    def __init__(self, request_id: str, prompt: str, model: str = "test-model"):
        self.request_id = request_id
        self.prompt = prompt
        self.model = model
        self.tokens = self._tokenize(prompt)
        self.arrival_time = asyncio.get_event_loop().time()

    def _tokenize(self, prompt: str) -> List[int]:
        """简单的模拟分词"""
        return [hash(word) % 1000 for word in prompt.split()]


@pytest.fixture
def full_xconnector_setup():
    """创建完整的 XConnector 设置"""
    # 重置单例
    XConnector._instance = None
    XConnector._initialized = False

    # 创建配置
    config = ConnectorConfig()

    # 创建连接器
    connector = XConnector(config)

    # 创建所有适配器
    vllm_adapter = VLLMAdapter(connector, {
        "model_name": "test-model",
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True
    })

    lmcache_adapter = LMCacheAdapter(connector, {
        "storage_backend": "memory",
        "max_cache_size": 1024,
        "enable_compression": True
    })

    dynamo_adapter = DynamoAdapter(connector, {
        "namespace": "test",
        "routing_policy": {
            "strategy": "least_loaded",
            "health_check_interval": 0.1
        }
    })

    # 注册适配器
    connector.inference_adapters["vllm"] = vllm_adapter
    connector.cache_adapters["lmcache"] = lmcache_adapter
    connector.distributed_adapters["dynamo"] = dynamo_adapter

    # 注册到路由器
    connector.router.register_adapter("vllm", vllm_adapter)
    connector.router.register_adapter("lmcache", lmcache_adapter)
    connector.router.register_adapter("dynamo", dynamo_adapter)

    return connector, vllm_adapter, lmcache_adapter, dynamo_adapter


@pytest.mark.asyncio
async def test_complete_request_lifecycle_with_cache_miss(full_xconnector_setup):
    """测试缓存未命中情况下的完整请求生命周期"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    # 启动系统
    await connector.start()

    # 注册工作节点
    worker_id = "worker-001"
    await dynamo_adapter.register_worker(worker_id, {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8001"
    })

    # 创建推理请求
    request = MockInferenceRequest("req_001", "What is artificial intelligence?")

    # 设置适配器 mock 行为
    _setup_cache_miss_behavior(lmcache_adapter)

    # 1. 路由请求到工作节点
    selected_worker = await dynamo_adapter.route_request({
        "model": request.model,
        "request_id": request.request_id
    })

    assert selected_worker == worker_id

    # 2. 模拟工作节点开始推理 - 检索缓存
    model_input = _create_model_input(request)
    kv_caches = [torch.randn(2, 8, 64, 64)]

    # 通过 XConnector 路由缓存检索
    cache_result = await connector.route_message(
        source="vllm",
        target="lmcache",
        method="retrieve_kv",
        model_input=model_input,
        kv_caches=kv_caches
    )

    # 验证缓存未命中
    assert cache_result.found is False
    assert cache_result.status == CacheStatus.MISS

    # 3. 模拟推理执行生成隐藏状态
    generated_hidden_states = torch.randn(1, len(request.tokens), 768)

    # 4. 存储 KV 缓存
    store_result = await connector.route_message(
        source="vllm",
        target="lmcache",
        method="store_kv",
        model_input=model_input,
        kv_caches=kv_caches,
        hidden_states=generated_hidden_states
    )

    assert store_result is True

    # 5. 模拟请求完成，清理资源
    finished_request_ids = [request.request_id]
    cleaned_count = await lmcache_adapter.cleanup_finished(finished_request_ids)

    # 验证系统状态
    stats = await lmcache_adapter.get_stats()
    assert stats.total_queries == 1
    assert stats.miss_count == 1
    assert stats.hit_count == 0

    await connector.stop()


@pytest.mark.asyncio
async def test_complete_request_lifecycle_with_cache_hit(full_xconnector_setup):
    """测试缓存命中情况下的完整请求生命周期"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    await connector.start()

    # 注册工作节点
    worker_id = "worker-001"
    await dynamo_adapter.register_worker(worker_id, {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8001"
    })

    # 第一个请求 - 缓存未命中
    request1 = MockInferenceRequest("req_001", "Hello world")
    _setup_cache_miss_behavior(lmcache_adapter)

    # 执行第一个请求
    model_input1 = _create_model_input(request1)
    kv_caches1 = [torch.randn(2, 8, 64, 64)]

    # 缓存未命中
    await connector.route_message(
        source="vllm", target="lmcache", method="retrieve_kv",
        model_input=model_input1, kv_caches=kv_caches1
    )

    # 存储到缓存
    await connector.route_message(
        source="vllm", target="lmcache", method="store_kv",
        model_input=model_input1, kv_caches=kv_caches1,
        hidden_states=torch.randn(1, len(request1.tokens), 768)
    )

    # 第二个请求 - 相同内容，应该命中缓存
    request2 = MockInferenceRequest("req_002", "Hello world")  # 相同内容
    _setup_cache_hit_behavior(lmcache_adapter)

    # 执行第二个请求
    model_input2 = _create_model_input(request2)
    kv_caches2 = [torch.randn(2, 8, 64, 64)]

    # 缓存命中
    cache_result = await connector.route_message(
        source="vllm", target="lmcache", method="retrieve_kv",
        model_input=model_input2, kv_caches=kv_caches2
    )

    # 验证缓存命中
    assert cache_result.found is True
    assert cache_result.status == CacheStatus.HIT
    assert "hidden_states" in cache_result.data
    assert cache_result.data["skip_forward"] is False

    # 验证缓存统计
    stats = await lmcache_adapter.get_stats()
    assert stats.total_queries == 2
    assert stats.hit_count == 1
    assert stats.miss_count == 1
    assert stats.hit_rate == 50.0

    await connector.stop()


@pytest.mark.asyncio
async def test_multi_worker_load_balancing(full_xconnector_setup):
    """测试多工作节点负载均衡"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    await connector.start()

    # 注册多个工作节点
    workers = [
        {"id": "worker-001", "active_requests": 2},
        {"id": "worker-002", "active_requests": 1},
        {"id": "worker-003", "active_requests": 3},
    ]

    for worker in workers:
        await dynamo_adapter.register_worker(worker["id"], {
            "model": "test-model",
            "gpu_memory": 4096,
            "endpoint": f"http://localhost:800{worker['id'][-1]}"
        })
        # 设置当前负载
        dynamo_adapter.workers[worker["id"]].active_requests = worker["active_requests"]

    # 创建多个请求并测试负载均衡
    requests = [
        {"id": f"req_{i:03d}", "prompt": f"Request {i}"}
        for i in range(6)
    ]

    routing_results = []
    for req in requests:
        selected_worker = await dynamo_adapter.route_request({
            "model": "test-model",
            "request_id": req["id"]
        })
        routing_results.append(selected_worker)

        # 模拟工作节点负载增加
        dynamo_adapter.workers[selected_worker].active_requests += 1

    # 验证负载均衡效果 - 应该优先选择负载较低的节点
    assert "worker-002" in routing_results  # 初始负载最低
    assert len(set(routing_results)) > 1  # 使用了多个工作节点

    # 验证最终负载分布
    final_loads = {
        worker_id: worker.active_requests
        for worker_id, worker in dynamo_adapter.workers.items()
    }

    # 负载应该相对均衡
    max_load = max(final_loads.values())
    min_load = min(final_loads.values())
    assert (max_load - min_load) <= 3  # 负载差异不超过3

    await connector.stop()


@pytest.mark.asyncio
async def test_concurrent_requests_with_mixed_cache_behavior(full_xconnector_setup):
    """测试并发请求的混合缓存行为"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    await connector.start()

    # 注册工作节点
    await dynamo_adapter.register_worker("worker-001", {
        "model": "test-model",
        "gpu_memory": 8192,
        "endpoint": "http://localhost:8001"
    })

    # 创建混合请求：一些重复（会命中缓存），一些唯一
    requests = [
        MockInferenceRequest("req_001", "What is AI?"),  # 新请求
        MockInferenceRequest("req_002", "Hello world"),  # 新请求
        MockInferenceRequest("req_003", "What is AI?"),  # 重复，应该命中缓存
        MockInferenceRequest("req_004", "How are you?"),  # 新请求
        MockInferenceRequest("req_005", "Hello world"),  # 重复，应该命中缓存
    ]

    async def process_single_request(request: MockInferenceRequest, should_hit_cache: bool):
        """处理单个请求"""
        # 路由到工作节点
        selected_worker = await dynamo_adapter.route_request({
            "model": request.model,
            "request_id": request.request_id
        })

        # 设置缓存行为
        if should_hit_cache:
            _setup_cache_hit_behavior(lmcache_adapter)
        else:
            _setup_cache_miss_behavior(lmcache_adapter)

        # 模拟推理过程
        model_input = _create_model_input(request)
        kv_caches = [torch.randn(2, 8, 64, 64)]

        # 检索缓存
        cache_result = await connector.route_message(
            source="vllm", target="lmcache", method="retrieve_kv",
            model_input=model_input, kv_caches=kv_caches
        )

        # 如果缓存未命中，存储新的缓存
        if not cache_result.found:
            await connector.route_message(
                source="vllm", target="lmcache", method="store_kv",
                model_input=model_input, kv_caches=kv_caches,
                hidden_states=torch.randn(1, len(request.tokens), 768)
            )

        return {
            "request_id": request.request_id,
            "worker": selected_worker,
            "cache_hit": cache_result.found
        }

    # 定义哪些请求应该命中缓存
    cache_expectations = [False, False, True, False, True]  # req_003 和 req_005 应该命中

    # 并发执行所有请求
    tasks = [
        process_single_request(req, should_hit)
        for req, should_hit in zip(requests, cache_expectations)
    ]

    results = await asyncio.gather(*tasks)

    # 验证结果
    assert len(results) == 5

    # 验证缓存命中情况
    cache_hits = [r["cache_hit"] for r in results]
    expected_hits = cache_expectations

    # 验证统计信息
    stats = await lmcache_adapter.get_stats()
    assert stats.total_queries == 5
    assert stats.hit_count == sum(expected_hits)
    assert stats.miss_count == 5 - sum(expected_hits)

    # 验证所有请求都被分配到了工作节点
    workers_used = set(r["worker"] for r in results)
    assert len(workers_used) >= 1
    assert "worker-001" in workers_used

    await connector.stop()


@pytest.mark.asyncio
async def test_system_failure_recovery(full_xconnector_setup):
    """测试系统故障恢复"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    await connector.start()

    # 注册工作节点
    worker_id = "worker-001"
    await dynamo_adapter.register_worker(worker_id, {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8001"
    })

    # 1. 正常请求处理
    request1 = MockInferenceRequest("req_001", "Normal request")
    _setup_cache_miss_behavior(lmcache_adapter)

    result1 = await connector.route_message(
        source="vllm", target="lmcache", method="retrieve_kv",
        model_input=_create_model_input(request1),
        kv_caches=[torch.randn(2, 8, 64, 64)]
    )

    assert result1.found is False  # 正常的缓存未命中

    # 2. 模拟缓存系统故障
    request2 = MockInferenceRequest("req_002", "Request during failure")
    _setup_cache_error_behavior(lmcache_adapter)

    # 系统应该优雅处理缓存错误
    try:
        result2 = await connector.route_message(
            source="vllm", target="lmcache", method="retrieve_kv",
            model_input=_create_model_input(request2),
            kv_caches=[torch.randn(2, 8, 64, 64)]
        )
        # 即使出错，也应该返回结果而不是抛出异常
        assert result2.status == CacheStatus.ERROR
    except Exception as e:
        # 如果抛出异常，记录错误信息
        assert "error" in str(e).lower()

    # 3. 模拟系统恢复
    request3 = MockInferenceRequest("req_003", "Request after recovery")
    _setup_cache_miss_behavior(lmcache_adapter)

    result3 = await connector.route_message(
        source="vllm", target="lmcache", method="retrieve_kv",
        model_input=_create_model_input(request3),
        kv_caches=[torch.randn(2, 8, 64, 64)]
    )

    assert result3.found is False  # 系统恢复正常

    await connector.stop()


@pytest.mark.asyncio
async def test_performance_metrics_collection(full_xconnector_setup):
    """测试性能指标收集"""
    connector, vllm_adapter, lmcache_adapter, dynamo_adapter = full_xconnector_setup

    await connector.start()

    # 注册工作节点
    await dynamo_adapter.register_worker("worker-001", {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8001"
    })

    # 处理多个请求以收集指标
    requests = [
        MockInferenceRequest(f"req_{i:03d}", f"Request {i}")
        for i in range(10)
    ]

    cache_hit_requests = [2, 4, 6, 8]  # 这些请求应该命中缓存

    for i, request in enumerate(requests):
        if i in cache_hit_requests:
            _setup_cache_hit_behavior(lmcache_adapter)
        else:
            _setup_cache_miss_behavior(lmcache_adapter)

        # 处理请求
        model_input = _create_model_input(request)
        kv_caches = [torch.randn(2, 8, 64, 64)]

        await connector.route_message(
            source="vllm", target="lmcache", method="retrieve_kv",
            model_input=model_input, kv_caches=kv_caches
        )

        if i not in cache_hit_requests:  # 缓存未命中时存储
            await connector.route_message(
                source="vllm", target="lmcache", method="store_kv",
                model_input=model_input, kv_caches=kv_caches,
                hidden_states=torch.randn(1, len(request.tokens), 768)
            )

    # 收集系统指标
    cache_stats = await lmcache_adapter.get_stats()
    health_status = await connector.get_health_status()
    dynamo_metrics = dynamo_adapter._get_custom_metrics()

    # 验证缓存指标
    assert cache_stats.total_queries == 10
    assert cache_stats.hit_count == len(cache_hit_requests)
    assert cache_stats.miss_count == 10 - len(cache_hit_requests)
    assert cache_stats.hit_rate == (len(cache_hit_requests) / 10) * 100

    # 验证系统健康状态
    assert health_status["connector"]["status"] == "healthy"
    assert health_status["connector"]["adapters_count"]["inference"] == 1
    assert health_status["connector"]["adapters_count"]["cache"] == 1
    assert health_status["connector"]["adapters_count"]["distributed"] == 1

    # 验证 Dynamo 指标
    assert dynamo_metrics["total_workers"] == 1
    assert dynamo_metrics["healthy_workers"] == 1

    await connector.stop()


# === 辅助函数 ===

def _create_model_input(request: MockInferenceRequest):
    """创建模型输入对象"""
    model_input = MagicMock()
    model_input.request_id = request.request_id
    model_input.input_tokens = request.tokens
    model_input.seq_len = len(request.tokens)
    model_input.is_prompt = True
    model_input.do_sample = True
    return model_input


def _setup_cache_miss_behavior(lmcache_adapter):
    """设置缓存未命中行为"""
    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(return_value=(None, False, None))
    lmcache_adapter.lmcache_should_store = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_store_kv = MagicMock()

    # 设置状态枚举
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"
    lmcache_adapter.StoreStatus = MagicMock()
    lmcache_adapter.StoreStatus.SKIP = "skip"


def _setup_cache_hit_behavior(lmcache_adapter):
    """设置缓存命中行为"""
    cached_hidden_states = torch.randn(1, 5, 768)

    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(
        return_value=(None, False, cached_hidden_states)  # 返回缓存的隐藏状态
    )

    # 设置状态枚举
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"


def _setup_cache_error_behavior(lmcache_adapter):
    """设置缓存错误行为"""
    lmcache_adapter.lmcache_should_retrieve = MagicMock(
        side_effect=Exception("Simulated cache error")
    )
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(
        side_effect=Exception("Simulated retrieval error")
    )

    # 设置状态枚举
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"