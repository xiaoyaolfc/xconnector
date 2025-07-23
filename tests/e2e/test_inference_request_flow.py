# tests/e2e/test_inference_request_flow.py
"""
端到端推理请求流程测试

测试从请求到响应的完整流程，包括缓存命中/未命中的情况
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List
import torch

from xconnector.core.connector import XConnector, AdapterConfig, AdapterType
from xconnector.adapters.inference.vllm_adapter import VLLMAdapter
from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter
from xconnector.interfaces.cache_manager import CacheResult, CacheStatus
from xconnector.utils.config import ConnectorConfig


class MockModelInput:
    """模拟 vLLM 模型输入"""

    def __init__(self, request_id: str, tokens: List[int], seq_len: int = None):
        self.request_id = request_id
        self.input_tokens = tokens
        self.seq_len = seq_len or len(tokens)
        self.is_prompt = True
        self.do_sample = True


class MockVLLMEngine:
    """模拟 vLLM 引擎"""

    def __init__(self):
        self.requests = {}
        self.current_step = 0

    def add_request(self, request_id: str, prompt, params, **kwargs):
        self.requests[request_id] = {
            "prompt": prompt,
            "params": params,
            "status": "running"
        }

    def step(self):
        # 模拟推理步骤，返回部分完成的请求
        responses = []
        for req_id, req_data in list(self.requests.items()):
            if self.current_step > 0:  # 第二步完成
                req_data["status"] = "finished"
                responses.append(self._create_mock_output(req_id))
                del self.requests[req_id]

        self.current_step += 1
        return responses

    def _create_mock_output(self, request_id: str):
        """创建模拟的输出"""
        mock_output = MagicMock()
        mock_output.request_id = request_id
        mock_output.finished = True
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Generated response text"
        mock_output.outputs[0].token_ids = [101, 102, 103]
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.prompt_token_ids = [1, 2, 3, 4]
        return mock_output

    def has_unfinished_requests(self):
        return len(self.requests) > 0

    def get_num_unfinished_requests(self):
        return len(self.requests)

    def abort_request(self, request_id: str):
        if request_id in self.requests:
            del self.requests[request_id]


@pytest.fixture
def mock_vllm_engine():
    """提供模拟的 vLLM 引擎"""
    return MockVLLMEngine()


@pytest.fixture
def connector_with_adapters():
    """创建带有适配器的 XConnector 实例"""
    # 重置单例
    XConnector._instance = None
    XConnector._initialized = False

    # 创建配置
    config = ConnectorConfig()

    # 创建连接器
    connector = XConnector(config)

    # 手动创建适配器（避免实际加载）
    vllm_adapter = VLLMAdapter(connector, {"model_name": "test-model"})
    lmcache_adapter = LMCacheAdapter(connector, {"storage_backend": "memory"})

    # 手动注册适配器
    connector.inference_adapters["vllm"] = vllm_adapter
    connector.cache_adapters["lmcache"] = lmcache_adapter

    # 注册到路由器
    connector.router.register_adapter("vllm", vllm_adapter)
    connector.router.register_adapter("lmcache", lmcache_adapter)

    return connector, vllm_adapter, lmcache_adapter


@pytest.mark.asyncio
async def test_inference_request_cache_miss_flow(connector_with_adapters, mock_vllm_engine):
    """测试缓存未命中的完整推理流程"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 模拟请求数据
    request_id = "test_request_001"
    model_input = MockModelInput(request_id, [1, 2, 3, 4, 5])
    kv_caches = [torch.randn(2, 8, 64, 64)]  # 模拟 KV 缓存张量
    hidden_states = torch.randn(1, 5, 768)  # 模拟隐藏状态

    # 设置缓存适配器行为（缓存未命中）
    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(return_value=(model_input, False, None))
    lmcache_adapter.lmcache_should_store = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_store_kv = MagicMock()
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"
    lmcache_adapter.StoreStatus = MagicMock()
    lmcache_adapter.StoreStatus.SKIP = "skip"

    # 模拟 vLLM 适配器的 KV 缓存检索（缓存未命中）
    with patch.object(vllm_adapter, 'lmcache_should_retrieve') as mock_should_retrieve:
        mock_should_retrieve.return_value = lmcache_adapter.RetrieveStatus.MISS

        # 1. 接收 KV 缓存（缓存未命中）
        result = await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)

        # 验证缓存未命中的行为
        assert result[0] is None  # hidden_states
        assert result[1] is False  # skip_forward
        assert result[2] == model_input  # updated_input

        # 验证统计信息
        assert lmcache_adapter.miss_count == 1
        assert lmcache_adapter.total_queries == 1

    # 模拟推理过程生成了新的隐藏状态
    generated_hidden_states = torch.randn(1, 5, 768)

    # 2. 发送 KV 缓存（存储到缓存）
    with patch.object(vllm_adapter, 'lmcache_should_store') as mock_should_store:
        mock_should_store.return_value = MagicMock()  # 不是 SKIP

        await vllm_adapter.send_kv_caches(None, model_input, kv_caches, generated_hidden_states)

        # 验证存储被调用
        lmcache_adapter.lmcache_store_kv.assert_called_once()


@pytest.mark.asyncio
async def test_inference_request_cache_hit_flow(connector_with_adapters):
    """测试缓存命中的完整推理流程"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 模拟请求数据
    request_id = "test_request_002"
    model_input = MockModelInput(request_id, [1, 2, 3, 4, 5])
    kv_caches = [torch.randn(2, 8, 64, 64)]
    cached_hidden_states = torch.randn(1, 5, 768)

    # 设置缓存适配器行为（缓存命中）
    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(
        return_value=(model_input, True, cached_hidden_states)  # skip_forward=True
    )
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"

    # 模拟 vLLM 适配器的 KV 缓存检索（缓存命中）
    with patch.object(vllm_adapter, 'lmcache_should_retrieve') as mock_should_retrieve:
        mock_should_retrieve.return_value = MagicMock()  # 不是 MISS

        # 接收 KV 缓存（缓存命中）
        result = await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)

        # 验证缓存命中的行为
        assert torch.equal(result[0], cached_hidden_states)  # 返回缓存的隐藏状态
        assert result[1] is True  # skip_forward=True
        assert result[2] == model_input  # updated_input

        # 验证统计信息
        assert lmcache_adapter.hit_count == 1
        assert lmcache_adapter.total_queries == 1


@pytest.mark.asyncio
async def test_complete_inference_request_lifecycle(connector_with_adapters):
    """测试完整的推理请求生命周期"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 启动连接器
    await connector.start()

    # 模拟多个请求的处理
    requests = [
        {"id": "req_001", "tokens": [1, 2, 3]},
        {"id": "req_002", "tokens": [4, 5, 6]},
        {"id": "req_003", "tokens": [1, 2, 3]},  # 与 req_001 相同，应该命中缓存
    ]

    results = []

    for req in requests:
        model_input = MockModelInput(req["id"], req["tokens"])
        kv_caches = [torch.randn(2, 8, 64, 64)]

        # 设置适配器行为
        if req["id"] == "req_003":
            # 第三个请求应该命中缓存
            lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
            lmcache_adapter.lmcache_retrieve_kv = MagicMock(
                return_value=(model_input, True, torch.randn(1, 3, 768))
            )
        else:
            # 前两个请求缓存未命中
            lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
            lmcache_adapter.lmcache_retrieve_kv = MagicMock(
                return_value=(model_input, False, None)
            )

        lmcache_adapter.RetrieveStatus = MagicMock()
        lmcache_adapter.RetrieveStatus.MISS = "miss"

        # 模拟推理过程
        with patch.object(vllm_adapter, 'lmcache_should_retrieve') as mock_should_retrieve:
            if req["id"] == "req_003":
                mock_should_retrieve.return_value = MagicMock()  # 命中
            else:
                mock_should_retrieve.return_value = lmcache_adapter.RetrieveStatus.MISS  # 未命中

            # 执行推理
            result = await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)
            results.append(result)

            # 如果是缓存未命中，模拟存储
            if not result[1]:  # skip_forward=False
                lmcache_adapter.lmcache_should_store = MagicMock(return_value=MagicMock())
                lmcache_adapter.lmcache_store_kv = MagicMock()

                with patch.object(vllm_adapter, 'lmcache_should_store') as mock_should_store:
                    mock_should_store.return_value = MagicMock()
                    await vllm_adapter.send_kv_caches(None, model_input, kv_caches, torch.randn(1, 3, 768))

    # 验证结果
    assert len(results) == 3

    # 前两个请求应该是缓存未命中
    assert results[0][1] is False  # req_001: skip_forward=False
    assert results[1][1] is False  # req_002: skip_forward=False

    # 第三个请求应该是缓存命中
    assert results[2][1] is True  # req_003: skip_forward=True

    # 验证缓存统计
    assert lmcache_adapter.total_queries >= 3
    assert lmcache_adapter.hit_count >= 1
    assert lmcache_adapter.miss_count >= 2

    await connector.stop()


@pytest.mark.asyncio
async def test_inference_error_handling(connector_with_adapters):
    """测试推理过程中的错误处理"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    request_id = "error_request"
    model_input = MockModelInput(request_id, [1, 2, 3])
    kv_caches = [torch.randn(2, 8, 64, 64)]

    # 模拟缓存检索出错
    lmcache_adapter.lmcache_should_retrieve = MagicMock(side_effect=Exception("Cache retrieval error"))

    # 执行推理，应该优雅处理错误
    result = await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)

    # 验证错误处理
    assert result[0] is None  # hidden_states
    assert result[1] is True  # skip_forward (错误时返回 True 跳过)
    assert result[2] == model_input  # updated_input


@pytest.mark.asyncio
async def test_concurrent_inference_requests(connector_with_adapters):
    """测试并发推理请求处理"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 设置适配器行为
    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.lmcache_retrieve_kv = MagicMock(return_value=(None, False, None))
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"

    async def process_request(request_id: str, tokens: List[int]):
        """处理单个推理请求"""
        model_input = MockModelInput(request_id, tokens)
        kv_caches = [torch.randn(2, 8, 64, 64)]

        with patch.object(vllm_adapter, 'lmcache_should_retrieve') as mock_should_retrieve:
            mock_should_retrieve.return_value = lmcache_adapter.RetrieveStatus.MISS
            result = await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)
            return result

    # 创建并发请求
    tasks = [
        process_request(f"concurrent_req_{i}", [i, i + 1, i + 2])
        for i in range(5)
    ]

    # 并发执行
    results = await asyncio.gather(*tasks)

    # 验证所有请求都被处理
    assert len(results) == 5
    for result in results:
        assert result[0] is None  # hidden_states
        assert result[1] is False  # skip_forward

    # 验证并发安全性
    assert lmcache_adapter.total_queries == 5
    assert lmcache_adapter.miss_count == 5


@pytest.mark.asyncio
async def test_request_cleanup_flow(connector_with_adapters):
    """测试请求完成后的清理流程"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 模拟一些已完成的请求
    finished_request_ids = {"req_001", "req_002", "req_003"}

    # 先添加一些缓存条目
    for req_id in finished_request_ids:
        model_input = MockModelInput(req_id, [1, 2, 3])
        cache_key = lmcache_adapter._generate_cache_key(model_input)
        lmcache_adapter.cache_entries[cache_key] = MagicMock()

    # 添加一些不应该被清理的条目
    other_request = MockModelInput("other_req", [4, 5, 6])
    other_cache_key = lmcache_adapter._generate_cache_key(other_request)
    lmcache_adapter.cache_entries[other_cache_key] = MagicMock()

    initial_count = len(lmcache_adapter.cache_entries)

    # 执行清理
    cleaned_count = await lmcache_adapter.cleanup_finished(list(finished_request_ids))

    # 验证清理结果
    assert cleaned_count > 0
    assert len(lmcache_adapter.cache_entries) < initial_count

    # 验证其他请求的缓存条目未被清理
    assert other_cache_key in lmcache_adapter.cache_entries


@pytest.mark.asyncio
async def test_cache_adapter_stats_tracking(connector_with_adapters):
    """测试缓存适配器的统计信息跟踪"""
    connector, vllm_adapter, lmcache_adapter = connector_with_adapters

    # 初始统计
    initial_stats = await lmcache_adapter.get_stats()
    assert initial_stats.total_queries == 0
    assert initial_stats.hit_count == 0
    assert initial_stats.miss_count == 0

    # 模拟多个请求
    requests = [
        ("req_001", [1, 2, 3]),
        ("req_002", [4, 5, 6]),
        ("req_001", [1, 2, 3]),  # 重复请求，应该命中缓存
    ]

    lmcache_adapter.lmcache_should_retrieve = MagicMock(return_value=MagicMock())
    lmcache_adapter.RetrieveStatus = MagicMock()
    lmcache_adapter.RetrieveStatus.MISS = "miss"

    for i, (req_id, tokens) in enumerate(requests):
        model_input = MockModelInput(req_id, tokens)
        kv_caches = [torch.randn(2, 8, 64, 64)]

        if i == 2:  # 第三个请求（重复）应该命中缓存
            lmcache_adapter.lmcache_retrieve_kv = MagicMock(
                return_value=(model_input, True, torch.randn(1, 3, 768))
            )
        else:
            lmcache_adapter.lmcache_retrieve_kv = MagicMock(
                return_value=(model_input, False, None)
            )

        with patch.object(vllm_adapter, 'lmcache_should_retrieve') as mock_should_retrieve:
            if i == 2:
                mock_should_retrieve.return_value = MagicMock()  # 命中
            else:
                mock_should_retrieve.return_value = lmcache_adapter.RetrieveStatus.MISS

            await vllm_adapter.recv_kv_caches(None, model_input, kv_caches)

    # 验证最终统计
    final_stats = await lmcache_adapter.get_stats()
    assert final_stats.total_queries == 3
    assert final_stats.hit_count == 1
    assert final_stats.miss_count == 2
    assert final_stats.hit_rate == 33.33333333333333  # 1/3 * 100