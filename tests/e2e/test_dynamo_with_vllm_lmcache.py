import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from xconnector.adapters.distributed.dynamo_adapter import DynamoAdapter
from xconnector.interfaces.vllm_integration import XConnectorVLLMBridge
from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter


@pytest.fixture
def full_integration_setup():
    # 创建XConnector实例
    connector = MagicMock()

    # 创建DynamoAdapter
    dynamo = DynamoAdapter(config={
        "namespace": "test",
        "routing_policy": {"strategy": "least_loaded"}
    })
    dynamo.core = connector

    # 创建vLLM桥接器
    vllm_bridge = XConnectorVLLMBridge(rank=0, local_rank=0, config=MagicMock())
    vllm_bridge.xconnector = connector

    # 创建LMCache适配器
    lmcache = LMCacheAdapter(config={
        "storage_backend": "memory",
        "max_cache_size": 1024
    })
    lmcache.core = connector

    # 配置XConnector返回适配器
    connector.get_adapter.side_effect = lambda name, _: {
        "dynamo": dynamo,
        "vllm": vllm_bridge,
        "lmcache": lmcache
    }.get(name)

    # 配置路由消息
    async def route_message(source, target, method, **kwargs):
        adapter = connector.get_adapter(target, None)
        if adapter and hasattr(adapter, method):
            return await getattr(adapter, method)(**kwargs)
        return None

    connector.route_message.side_effect = route_message

    # 注册工作节点
    asyncio.run(dynamo.register_worker("worker-001", {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8000"
    }))

    return {
        "dynamo": dynamo,
        "vllm": vllm_bridge,
        "lmcache": lmcache,
        "connector": connector
    }


@pytest.mark.asyncio
async def test_end_to_end_inference(full_integration_setup):
    dynamo = full_integration_setup["dynamo"]
    lmcache = full_integration_setup["lmcache"]

    # 模拟缓存方法
    lmcache.retrieve_kv = AsyncMock(return_value={"found": False})
    lmcache.store_kv = AsyncMock()

    # 创建推理请求
    request = {
        "model": "test-model",
        "prompt": "What is AI?",
        "max_tokens": 100
    }

    # 1. 通过Dynamo路由请求
    worker_id = await dynamo.route_request(request)
    assert worker_id == "worker-001"

    # 2. 模拟工作节点执行推理
    model_input = MagicMock()
    kv_caches = [MagicMock()]
    hidden_states = MagicMock()

    # 3. 存储KV缓存
    await full_integration_setup["vllm"].send_kv_caches_and_hidden_states(
        None, model_input, kv_caches, hidden_states
    )

    # 验证缓存被存储
    lmcache.store_kv.assert_called_once()

    # 4. 模拟第二次请求（缓存应命中）
    lmcache.retrieve_kv.reset_mock()
    lmcache.retrieve_kv.return_value = {
        "found": True,
        "hidden_states": "cached_states",
        "skip_forward": True,
        "updated_input": "updated_input"
    }

    # 再次路由请求
    worker_id = await dynamo.route_request(request)

    # 尝试检索缓存
    result = await full_integration_setup["vllm"].recv_kv_caches_and_hidden_states(
        None, model_input, []
    )

    # 验证缓存命中
    assert result == ("cached_states", True, "updated_input")
    lmcache.retrieve_kv.assert_called_once()