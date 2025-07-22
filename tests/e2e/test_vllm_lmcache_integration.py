import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from xconnector.interfaces.vllm_integration import XConnectorVLLMBridge
from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter
from xconnector.core.connector import XConnector, AdapterType
import torch


@pytest.fixture
def mock_xconnector():
    connector = MagicMock()
    # 设置 mock 适配器
    cache_adapter = MagicMock()
    cache_adapter.retrieve_kv = AsyncMock()
    cache_adapter.store_kv = AsyncMock()

    # 配置 get_adapter 返回
    connector.get_adapter.side_effect = lambda name, _: cache_adapter if name == "lmcache" else None

    # 配置 route_message 直接返回值而不是协程
    connector.route_message = AsyncMock()

    return connector


@pytest.fixture
def vllm_bridge(mock_xconnector):
    # 创建模拟的 vLLM 配置
    class MockVllmConfig:
        model_config = {"name": "test-model"}
        parallel_config = {"size": 1}
        cache_config = {"size": 1024}

    # 创建桥接器
    bridge = XConnectorVLLMBridge(
        rank=0,
        local_rank=0,
        config=MockVllmConfig()
    )
    bridge.xconnector = mock_xconnector
    return bridge


@pytest.mark.asyncio
async def test_vllm_store_kv_caches(vllm_bridge):
    # 模拟输入数据
    model_input = MagicMock()
    kv_caches = [MagicMock(spec=torch.Tensor)]
    hidden_states = MagicMock(spec=torch.Tensor)

    # 调用发送KV缓存方法
    await vllm_bridge.send_kv_caches_and_hidden_states(
        model_executable=None,
        model_input=model_input,
        kv_caches=kv_caches,
        hidden_or_intermediate_states=hidden_states
    )

    # 验证调用了缓存适配器的存储方法
    vllm_bridge.xconnector.route_message.assert_called_with(
        source="vllm",
        target="lmcache",
        method="store_kv",
        model_input=model_input,
        kv_caches=kv_caches,
        hidden_states=hidden_states
    )


@pytest.mark.asyncio
async def test_vllm_recv_kv_caches_hit(vllm_bridge):
    # 模拟缓存命中
    cache_result = {
        "found": True,
        "hidden_states": MagicMock(),
        "skip_forward": True,
        "updated_input": MagicMock()
    }
    vllm_bridge.xconnector.route_message.return_value = cache_result

    # 调用接收KV缓存方法
    result = await vllm_bridge.recv_kv_caches_and_hidden_states(
        model_executable=None,
        model_input=MagicMock(),
        kv_caches=[]
    )

    # 验证返回了缓存结果
    assert result == (
        cache_result["hidden_states"],
        cache_result["skip_forward"],
        cache_result["updated_input"]
    )


@pytest.mark.asyncio
async def test_vllm_recv_kv_caches_miss(vllm_bridge):
    # 模拟缓存未命中
    vllm_bridge.xconnector.route_message.return_value = {"found": False}

    # 调用接收KV缓存方法
    result = await vllm_bridge.recv_kv_caches_and_hidden_states(
        model_executable=None,
        model_input="original_input",
        kv_caches=[]
    )

    # 解包结果
    hidden_states, bypass, model_input = result

    # 验证返回了未命中结果
    assert hidden_states is None
    assert bypass is False
    assert model_input == "original_input"