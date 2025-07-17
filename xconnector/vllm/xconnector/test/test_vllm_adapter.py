import asyncio
import pytest
from xconnector.vllm.xconnector.lmcache_adapter import LMCacheAdapter
from xconnector.vllm.xconnector.core import XConnectorCore
from xconnector.vllm.xconnector.vllm_adapter import VLLMAdapter


@pytest.mark.asyncio
async def test_vllm_adapter():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    with pytest.raises(AttributeError):
        await adapter.recv_kv_caches(None, None, [])
    with pytest.raises(AttributeError):
        await adapter.send_kv_caches(None, None, [], None)
    with pytest.raises(AttributeError):
        await adapter.get_finished(set())
