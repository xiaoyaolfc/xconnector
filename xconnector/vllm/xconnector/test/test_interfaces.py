import pytest
from xconnector.vllm.xconnector.interfaces import VLLMInterface, LMCacheInterface


@pytest.mark.asyncio
async def test_vllm_interface():
    interface = VLLMInterface()
    with pytest.raises(NotImplementedError):
        await interface.recv_kv_caches(None, None, [])
    with pytest.raises(NotImplementedError):
        await interface.send_kv_caches(None, None, [], None)
    with pytest.raises(NotImplementedError):
        await interface.get_finished(set())


@pytest.mark.asyncio
async def test_lmcache_interface():
    interface = LMCacheInterface()
    with pytest.raises(NotImplementedError):
        await interface.start_load_kv(None)
    with pytest.raises(NotImplementedError):
        await interface.wait_for_layer_load('layer')
    with pytest.raises(NotImplementedError):
        await interface.save_kv_layer('layer', None, None)
    with pytest.raises(NotImplementedError):
        await interface.wait_for_save()
    with pytest.raises(NotImplementedError):
        await interface.get_num_new_matched_tokens(None, 0)
    with pytest.raises(NotImplementedError):
        await interface.update_state_after_alloc(None, None, 0)
