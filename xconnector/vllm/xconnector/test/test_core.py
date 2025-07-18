import asyncio
import pytest
from xconnector.core.core import XConnectorCore
from xconnector.vllm.xconnector.interfaces import VLLMInterface, LMCacheInterface


class MockVLLMInterface(VLLMInterface):
    async def recv_kv_caches(self, *args, **kwargs):
        return args, kwargs

    async def send_kv_caches(self, *args, **kwargs):
        pass

    async def get_finished(self, *args, **kwargs):
        return args, kwargs


class MockLMCacheInterface(LMCacheInterface):
    async def start_load_kv(self, *args, **kwargs):
        pass

    async def wait_for_layer_load(self, *args, **kwargs):
        pass

    async def save_kv_layer(self, *args, **kwargs):
        pass

    async def wait_for_save(self, *args, **kwargs):
        pass

    async def get_num_new_matched_tokens(self, *args, **kwargs):
        return args, kwargs

    async def update_state_after_alloc(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_xconnector_core():
    core = XConnectorCore()
    vllm_adapter = MockVLLMInterface()
    lmcache_adapter = MockLMCacheInterface()
    core.register_vllm(vllm_adapter)
    core.register_lmcache(lmcache_adapter)

    # 测试路由功能
    result = await core.route('vllm/recv_kv_caches', 1, 2, key='value')
    assert result[0] == (1, 2)
    assert result[1] == {'key': 'value'}

    # 测试创建端点
    core.create_endpoint('test_endpoint')
    assert 'test_endpoint' in core.connection_table

    # 测试发送和接收消息
    await core.send('test_endpoint', 3, 4, key2='value2')
    args, kwargs = await core.receive('test_endpoint')
    assert args == (3, 4)
    assert kwargs == {'key2': 'value2'}

    # 测试启动和停止任务
    async def mock_task():
        await asyncio.sleep(0.1)
        return 'task_result'

    core.start_task('test_task', mock_task())
    await asyncio.sleep(0.2)
    core.stop_task('test_task')
