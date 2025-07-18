import pytest
from xconnector.core.core import XConnectorCore
from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter
from vllm.forward_context import ForwardContext
from vllm.v1.request import Request  # 导入正确的Request类
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.sampling_params import SamplingParams  # 新增：导入采样参数类
from unittest.mock import Mock
import torch

# 外部定义模拟对象
no_compile_layers = []
attn_metadata = Mock(spec=AttentionMetadata)
virtual_engine = Mock()

@pytest.mark.asyncio
async def test_lmcache_adapter():
    core = XConnectorCore()
    adapter = LMCacheAdapter(core)
    
    # 创建ForwardContext
    context = ForwardContext(
        no_compile_layers=no_compile_layers,
        attn_metadata=attn_metadata,
        virtual_engine=virtual_engine
    )
    
    # 1. 准备Request的必填参数
    request_id = "test_request_1"  # 对应request_id参数
    prompt_token_ids = [100, 200, 300]  # 示例：prompt的token列表（需整数）
    sampling_params = SamplingParams(max_tokens=10)  # 最小采样参数（必填）
    
    # 2. 正确初始化Request（按源码参数）
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,  # 提供采样参数
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        pooling_params=None,
        eos_token_id=50000  # 示例：eos token id（可填实际值）
    )
    
    # 其他测试对象初始化
    blocks = KVCacheBlocks(blocks=())  # 关键修改：添加blocks参数
    kv_layer = torch.randn(10, 10)
    
    # 测试逻辑（保持不变）
    with pytest.raises(AttributeError):
        await adapter.start_load_kv(context)
    with pytest.raises(AttributeError):
        await adapter.wait_for_layer_load('layer')
    with pytest.raises(AttributeError):
        await adapter.save_kv_layer('layer', kv_layer, attn_metadata)
    with pytest.raises(AttributeError):
        await adapter.wait_for_save()
    with pytest.raises(AttributeError):
        await adapter.get_num_new_matched_tokens(request, 0)
    with pytest.raises(AttributeError):
        await adapter.update_state_after_alloc(request, blocks, 0)