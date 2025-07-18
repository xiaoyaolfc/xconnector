import pytest
import torch
from unittest.mock import Mock, patch, call
from xconnector.core.core import XConnectorCore
from xconnector.adapters.inference.vllm_adapter import VLLMAdapter

# 模拟模型和输入数据
def create_mock_model():
    return Mock(spec=torch.nn.Module)

def create_mock_input(request_id="test_req", prompt="Hello World"):
    mock_input = Mock()
    mock_input.request_id = request_id
    mock_input.prompt = prompt
    return mock_input

def create_mock_kv_caches(num_layers=2, seq_len=10, hidden_size=64):
    return [torch.randn(1, 2, seq_len, hidden_size) for _ in range(num_layers)]

# 测试初始化和端点注册
@pytest.mark.asyncio
async def test_vllm_adapter_initialization():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 验证初始化状态
    assert adapter.core == core
    assert isinstance(adapter.kv_cache_store, dict)
    assert isinstance(adapter.finished_requests, set)
    
    # 验证端点注册
    with patch.object(core, 'register_vllm') as mock_register:
        adapter.register_endpoints()
        mock_register.assert_called_once_with(adapter)

# 测试KV缓存接收流程（无缓存）
@pytest.mark.asyncio
async def test_recv_kv_caches_no_cache():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 模拟依赖调用
    with patch.object(adapter, 'call') as mock_call:
        with patch.object(adapter, '_execute_model') as mock_execute:
            # 设置模拟返回值
            mock_call.side_effect = [
                False,  # should_retrieve 返回 False
            ]
            mock_execute.return_value = torch.randn(1, 512, 4096)
            
            # 执行测试
            model = create_mock_model()
            model_input = create_mock_input()
            kv_caches = create_mock_kv_caches()
            
            hidden_states, is_error, output = await adapter.recv_kv_caches(
                model, model_input, kv_caches
            )
            
            # 验证调用
            mock_call.assert_called_once_with('lmcache/should_retrieve', model_input)
            mock_execute.assert_called_once_with(model, model_input, kv_caches)
            
            # 验证结果
            assert not is_error
            assert hidden_states is not None
            assert output == model_input

# 测试KV缓存接收流程（有缓存）
@pytest.mark.asyncio
async def test_recv_kv_caches_with_cache():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 模拟依赖调用
    with patch.object(adapter, 'call') as mock_call:
        with patch.object(adapter, '_execute_model') as mock_execute:
            with patch.object(adapter, '_merge_kv_caches') as mock_merge:
                # 设置模拟返回值
                mock_call.side_effect = [
                    True,  # should_retrieve 返回 True
                    create_mock_kv_caches(seq_len=5),  # 模拟缓存的KV
                ]
                mock_merge.return_value = create_mock_kv_caches(seq_len=15)
                mock_execute.return_value = torch.randn(1, 512, 4096)
                
                # 执行测试
                model = create_mock_model()
                model_input = create_mock_input()
                kv_caches = create_mock_kv_caches(seq_len=10)
                
                hidden_states, is_error, output = await adapter.recv_kv_caches(
                    model, model_input, kv_caches
                )
                
                # 验证调用序列
                assert mock_call.call_count == 2
                mock_call.assert_has_calls([
                    call('lmcache/should_retrieve', model_input),
                    call('lmcache/retrieve_kv', model_input)
                ])
                
                # 验证合并调用
                cached_kv = create_mock_kv_caches(seq_len=5)
                mock_merge.assert_called_once_with(cached_kv, kv_caches)
                
                # 验证模型执行
                merged_kv = create_mock_kv_caches(seq_len=15)
                mock_execute.assert_called_once_with(model, model_input, merged_kv)
                
                # 验证结果
                assert not is_error
                assert hidden_states is not None
                assert output == model_input

# 测试KV缓存合并逻辑
@pytest.mark.asyncio
async def test_merge_kv_caches():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 创建测试数据
    cached_kv = [
        torch.ones(1, 2, 5, 64),  # 缓存的KV (seq_len=5)
        torch.ones(1, 2, 5, 64),
    ]
    
    new_kv = [
        torch.zeros(1, 2, 10, 64),  # 新的KV (seq_len=10)
        None,  # 第二层没有新的KV
    ]
    
    # 执行合并
    merged = adapter._merge_kv_caches(cached_kv, new_kv)
    
    # 验证结果
    assert len(merged) == 2
    
    # 验证第一层合并结果 (5+10=15)
    assert merged[0].shape == (1, 2, 15, 64)
    assert torch.all(merged[0][:, :, :5, :] == 1.0)  # 前5个是缓存的
    assert torch.all(merged[0][:, :, 5:, :] == 0.0)  # 后10个是新的
    
    # 验证第二层合并结果 (使用缓存的)
    assert torch.all(merged[1] == cached_kv[1])

# 测试KV缓存发送流程
@pytest.mark.asyncio
async def test_send_kv_caches():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 模拟依赖调用
    with patch.object(adapter, 'call') as mock_call:
        # 设置模拟返回值
        mock_call.return_value = True  # should_store 返回 True
        
        # 执行测试
        model = create_mock_model()
        model_input = create_mock_input("test_req")
        kv_caches = create_mock_kv_caches()
        hidden_states = torch.randn(1, 512, 4096)
        
        await adapter.send_kv_caches(model, model_input, kv_caches, hidden_states)
        
        # 验证调用序列
        assert mock_call.call_count == 2
        mock_call.assert_has_calls([
            call('lmcache/should_store', model_input),
            call('lmcache/store_kv', model_input, kv_caches, hidden_states)
        ])
        
        # 验证本地存储
        assert "test_req" in adapter.kv_cache_store
        stored = adapter.kv_cache_store["test_req"]
        assert torch.allclose(stored['kv_caches'][0], kv_caches[0])
        assert torch.allclose(stored['hidden_states'], hidden_states)

# 测试已完成请求处理
@pytest.mark.asyncio
async def test_get_finished():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 预先添加一些缓存数据
    adapter.kv_cache_store = {
        "req1": {"data": "cache1"},
        "req2": {"data": "cache2"},
        "req3": {"data": "cache3"},
    }
    
    # 模拟依赖调用
    with patch.object(adapter, 'call') as mock_call:
        # 执行测试
        finished_ids = {"req1", "req2"}
        result_ids, _ = await adapter.get_finished(finished_ids)
        
        # 验证调用
        mock_call.assert_called_once_with('lmcache/cleanup_finished', finished_ids)
        
        # 验证本地状态更新
        assert adapter.finished_requests == finished_ids
        assert "req1" not in adapter.kv_cache_store
        assert "req2" not in adapter.kv_cache_store
        assert "req3" in adapter.kv_cache_store  # 未完成的请求应保留
        
        # 验证返回结果
        assert result_ids == finished_ids

# 测试recv_kv_caches异常处理
@pytest.mark.asyncio
async def test_recv_kv_caches_error_handling():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 模拟抛出异常
    with patch.object(adapter, 'call') as mock_call:
        mock_call.side_effect = Exception("Mock error")
        
        # 执行测试
        model = create_mock_model()
        model_input = create_mock_input()
        kv_caches = create_mock_kv_caches()
        
        hidden_states, is_error, output = await adapter.recv_kv_caches(
            model, model_input, kv_caches
        )
        
        # 验证结果
        assert is_error
        assert hidden_states is None
        assert output == model_input

# 测试send_kv_caches异常处理
@pytest.mark.asyncio
async def test_send_kv_caches_error_handling():
    core = XConnectorCore()
    adapter = VLLMAdapter(core)
    
    # 模拟抛出异常
    with patch.object(adapter, 'call') as mock_call:
        mock_call.side_effect = Exception("Mock error")
        
        # 执行测试
        model = create_mock_model()
        model_input = create_mock_input()
        kv_caches = create_mock_kv_caches()
        hidden_states = torch.randn(1, 512, 4096)
        
        await adapter.send_kv_caches(model, model_input, kv_caches, hidden_states)
        
        # 验证未存储到本地
        assert model_input.request_id not in adapter.kv_cache_store