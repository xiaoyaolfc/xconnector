# xconnector/test/test_integration.py
import asyncio
import pytest
import torch
from unittest.mock import Mock, AsyncMock, patch
from xconnector.connector import XConnector
from xconnector.config import XConnectorConfig, ConfigManager

class TestXConnectorIntegration:
    """端到端集成测试"""
    
    @pytest.fixture
    async def connector(self):
        """创建测试用的connector"""
        config = XConnectorConfig()
        connector = XConnector()
        yield connector
        # 清理
        if hasattr(connector, 'cleanup'):
            await connector.cleanup()
    
    @pytest.fixture
    def mock_request(self):
        """模拟请求"""
        request = Mock()
        request.request_id = "test_request_123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        request.model = "test_model"
        return request
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, connector, mock_request):
        """测试完整的请求处理流程"""
        # 1. 模拟Dynamo调度请求
        with patch.object(connector.dynamo, 'schedule_request') as mock_schedule:
            mock_schedule.return_value = "test_request_123"
            
            request_id = await connector.dynamo.schedule_request(mock_request)
            assert request_id == "test_request_123"
            mock_schedule.assert_called_once_with(mock_request)
        
        # 2. 模拟VLLM处理请求
        with patch.object(connector.vllm, 'recv_kv_caches') as mock_recv:
            mock_recv.return_value = (torch.randn(1, 100, 512), False, mock_request)
            
            model_executable = Mock()
            kv_caches = [torch.randn(1, 8, 100, 64) for _ in range(4)]
            
            result = await connector.vllm.recv_kv_caches(
                model_executable, mock_request, kv_caches
            )
            
            assert result[0].shape == (1, 100, 512)
            assert result[1] is False
            assert result[2] == mock_request
        
        # 3. 模拟LMCache存储
        with patch.object(connector.lmcache, 'save_kv_layer') as mock_save:
            mock_save.return_value = None
            
            await connector.lmcache.save_kv_layer(
                "layer_0", kv_caches[0], Mock()
            )
            
            mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, connector, mock_request):
        """测试缓存命中场景"""
        # 1. 模拟缓存命中
        cached_kv = [torch.randn(1, 8, 50, 64) for _ in range(4)]
        
        with patch.object(connector.lmcache, 'get_num_new_matched_tokens') as mock_match:
            mock_match.return_value = (50, True)  # 命中50个token
            
            result = await connector.lmcache.get_num_new_matched_tokens(
                mock_request, 100
            )
            
            assert result[0] == 50
            assert result[1] is True
    
    @pytest.mark.asyncio
    async def test_multi_worker_scenario(self, connector):
        """测试多worker场景"""
        # 模拟多个worker
        workers = ["worker_1", "worker_2", "worker_3"]
        
        with patch.object(connector.dynamo, 'get_worker_status') as mock_status:
            mock_status.return_value = {
                worker_id: {
                    'vllm_status': {'active': True, 'load': 0.5},
                    'cache_status': {'hit_rate': 0.8, 'size': 1000},
                    'active_requests': 2
                }
                for worker_id in workers
            }
            
            status = await connector.dynamo.get_worker_status()
            
            assert len(status) == 3
            for worker_id in workers:
                assert worker_id in status
                assert status[worker_id]['vllm_status']['active'] is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, connector, mock_request):
        """测试错误处理"""
        # 模拟VLLM错误
        with patch.object(connector.vllm, 'recv_kv_caches') as mock_recv:
            mock_recv.side_effect = Exception("VLLM error")
            
            try:
                await connector.vllm.recv_kv_caches(Mock(), mock_request, [])
            except Exception as e:
                assert "VLLM error" in str(e)
        
        # 模拟LMCache错误
        with patch.object(connector.lmcache, 'save_kv_layer') as mock_save:
            mock_save.side_effect = Exception("LMCache error")
            
            try:
                await connector.lmcache.save_kv_layer("layer", torch.randn(1, 1), Mock())
            except Exception as e:
                assert "LMCache error" in str(e)

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """测试并发请求处理"""
        connector = XConnector()
        
        async def process_request(request_id: str):
            """处理单个请求"""
            request = Mock()
            request.request_id = request_id
            request.prompt = f"Request {request_id}"
            
            # 模拟处理时间
            await asyncio.sleep(0.01)
            
            return f"Response for {request_id}"
        
        # 创建100个并发请求
        tasks = [
            process_request(f"req_{i}")
            for i in range(100)
        ]
        
        # 执行并发请求
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert all("Response for" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """测试内存使用情况"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 创建大量KV缓存
        kv_caches = []
        for i in range(100):
            kv_cache = [torch.randn(1, 8, 1000, 64) for _ in range(32)]
            kv_caches.append(kv_cache)
        
        peak_memory = process.memory_info().rss
        
        # 清理
        del kv_caches
        gc.collect()
        
        final_memory = process.memory_info().rss
        
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
        print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
        
        # 内存应该得到释放
        assert final_memory < peak_memory * 0.9

class TestConfigManager:
    """配置管理测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = XConnectorConfig()
        
        assert config.vllm.model_path == "facebook/opt-125m"
        assert config.lmcache.cache_size == 1000
        assert config.dynamo.max_workers == 4
        assert config.host == "localhost"
        assert config.port == 8000
    
    def test_config_loading(self, tmp_path):
        """测试配置加载"""
        config_file = tmp_path / "test_config.yaml"
        config_data = """
vllm:
  model_path: "