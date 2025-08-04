# tests/unit/integrations/dynamo/test_minimal_sdk.py
"""
最小SDK单元测试

测试 minimal_sdk.py 中的所有核心功能：
- MinimalXConnectorSDK 初始化和配置解析
- 适配器动态导入和初始化
- SimpleKVHandler 的KV缓存处理
- 异步/同步初始化机制
- 错误处理和优雅降级
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any, List
import importlib

# 导入被测试的模块
from integrations.dynamo.minimal_sdk import (
    MinimalXConnectorSDK,
    SimpleKVHandler,
    AdapterInfo,
    create_minimal_sdk
)


@pytest.fixture
def basic_config():
    """基础配置fixture"""
    return {
        "enabled": True,
        "mode": "embedded",
        "log_level": "INFO",
        "adapters": [
            {
                "name": "lmcache",
                "type": "cache",
                "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                "config": {
                    "storage_backend": "memory",
                    "max_cache_size": 1024
                },
                "enabled": True
            }
        ]
    }


@pytest.fixture
def empty_config():
    """空配置fixture"""
    return {
        "enabled": True,
        "adapters": []
    }


@pytest.fixture
def mock_adapter_class():
    """模拟适配器类的fixture"""
    mock_class = MagicMock()
    mock_instance = MagicMock()

    # 模拟异步初始化方法
    mock_instance.initialize = AsyncMock(return_value=True)
    mock_instance.retrieve_kv = AsyncMock()
    mock_instance.store_kv = AsyncMock()
    mock_instance.cleanup_finished = AsyncMock()

    mock_class.return_value = mock_instance

    return mock_class, mock_instance


class TestAdapterInfo:
    """测试AdapterInfo数据类"""

    def test_adapter_info_creation(self):
        """测试AdapterInfo创建"""
        adapter_info = AdapterInfo(
            name="test_adapter",
            type="cache",
            class_path="test.module.TestAdapter",
            config={"key": "value"},
            enabled=True
        )

        assert adapter_info.name == "test_adapter"
        assert adapter_info.type == "cache"
        assert adapter_info.class_path == "test.module.TestAdapter"
        assert adapter_info.config == {"key": "value"}
        assert adapter_info.enabled is True

    def test_adapter_info_default_enabled(self):
        """测试AdapterInfo默认enabled值"""
        adapter_info = AdapterInfo(
            name="test",
            type="cache",
            class_path="test.TestAdapter",
            config={}
        )

        assert adapter_info.enabled is True


class TestMinimalXConnectorSDK:
    """测试MinimalXConnectorSDK类"""

    def test_sdk_initialization(self, basic_config):
        """测试SDK基础初始化"""
        sdk = MinimalXConnectorSDK(basic_config)

        assert sdk.config == basic_config
        assert sdk.initialized is False
        assert sdk.started is False
        assert sdk.kv_handler is None
        assert sdk.adapters == {}
        assert len(sdk.adapter_configs) == 1

        # 检查适配器配置解析
        adapter_config = sdk.adapter_configs[0]
        assert adapter_config.name == "lmcache"
        assert adapter_config.type == "cache"
        assert adapter_config.enabled is True

    def test_parse_adapter_configs_empty(self, empty_config):
        """测试解析空适配器配置"""
        sdk = MinimalXConnectorSDK(empty_config)

        assert len(sdk.adapter_configs) == 0

    def test_parse_adapter_configs_invalid(self):
        """测试解析无效适配器配置"""
        config = {
            "enabled": True,
            "adapters": [
                "invalid_string",  # 应该是字典
                {"name": "valid"}  # 缺少必需字段
            ]
        }

        sdk = MinimalXConnectorSDK(config)

        # 只有有效的配置被解析（虽然可能不完整）
        assert len(sdk.adapter_configs) == 1
        assert sdk.adapter_configs[0].name == "valid"
        assert sdk.adapter_configs[0].type == ""  # 缺少的字段默认为空

    def test_find_cache_adapter(self, basic_config):
        """测试查找缓存适配器"""
        sdk = MinimalXConnectorSDK(basic_config)

        cache_adapter = sdk._find_cache_adapter()

        assert cache_adapter is not None
        assert cache_adapter.name == "lmcache"
        assert cache_adapter.type == "cache"
        assert cache_adapter.enabled is True

    def test_find_cache_adapter_none_found(self):
        """测试没有找到缓存适配器"""
        config = {
            "enabled": True,
            "adapters": [
                {
                    "name": "vllm",
                    "type": "inference",  # 不是cache类型
                    "class_path": "test.VLLMAdapter",
                    "enabled": True
                }
            ]
        }

        sdk = MinimalXConnectorSDK(config)
        cache_adapter = sdk._find_cache_adapter()

        assert cache_adapter is None

    def test_find_cache_adapter_disabled(self):
        """测试查找被禁用的缓存适配器"""
        config = {
            "enabled": True,
            "adapters": [
                {
                    "name": "lmcache",
                    "type": "cache",
                    "class_path": "test.LMCacheAdapter",
                    "enabled": False  # 被禁用
                }
            ]
        }

        sdk = MinimalXConnectorSDK(config)
        cache_adapter = sdk._find_cache_adapter()

        assert cache_adapter is None

    def test_import_adapter_class_success(self, basic_config):
        """测试成功导入适配器类"""
        sdk = MinimalXConnectorSDK(basic_config)

        # 模拟导入成功
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.LMCacheAdapter = mock_class

        with patch('importlib.import_module', return_value=mock_module):
            result = sdk._import_adapter_class(
                "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter"
            )

        assert result == mock_class

    def test_import_adapter_class_import_error(self, basic_config):
        """测试导入适配器类失败"""
        sdk = MinimalXConnectorSDK(basic_config)

        with patch('importlib.import_module', side_effect=ImportError("No module")):
            result = sdk._import_adapter_class("nonexistent.module.Class")

        assert result is None

    def test_import_adapter_class_attribute_error(self, basic_config):
        """测试适配器类不存在"""
        sdk = MinimalXConnectorSDK(basic_config)

        mock_module = MagicMock()
        del mock_module.NonExistentClass  # 确保属性不存在

        with patch('importlib.import_module', return_value=mock_module), \
                patch('builtins.getattr', side_effect=AttributeError("No attribute")):
            result = sdk._import_adapter_class("test.module.NonExistentClass")

        assert result is None

    @pytest.mark.asyncio
    async def test_initialize_success(self, basic_config, mock_adapter_class):
        """测试异步初始化成功"""
        mock_class, mock_instance = mock_adapter_class

        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_import_adapter_class', return_value=mock_class):
            result = await sdk.initialize()

        assert result is True
        assert sdk.initialized is True
        assert sdk.kv_handler is not None
        assert "lmcache" in sdk.adapters

        # 验证适配器被正确初始化
        mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_cache_adapter(self, empty_config):
        """测试没有缓存适配器时的初始化"""
        sdk = MinimalXConnectorSDK(empty_config)

        result = await sdk.initialize()

        assert result is True  # 仍然成功，只是没有KV处理器
        assert sdk.initialized is True
        assert sdk.kv_handler is None

    @pytest.mark.asyncio
    async def test_initialize_adapter_import_failure(self, basic_config):
        """测试适配器导入失败时的初始化"""
        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_import_adapter_class', return_value=None):
            result = await sdk.initialize()

        assert result is True  # 初始化仍然成功，只是没有适配器
        assert sdk.initialized is True
        assert sdk.kv_handler is None

    @pytest.mark.asyncio
    async def test_initialize_adapter_init_failure(self, basic_config, mock_adapter_class):
        """测试适配器初始化失败"""
        mock_class, mock_instance = mock_adapter_class
        mock_instance.initialize.return_value = False  # 初始化失败

        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_import_adapter_class', return_value=mock_class):
            result = await sdk.initialize()

        assert result is True  # SDK初始化仍然成功
        assert sdk.initialized is True
        # 但是没有创建KV处理器，因为适配器初始化失败

    @pytest.mark.asyncio
    async def test_initialize_exception_handling(self, basic_config):
        """测试初始化过程中的异常处理"""
        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_find_cache_adapter', side_effect=Exception("Test error")):
            result = await sdk.initialize()

        assert result is False
        assert sdk.initialized is False

    def test_initialize_sync_success(self, basic_config, mock_adapter_class):
        """测试同步初始化成功"""
        mock_class, mock_instance = mock_adapter_class

        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_import_adapter_class', return_value=mock_class):
            sdk.initialize_sync()

        assert sdk.initialized is True
        assert sdk.kv_handler is not None
        assert "lmcache" in sdk.adapters

    def test_initialize_sync_failure(self, basic_config):
        """测试同步初始化失败"""
        sdk = MinimalXConnectorSDK(basic_config)

        with patch.object(sdk, '_find_cache_adapter', side_effect=Exception("Test error")):
            sdk.initialize_sync()

        # 同步初始化失败时不会改变initialized状态
        assert sdk.initialized is False

    def test_get_kv_handler(self, basic_config):
        """测试获取KV处理器"""
        sdk = MinimalXConnectorSDK(basic_config)
        mock_handler = MagicMock()
        sdk.kv_handler = mock_handler

        result = sdk.get_kv_handler()

        assert result == mock_handler

    def test_is_ready_true(self, basic_config):
        """测试SDK就绪状态为True"""
        sdk = MinimalXConnectorSDK(basic_config)
        sdk.initialized = True
        sdk.kv_handler = MagicMock()

        assert sdk.is_ready() is True

    def test_is_ready_false_not_initialized(self, basic_config):
        """测试SDK未初始化时就绪状态为False"""
        sdk = MinimalXConnectorSDK(basic_config)
        sdk.initialized = False
        sdk.kv_handler = MagicMock()

        assert sdk.is_ready() is False

    def test_is_ready_false_no_handler(self, basic_config):
        """测试没有KV处理器时就绪状态为False"""
        sdk = MinimalXConnectorSDK(basic_config)
        sdk.initialized = True
        sdk.kv_handler = None

        assert sdk.is_ready() is False

    def test_get_status(self, basic_config):
        """测试获取SDK状态"""
        sdk = MinimalXConnectorSDK(basic_config)
        sdk.initialized = True
        sdk.started = False
        sdk.kv_handler = MagicMock()
        sdk.adapters = {"lmcache": MagicMock()}

        status = sdk.get_status()

        assert status["initialized"] is True
        assert status["started"] is False
        assert status["kv_handler_available"] is True
        assert status["adapters_count"] == 1
        assert status["adapter_names"] == ["lmcache"]


class TestSimpleKVHandler:
    """测试SimpleKVHandler类"""

    @pytest.fixture
    def mock_cache_adapter(self):
        """模拟缓存适配器"""
        adapter = MagicMock()
        adapter.retrieve_kv = AsyncMock()
        adapter.store_kv = AsyncMock()
        adapter.cleanup_finished = AsyncMock()
        return adapter

    def test_kv_handler_initialization(self, mock_cache_adapter):
        """测试KV处理器初始化"""
        handler = SimpleKVHandler(mock_cache_adapter)

        assert handler.cache_adapter == mock_cache_adapter
        assert handler.total_requests == 0
        assert handler.cache_hits == 0

    @pytest.mark.asyncio
    async def test_retrieve_kv_success_found(self, mock_cache_adapter):
        """测试KV检索成功（命中）"""
        mock_cache_adapter.retrieve_kv.return_value = {
            "found": True,
            "data": "cached_data",
            "hidden_states": "cached_states"
        }

        handler = SimpleKVHandler(mock_cache_adapter)
        model_input = MagicMock()
        kv_caches = [MagicMock()]

        result = await handler.retrieve_kv(model_input, kv_caches)

        assert result["found"] is True
        assert result["data"] == "cached_data"
        assert handler.total_requests == 1
        assert handler.cache_hits == 1

        mock_cache_adapter.retrieve_kv.assert_called_once_with(model_input, kv_caches)

    @pytest.mark.asyncio
    async def test_retrieve_kv_success_not_found(self, mock_cache_adapter):
        """测试KV检索成功（未命中）"""
        mock_cache_adapter.retrieve_kv.return_value = {"found": False}

        handler = SimpleKVHandler(mock_cache_adapter)
        model_input = MagicMock()
        kv_caches = [MagicMock()]

        result = await handler.retrieve_kv(model_input, kv_caches)

        assert result["found"] is False
        assert handler.total_requests == 1
        assert handler.cache_hits == 0

    @pytest.mark.asyncio
    async def test_retrieve_kv_non_dict_return(self, mock_cache_adapter):
        """测试KV检索返回非字典结果"""
        mock_cache_adapter.retrieve_kv.return_value = "non_dict_result"

        handler = SimpleKVHandler(mock_cache_adapter)
        model_input = MagicMock()
        kv_caches = [MagicMock()]

        result = await handler.retrieve_kv(model_input, kv_caches)

        assert result["found"] is False
        assert result["data"] == "non_dict_result"
        assert handler.total_requests == 1
        assert handler.cache_hits == 0

    @pytest.mark.asyncio
    async def test_retrieve_kv_no_method(self):
        """测试适配器没有retrieve_kv方法"""
        adapter = MagicMock()
        del adapter.retrieve_kv  # 删除方法

        handler = SimpleKVHandler(adapter)

        result = await handler.retrieve_kv(MagicMock(), [MagicMock()])

        assert result["found"] is False
        assert result["reason"] == "method_not_available"

    @pytest.mark.asyncio
    async def test_retrieve_kv_exception(self, mock_cache_adapter):
        """测试KV检索时发生异常"""
        mock_cache_adapter.retrieve_kv.side_effect = Exception("Cache error")

        handler = SimpleKVHandler(mock_cache_adapter)

        result = await handler.retrieve_kv(MagicMock(), [MagicMock()])

        assert result["found"] is False
        assert "error" in result
        assert "Cache error" in result["error"]

    @pytest.mark.asyncio
    async def test_store_kv_success(self, mock_cache_adapter):
        """测试KV存储成功"""
        mock_cache_adapter.store_kv.return_value = True

        handler = SimpleKVHandler(mock_cache_adapter)
        model_input = MagicMock()
        kv_caches = [MagicMock()]
        hidden_states = MagicMock()
        metadata = {"key": "value"}

        result = await handler.store_kv(model_input, kv_caches, hidden_states, metadata)

        assert result is True
        mock_cache_adapter.store_kv.assert_called_once_with(
            model_input, kv_caches, hidden_states, metadata
        )

    @pytest.mark.asyncio
    async def test_store_kv_failure(self, mock_cache_adapter):
        """测试KV存储失败"""
        mock_cache_adapter.store_kv.return_value = False

        handler = SimpleKVHandler(mock_cache_adapter)

        result = await handler.store_kv(MagicMock(), [MagicMock()], MagicMock())

        assert result is False

    @pytest.mark.asyncio
    async def test_store_kv_no_method(self):
        """测试适配器没有store_kv方法"""
        adapter = MagicMock()
        del adapter.store_kv

        handler = SimpleKVHandler(adapter)

        result = await handler.store_kv(MagicMock(), [MagicMock()], MagicMock())

        assert result is False

    @pytest.mark.asyncio
    async def test_store_kv_exception(self, mock_cache_adapter):
        """测试KV存储时发生异常"""
        mock_cache_adapter.store_kv.side_effect = Exception("Store error")

        handler = SimpleKVHandler(mock_cache_adapter)

        result = await handler.store_kv(MagicMock(), [MagicMock()], MagicMock())

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_finished_success(self, mock_cache_adapter):
        """测试清理完成请求成功"""
        mock_cache_adapter.cleanup_finished.return_value = 3

        handler = SimpleKVHandler(mock_cache_adapter)
        request_ids = ["req1", "req2", "req3"]

        result = await handler.cleanup_finished(request_ids)

        assert result == 3
        mock_cache_adapter.cleanup_finished.assert_called_once_with(request_ids)

    @pytest.mark.asyncio
    async def test_cleanup_finished_no_method(self):
        """测试适配器没有cleanup_finished方法"""
        adapter = MagicMock()
        del adapter.cleanup_finished

        handler = SimpleKVHandler(adapter)
        request_ids = ["req1", "req2"]

        result = await handler.cleanup_finished(request_ids)

        assert result == 2  # 返回输入的数量

    @pytest.mark.asyncio
    async def test_cleanup_finished_exception(self, mock_cache_adapter):
        """测试清理时发生异常"""
        mock_cache_adapter.cleanup_finished.side_effect = Exception("Cleanup error")

        handler = SimpleKVHandler(mock_cache_adapter)

        result = await handler.cleanup_finished(["req1"])

        assert result == 0

    def test_get_stats(self, mock_cache_adapter):
        """测试获取统计信息"""
        handler = SimpleKVHandler(mock_cache_adapter)
        handler.total_requests = 10
        handler.cache_hits = 7

        stats = handler.get_stats()

        assert stats["total_requests"] == 10
        assert stats["cache_hits"] == 7
        assert stats["hit_rate"] == "70.0%"
        assert stats["adapter_available"] is True

    def test_get_stats_no_requests(self, mock_cache_adapter):
        """测试没有请求时的统计信息"""
        handler = SimpleKVHandler(mock_cache_adapter)

        stats = handler.get_stats()

        assert stats["total_requests"] == 0
        assert stats["cache_hits"] == 0
        assert stats["hit_rate"] == "0.0%"


class TestCreateMinimalSDK:
    """测试便捷创建函数"""

    def test_create_minimal_sdk(self, basic_config):
        """测试创建最小SDK的便捷函数"""
        sdk = create_minimal_sdk(basic_config)

        assert isinstance(sdk, MinimalXConnectorSDK)
        assert sdk.config == basic_config