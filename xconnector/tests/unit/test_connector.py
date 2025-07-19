import pytest
import asyncio
import logging

# 添加这个配置以支持异步测试
pytestmark = pytest.mark.asyncio

# 原有的导入保持不变
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Optional, Dict, List, Any
from xconnector.core.connector import XConnector
from xconnector.utils.config import (
    ConnectorConfig, AdapterConfig, AdapterType
)
from xconnector.interfaces.base_interface import BaseInterface
from xconnector.core.plugin_manager import PluginManager
from xconnector.core.router import Router


# 配置日志记录
@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """设置日志捕获"""
    caplog.set_level(logging.DEBUG)
    yield

@pytest.fixture
def connector_config():
    """Fixture to create a valid ConnectorConfig instance"""
    # 创建默认配置
    config = ConnectorConfig(adapters=[])

    # 修改健康检查配置
    config.health_check.interval = 10
    config.health_check.log_results = True

    return config


@pytest.fixture
def mock_plugin_manager():
    """Fixture to mock PluginManager"""
    manager = MagicMock(spec=PluginManager)
    manager.load_adapter = AsyncMock()
    manager.unload_adapter = AsyncMock()
    manager.register_adapter = MagicMock()
    return manager


@pytest.fixture
def mock_router():
    """Fixture to mock Router"""
    router = MagicMock(spec=Router)
    router.add_route = MagicMock()
    router.route = AsyncMock()
    return router


@pytest.fixture
def xconnector(connector_config, mock_plugin_manager, mock_router):
    """统一的主fixture，用于创建带mock组件的XConnector实例"""
    # 重置单例状态
    XConnector._instance = None
    XConnector._initialized = False

    with patch('xconnector.core.connector.PluginManager', return_value=mock_plugin_manager), \
            patch('xconnector.core.connector.Router', return_value=mock_router), \
            patch.object(XConnector, '_setup_health_check', new=MagicMock()), \
            patch.object(XConnector, '_register_builtin_adapters', new=MagicMock()):  # 跳过内置适配器注册

        # 创建实例
        connector = XConnector(config=connector_config)

        # 确保初始化完成
        XConnector._initialized = True

        # 返回之前，重置mock的调用记录
        mock_plugin_manager.reset_mock()
        mock_router.reset_mock()

        yield connector

        # 清理
        asyncio.run(connector.stop())
        XConnector._instance = None
        XConnector._initialized = False


# 改进的模拟适配器类
class MockAdapter(BaseInterface):
    def __init__(self, name):
        self.name = name
        self.adapter_name = name  # 添加 adapter_name 属性
        self.started = False
        self.stopped = False
        self.cleaned_up = False

    async def initialize(self):
        """模拟初始化方法"""
        pass

    async def get_capabilities(self):
        """模拟获取能力信息"""
        return {"inference": True}

    async def get_metrics(self):
        """模拟获取指标"""
        return {"requests": 0, "latency": 0.0}

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def cleanup(self):
        self.cleaned_up = True

    async def health_check(self):
        return {"status": "healthy"}


def test_singleton_pattern(xconnector, connector_config):
    """测试单例模式"""
    # 创建第二个实例
    connector2 = XConnector(connector_config)

    # 验证是同一个实例
    assert xconnector is connector2
    assert id(xconnector) == id(connector2)

    # 验证初始化状态
    assert XConnector._initialized is True


async def test_load_adapter(xconnector, mock_plugin_manager, caplog):
    """测试加载适配器"""
    # 设置日志级别
    caplog.set_level(logging.DEBUG)

    # 创建模拟适配器配置
    adapter_config = AdapterConfig(
        name="test_adapter",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )

    # 创建模拟适配器实例
    mock_adapter = MockAdapter("test_adapter")

    # 设置插件管理器的load_adapter返回模拟适配器实例
    mock_plugin_manager.load_adapter.return_value = mock_adapter

    # 加载适配器
    result = await xconnector.load_adapter(adapter_config)

    # 检查返回值
    assert result is mock_adapter, f"返回值不是预期的适配器实例: {result} vs {mock_adapter}"

    # 验证插件管理器方法被调用
    mock_plugin_manager.load_adapter.assert_called_once_with(
        adapter_config, xconnector.core
    )

    # 打印适配器字典内容用于调试
    print(f"inference_adapters: {xconnector.inference_adapters}")

    # 验证适配器被正确加载
    assert "test_adapter" in xconnector.inference_adapters, \
        f"适配器字典内容: {xconnector.inference_adapters}"

    assert xconnector.inference_adapters["test_adapter"] is mock_adapter, \
        f"返回的适配器: {xconnector.inference_adapters['test_adapter']}, 期望的适配器: {mock_adapter}"

    # 检查日志
    assert any("Loaded adapter: test_adapter (inference)" in record.message for record in caplog.records), \
        f"日志中没有找到加载记录。实际日志: {caplog.text}"

async def test_unload_adapter(xconnector, mock_plugin_manager):
    """测试卸载适配器"""
    # 创建并加载一个模拟适配器
    adapter_config = AdapterConfig(
        name="test_adapter",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )

    # 创建模拟适配器实例
    mock_adapter = MockAdapter("test_adapter")

    # 设置插件管理器的load_adapter返回模拟适配器实例
    mock_plugin_manager.load_adapter.return_value = mock_adapter

    # 加载适配器
    await xconnector.load_adapter(adapter_config)

    # 验证适配器已加载
    assert "test_adapter" in xconnector.inference_adapters

    # 卸载适配器
    await xconnector.unload_adapter("test_adapter", AdapterType.INFERENCE)

    # 验证适配器被正确卸载和清理
    assert "test_adapter" not in xconnector.inference_adapters
    assert mock_adapter.cleaned_up is True
    assert mock_adapter.stopped is True


async def test_get_adapter(xconnector, mock_plugin_manager):
    """测试获取适配器"""
    # 创建并加载一个模拟适配器
    adapter_config = AdapterConfig(
        name="test_adapter",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )

    # 创建模拟适配器实例
    mock_adapter = MockAdapter("test_adapter")

    # 设置插件管理器的load_adapter返回模拟适配器实例
    mock_plugin_manager.load_adapter.return_value = mock_adapter

    # 加载适配器
    await xconnector.load_adapter(adapter_config)

    # 获取适配器
    adapter = xconnector.get_adapter("test_adapter", AdapterType.INFERENCE)

    # 验证返回的适配器正确
    assert adapter is mock_adapter


async def test_list_adapters(xconnector, mock_plugin_manager):
    """测试列出适配器"""
    # 创建并加载多个模拟适配器
    inference_config = AdapterConfig(
        name="test_inference",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )
    mock_inference = MockAdapter("test_inference")
    mock_plugin_manager.load_adapter.return_value = mock_inference
    await xconnector.load_adapter(inference_config)

    cache_config = AdapterConfig(
        name="test_cache",
        type=AdapterType.CACHE,
        class_path="test.path",
        config={}
    )
    mock_cache = MockAdapter("test_cache")
    mock_plugin_manager.load_adapter.return_value = mock_cache
    await xconnector.load_adapter(cache_config)

    distributed_config = AdapterConfig(
        name="test_distributed",
        type=AdapterType.DISTRIBUTED,
        class_path="test.path",
        config={}
    )
    mock_distributed = MockAdapter("test_distributed")
    mock_plugin_manager.load_adapter.return_value = mock_distributed
    await xconnector.load_adapter(distributed_config)

    # 列出适配器
    adapters = xconnector.list_adapters()

    # 验证返回的字典结构正确
    assert "test_inference" in adapters["inference"]
    assert "test_cache" in adapters["cache"]
    assert "test_distributed" in adapters["distributed"]


async def test_start_stop(xconnector, mock_plugin_manager):
    """测试启动和停止"""
    # 创建并加载多个模拟适配器
    adapter_config1 = AdapterConfig(
        name="adapter1",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )
    mock_adapter1 = MockAdapter("adapter1")
    mock_plugin_manager.load_adapter.return_value = mock_adapter1
    await xconnector.load_adapter(adapter_config1)

    adapter_config2 = AdapterConfig(
        name="adapter2",
        type=AdapterType.CACHE,
        class_path="test.path",
        config={}
    )
    mock_adapter2 = MockAdapter("adapter2")
    mock_plugin_manager.load_adapter.return_value = mock_adapter2
    await xconnector.load_adapter(adapter_config2)

    # 启动connector
    await xconnector.start()

    # 验证状态和适配器启动
    assert xconnector.is_running is True
    assert mock_adapter1.started is True
    assert mock_adapter2.started is True

    # 停止connector
    await xconnector.stop()

    # 验证状态和适配器停止
    assert xconnector.is_running is False
    assert mock_adapter1.stopped is True
    assert mock_adapter2.stopped is True


async def test_health_check(xconnector, mock_plugin_manager):
    """测试健康检查"""
    # 创建并加载多个模拟适配器
    adapter_config1 = AdapterConfig(
        name="adapter1",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )
    mock_adapter1 = MockAdapter("adapter1")
    mock_plugin_manager.load_adapter.return_value = mock_adapter1
    await xconnector.load_adapter(adapter_config1)

    adapter_config2 = AdapterConfig(
        name="adapter2",
        type=AdapterType.CACHE,
        class_path="test.path",
        config={}
    )
    mock_adapter2 = MockAdapter("adapter2")
    mock_plugin_manager.load_adapter.return_value = mock_adapter2
    await xconnector.load_adapter(adapter_config2)

    # 启动connector
    await xconnector.start()

    # 获取健康状态
    health_status = await xconnector.get_health_status()

    # 验证返回状态正确
    assert health_status["connector"]["status"] == "healthy"
    assert health_status["adapters"]["inference"]["adapter1"]["status"] == "healthy"
    assert health_status["adapters"]["cache"]["adapter2"]["status"] == "healthy"


async def test_route_message(xconnector, mock_router):
    """测试消息路由"""
    # 设置路由器的mock返回值
    mock_router.route.return_value = "routed_result"

    # 路由消息
    result = await xconnector.route_message(
        "source_adapter", "target_adapter", "test_method", "arg1", kwarg1="value1"
    )

    # 验证返回值和调用参数
    assert result == "routed_result"
    mock_router.route.assert_called_once_with(
        "source_adapter", "target_adapter", "test_method", "arg1", kwarg1="value1"
    )


async def test_compatibility_properties(xconnector, mock_plugin_manager):
    """测试兼容性属性"""
    # 创建并加载多个模拟适配器
    adapter_config = AdapterConfig(
        name="vllm",
        type=AdapterType.INFERENCE,
        class_path="test.path",
        config={}
    )
    mock_adapter = MockAdapter("vllm")
    mock_plugin_manager.load_adapter.return_value = mock_adapter
    await xconnector.load_adapter(adapter_config)

    # 访问兼容性属性
    assert xconnector.vllm is mock_adapter

    # 测试不存在的适配器
    with pytest.raises(AttributeError):
        _ = xconnector.non_existent_adapter