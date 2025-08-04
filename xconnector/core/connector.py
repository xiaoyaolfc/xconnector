# xconnector/core/connector.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import threading
import importlib
import inspect

from xconnector.core.plugin_manager import PluginManager
from xconnector.core.router import Router
from xconnector.interfaces.base_interface import BaseInterface
from xconnector.config import ConnectorConfig, AdapterConfig, AdapterType
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)

class XConnector:
    """
    XConnector 主类 - 支持插件化架构的分布式推理缓存中间件

    功能特性：
    1. 插件化适配器管理
    2. 多对多连接路由
    3. 动态适配器加载
    4. 健康检查和监控
    5. 分布式协调
    6. SDK嵌入模式支持
    """

    _instance = None
    _initialized = False

    def __new__(cls, config: Optional[ConnectorConfig] = None, force_new: bool = False):
        # SDK模式下支持创建新实例
        if force_new or not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ConnectorConfig] = None, sdk_mode: bool = False):
        # SDK模式下允许重新初始化
        if self._initialized and not sdk_mode:
            return

        self.config = config or ConnectorConfig()
        self.sdk_mode = sdk_mode

        # 原 XConnectorCore 的功能整合到这里
        self.plugin_manager = PluginManager()
        self.router = Router()

        # 适配器注册表
        self.inference_adapters: Dict[str, BaseInterface] = {}
        self.cache_adapters: Dict[str, BaseInterface] = {}
        self.distributed_adapters: Dict[str, BaseInterface] = {}

        # 原 core 的消息队列和任务管理功能（SDK模式下简化）
        if not sdk_mode:
            self.connection_table: Dict[str, asyncio.Queue] = {}
            self.task_table: Dict[str, asyncio.Task] = {}
        else:
            # SDK模式下不需要这些功能
            self.connection_table = {}
            self.task_table = {}

        # 运行时状态
        self.is_running = False
        self.health_check_task: Optional[asyncio.Task] = None

        # 线程锁（用于适配器管理）
        self._lock = threading.RLock()

        # 已加载模块缓存（从PluginManager移过来）
        self._loaded_modules: Dict[str, Any] = {}

        # 初始化核心组件
        self._initialize_components()

        # 只在非SDK模式下自动加载适配器
        if not sdk_mode:
            self._load_configured_adapters()

        self._initialized = True
        mode_info = "SDK mode" if sdk_mode else "standalone mode"
        logger.info(f"XConnector initialized successfully in {mode_info}")

    # 原core方法，保持兼容性
    def register_vllm(self, adapter: BaseInterface):
        """兼容性方法：注册 VLLM 适配器"""
        self.inference_adapters["vllm"] = adapter
        self.router.register_adapter("vllm", adapter)

    def register_lmcache(self, adapter: BaseInterface):
        """兼容性方法：注册 LMCache 适配器"""
        self.cache_adapters["lmcache"] = adapter
        self.router.register_adapter("lmcache", adapter)

    async def route(self, endpoint: str, *args, **kwargs) -> Any:
        """
        路由消息到目标端点（原 core 方法）

        Args:
            endpoint: 端点路径，格式为 'adapter_type/method'
            *args, **kwargs: 方法参数
        """
        try:
            if '/' not in endpoint:
                raise ValueError(f"Invalid endpoint format: {endpoint}. Expected 'adapter_type/method'")

            adapter_type, method = endpoint.split('/', 1)

            # 根据适配器类型获取适配器
            adapter = None
            if adapter_type == 'vllm':
                adapter = self.get_adapter("vllm", AdapterType.INFERENCE)
            elif adapter_type == 'lmcache':
                adapter = self.get_adapter("lmcache", AdapterType.CACHE)
            elif adapter_type in self.inference_adapters:
                adapter = self.inference_adapters[adapter_type]
            elif adapter_type in self.cache_adapters:
                adapter = self.cache_adapters[adapter_type]
            elif adapter_type in self.distributed_adapters:
                adapter = self.distributed_adapters[adapter_type]

            if not adapter:
                raise ValueError(f"Adapter not found for type: {adapter_type}")

            handler = getattr(adapter, method, None)
            if not handler:
                raise AttributeError(f"Method '{method}' not found in adapter '{adapter_type}'")

            if asyncio.iscoroutinefunction(handler):
                return await handler(*args, **kwargs)
            else:
                return handler(*args, **kwargs)

        except Exception as e:
            logger.error(f"Route failed for endpoint {endpoint}: {e}")
            raise

    def create_endpoint(self, endpoint: str, queue_size: int = 100):
        """创建端点的消息队列（仅非SDK模式）"""
        if self.sdk_mode:
            return  # SDK模式下不需要队列

        if endpoint not in self.connection_table:
            self.connection_table[endpoint] = asyncio.Queue(queue_size)
            logger.debug(f"Created endpoint queue: {endpoint}")

    async def send(self, endpoint: str, *args, **kwargs):
        """发送消息到指定端点（仅非SDK模式）"""
        if self.sdk_mode:
            return  # SDK模式下不需要队列

        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        await self.connection_table[endpoint].put((args, kwargs))

    async def receive(self, endpoint: str) -> Tuple[Tuple, Dict]:
        """从端点接收消息（仅非SDK模式）"""
        if self.sdk_mode:
            return ((), {})  # SDK模式下返回空

        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        return await self.connection_table[endpoint].get()

    def start_task(self, name: str, coro):
        """启动后台任务（仅非SDK模式）"""
        if self.sdk_mode:
            return  # SDK模式下不需要任务管理

        if asyncio.iscoroutine(coro):
            self.task_table[name] = asyncio.create_task(coro)
        else:
            # 如果传入的是协程函数，需要调用它
            self.task_table[name] = asyncio.create_task(coro)
        logger.debug(f"Started task: {name}")

    def stop_task(self, name: str):
        """停止后台任务（仅非SDK模式）"""
        if self.sdk_mode:
            return  # SDK模式下不需要任务管理

        if name in self.task_table:
            self.task_table[name].cancel()
            del self.task_table[name]
            logger.debug(f"Stopped task: {name}")

    # === 核心功能 ===
    def _initialize_components(self):
        """初始化核心组件"""
        # 注册内置适配器（仅在非SDK模式下）
        if not self.sdk_mode:
            self._register_builtin_adapters()

        # 配置路由规则
        self._setup_routing_rules()

        # 设置健康检查（仅在非SDK模式下）
        if not self.sdk_mode and self.config.enable_health_check:
            self._setup_health_check()

    def _register_builtin_adapters(self):
        """注册内置适配器（仅注册到插件管理器，不实际加载）"""
        try:
            # 只注册适配器配置，不传入具体配置
            # 具体配置由各适配器自己处理
            builtin_adapters = [
                AdapterConfig(
                    name="vllm",
                    type=AdapterType.INFERENCE,
                    class_path="xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
                    config={},  # 空配置，由适配器自己处理
                    enabled=False  # 默认不启用，需要手动加载
                ),
                AdapterConfig(
                    name="lmcache",
                    type=AdapterType.CACHE,
                    class_path="xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                    config={},  # 空配置，由适配器自己处理
                    enabled=False  # 默认不启用，需要手动加载
                )
            ]

            # 只注册到插件管理器，不实际加载
            for adapter_config in builtin_adapters:
                self.plugin_manager.register_adapter(adapter_config)

            logger.debug("Registered builtin adapters to plugin manager")

        except Exception as e:
            logger.warning(f"Failed to register builtin adapters: {e}")

    def _setup_routing_rules(self):
        """设置路由规则"""
        # 默认路由规则：inference -> cache
        self.router.add_route(
            source_type="inference",
            target_type="cache",
            handler=self._handle_inference_to_cache
        )

        # 反向路由：cache -> inference
        self.router.add_route(
            source_type="cache",
            target_type="inference",
            handler=self._handle_cache_to_inference
        )

        # 分布式路由
        self.router.add_route(
            source_type="distributed",
            target_type="inference",
            handler=self._handle_distributed_to_inference
        )

    def _setup_health_check(self):
        """设置健康检查任务（不直接启动）"""
        if self.config.enable_health_check:
            # 保存健康检查循环的协程，但不启动
            self._health_check_coro = self._perform_health_check()

    def _load_configured_adapters(self):
        """加载配置文件中的适配器"""
        for adapter_config in self.config.adapters:
            if adapter_config.enabled:
                asyncio.create_task(self.load_adapter(adapter_config))

    # === 公共 API ===

    async def load_adapter(self, adapter_config: AdapterConfig) -> BaseInterface:
        """
        动态加载适配器

        Args:
            adapter_config: 适配器配置

        Returns:
            BaseInterface: 加载的适配器实例
        """
        with self._lock:
            adapter_name = adapter_config.name

            try:
                logger.info(f"Loading adapter: {adapter_name}")

                # 检查是否已经加载
                if adapter_name in self._loaded_modules:
                    logger.warning(f"Adapter {adapter_name} is already loaded")

                # 根据类路径加载类
                adapter_class = self._load_class_from_path(adapter_config.class_path)

                # 创建适配器实例（SDK模式下传入当前实例）
                if self.sdk_mode:
                    adapter_instance = adapter_class(self, adapter_config.config)
                else:
                    adapter_instance = adapter_class(adapter_config.config)

                # 验证适配器接口
                if not isinstance(adapter_instance, BaseInterface):
                    raise TypeError(f"Adapter {adapter_name} must implement BaseInterface")

                # 缓存实例
                self._loaded_modules[adapter_name] = adapter_instance

                # 根据类型注册到相应的注册表
                adapter_type_value = adapter_config.type.value
                if adapter_type_value == AdapterType.INFERENCE.value:
                    self.inference_adapters[adapter_name] = adapter_instance
                elif adapter_type_value == AdapterType.CACHE.value:
                    self.cache_adapters[adapter_name] = adapter_instance
                elif adapter_type_value == AdapterType.DISTRIBUTED.value:
                    self.distributed_adapters[adapter_name] = adapter_instance
                else:
                    raise ValueError(f"Unknown adapter type: {adapter_type_value}")

                # 注册到路由器
                if hasattr(self, 'router'):
                    self.router.register_adapter(adapter_name, adapter_instance)

                logger.info(f"Successfully loaded adapter: {adapter_name} ({adapter_type_value})")
                return adapter_instance

            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_name}: {e}")
                raise RuntimeError(f"Failed to load adapter {adapter_name}: {e}") from e

    def _load_class_from_path(self, class_path: str) -> Type[BaseInterface]:
        """
        从类路径加载类

        Args:
            class_path: 类路径 (例如: "xconnector.adapters.vllm.VLLMAdapter")

        Returns:
            Type[BaseInterface]: 加载的类
        """
        try:
            # 分离模块路径和类名
            module_path, class_name = class_path.rsplit(".", 1)

            # 导入模块
            module = importlib.import_module(module_path)

            # 获取类
            adapter_class = getattr(module, class_name)

            if not inspect.isclass(adapter_class):
                raise TypeError(f"{class_path} is not a class")

            return adapter_class

        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Cannot import {class_path}: {e}") from e

    def unload_adapter(self, adapter_name: str, adapter_type: AdapterType) -> Optional[BaseInterface]:
        """
        卸载适配器

        Args:
            adapter_name: 适配器名称
            adapter_type: 适配器类型

        Returns:
            Optional[BaseInterface]: 卸载的适配器实例
        """
        with self._lock:
            try:
                adapter = None
                adapter_type_value = adapter_type.value

                # 从正确的字典中移除适配器
                if adapter_type_value == AdapterType.INFERENCE.value:
                    adapter = self.inference_adapters.pop(adapter_name, None)
                elif adapter_type_value == AdapterType.CACHE.value:
                    adapter = self.cache_adapters.pop(adapter_name, None)
                elif adapter_type_value == AdapterType.DISTRIBUTED.value:
                    adapter = self.distributed_adapters.pop(adapter_name, None)
                else:
                    logger.error(f"Unknown adapter type: {adapter_type_value}")
                    return None

                # 从缓存中移除
                if adapter_name in self._loaded_modules:
                    del self._loaded_modules[adapter_name]

                # 执行清理操作
                if adapter:
                    if hasattr(adapter, 'cleanup'):
                        adapter.cleanup()

                    if hasattr(adapter, 'stop') and callable(adapter.stop):
                        # 对于异步方法，我们只记录警告，不实际调用
                        if asyncio.iscoroutinefunction(adapter.stop):
                            logger.warning(f"Adapter {adapter_name} has async stop method, "
                                           "but it cannot be called from unload_adapter. "
                                           "Please call stop separately before unloading.")
                        else:
                            # 如果是同步方法，直接调用
                            adapter.stop()

                logger.info(f"Unloaded adapter: {adapter_name} ({adapter_type_value})")
                return adapter

            except Exception as e:
                logger.error(f"Failed to unload adapter {adapter_name}: {e}")
                raise

    def get_adapter(self, adapter_name: str, adapter_type: AdapterType) -> Optional[BaseInterface]:
        """
        获取适配器实例

        Args:
            adapter_name: 适配器名称
            adapter_type: 适配器类型

        Returns:
            Optional[BaseInterface]: 适配器实例，不存在则返回None
        """
        try:
            adapter_type_value = adapter_type.value

            # 添加调试日志
            logger.debug(f"Getting adapter: {adapter_name} of type {adapter_type_value}")

            if adapter_type_value == AdapterType.INFERENCE.value:
                logger.debug(f"Inference adapters: {list(self.inference_adapters.keys())}")
                return self.inference_adapters.get(adapter_name)
            elif adapter_type_value == AdapterType.CACHE.value:
                logger.debug(f"Cache adapters: {list(self.cache_adapters.keys())}")
                return self.cache_adapters.get(adapter_name)
            elif adapter_type_value == AdapterType.DISTRIBUTED.value:
                logger.debug(f"Distributed adapters: {list(self.distributed_adapters.keys())}")
                return self.distributed_adapters.get(adapter_name)
            else:
                logger.error(f"Unknown adapter type: {adapter_type_value}")
                return None
        except Exception as e:
            logger.error(f"Failed to get adapter {adapter_name}: {e}")
            return None

    def list_adapters(self) -> Dict[str, List[str]]:
        """
        列出所有已加载的适配器

        Returns:
            Dict[str, List[str]]: 按类型分组的适配器列表
        """
        return {
            "inference": list(self.inference_adapters.keys()),
            "cache": list(self.cache_adapters.keys()),
            "distributed": list(self.distributed_adapters.keys())
        }

    async def route_message(self, source: str, target: str, method: str, *args, **kwargs) -> Any:
        """
        路由消息到目标适配器

        Args:
            source: 源适配器名称
            target: 目标适配器名称
            method: 调用方法名
            *args, **kwargs: 方法参数

        Returns:
            Any: 方法执行结果
        """
        try:
            # 方式1: 使用 router 进行路由（推荐）
            return await self.router.route(source, target, method, *args, **kwargs)
        except Exception as router_error:
            logger.warning(f"Router failed, trying direct route: {router_error}")

    async def start(self):
        """启动 XConnector"""
        if self.is_running:
            logger.warning("XConnector is already running")
            return

        self.is_running = True

        # 启动健康检查任务（仅非SDK模式）
        if not self.sdk_mode and hasattr(self, '_health_check_coro'):
            self.health_check_task = asyncio.create_task(self._health_check_coro)

        # 添加调试日志
        logger.debug("Starting XConnector...")
        logger.debug(f"Inference adapters: {list(self.inference_adapters.keys())}")
        logger.debug(f"Cache adapters: {list(self.cache_adapters.keys())}")
        logger.debug(f"Distributed adapters: {list(self.distributed_adapters.keys())}")

        # 启动所有适配器
        for adapters in [self.inference_adapters, self.cache_adapters, self.distributed_adapters]:
            for name, adapter in adapters.items():
                if adapter is None:
                    logger.error(f"Adapter {name} is None, skipping start")
                    continue

                if hasattr(adapter, 'start'):
                    logger.debug(f"Starting adapter: {name}")
                    await adapter.start()
                else:
                    logger.warning(f"Adapter {name} has no start method")

        logger.info("XConnector started successfully")

    async def stop(self):
        """停止 XConnector（增强版本）"""
        if not self.is_running:
            logger.warning("XConnector is not running")
            return

        self.is_running = False

        # 停止所有后台任务（仅非SDK模式）
        if not self.sdk_mode:
            for task_name in list(self.task_table.keys()):
                self.stop_task(task_name)

        # 停止健康检查
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None

        # 停止所有适配器
        for adapters in [self.inference_adapters, self.cache_adapters, self.distributed_adapters]:
            for name, adapter in adapters.items():
                if adapter is None:
                    continue

                if hasattr(adapter, 'stop'):
                    try:
                        await adapter.stop()
                    except Exception as e:
                        logger.error(f"Error stopping adapter {name}: {e}")

        # 清理消息队列（仅非SDK模式）
        if not self.sdk_mode:
            for endpoint, queue in self.connection_table.items():
                try:
                    # 清空队列
                    while not queue.empty():
                        queue.get_nowait()
                except Exception:
                    pass

            self.connection_table.clear()

        logger.info("XConnector stopped successfully")

    async def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态（增强版本）
        """
        status = {
            "connector": {
                "status": "healthy" if self.is_running else "stopped",
                "mode": "SDK" if self.sdk_mode else "standalone",
                "adapters_count": {
                    "inference": len(self.inference_adapters),
                    "cache": len(self.cache_adapters),
                    "distributed": len(self.distributed_adapters)
                },
                "active_tasks": len(self.task_table) if not self.sdk_mode else 0,
                "active_endpoints": len(self.connection_table) if not self.sdk_mode else 0
            },
            "adapters": {},
            "tasks": list(self.task_table.keys()) if not self.sdk_mode else [],
            "endpoints": list(self.connection_table.keys()) if not self.sdk_mode else []
        }

        # 检查各个适配器的健康状态
        for adapter_type, adapters in [
            ("inference", self.inference_adapters),
            ("cache", self.cache_adapters),
            ("distributed", self.distributed_adapters)
        ]:
            status["adapters"][adapter_type] = {}
            for name, adapter in adapters.items():
                if hasattr(adapter, 'health_check'):
                    try:
                        adapter_status = await adapter.health_check()
                        status["adapters"][adapter_type][name] = adapter_status
                    except Exception as e:
                        status["adapters"][adapter_type][name] = {
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    status["adapters"][adapter_type][name] = {
                        "status": "unknown",
                        "message": "Health check not implemented"
                    }

        return status

    def cleanup(self):
        """清理所有资源"""
        try:
            # 清理任务（仅非SDK模式）
            if not self.sdk_mode:
                for task_name in list(self.task_table.keys()):
                    self.stop_task(task_name)

                # 清理队列
                self.connection_table.clear()

            # 清理适配器
            for adapters in [self.inference_adapters, self.cache_adapters, self.distributed_adapters]:
                for adapter in adapters.values():
                    if adapter and hasattr(adapter, 'cleanup'):
                        try:
                            adapter.cleanup()
                        except Exception as e:
                            logger.error(f"Error during adapter cleanup: {e}")

            logger.info("XConnector cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 忽略析构函数中的错误

    # === 兼容性属性 (保持与旧版本的兼容) ===

    @property
    def vllm(self) -> Optional[BaseInterface]:
        """获取 VLLM 适配器 (兼容性属性)"""
        return self.get_adapter("vllm", AdapterType.INFERENCE)

    @property
    def lmcache(self) -> Optional[BaseInterface]:
        """获取 LMCache 适配器 (兼容性属性)"""
        return self.get_adapter("lmcache", AdapterType.CACHE)

    # === 路由处理器 ===

    async def _handle_inference_to_cache(self, source_adapter: BaseInterface,
                                         target_adapter: BaseInterface,
                                         method: str, *args, **kwargs) -> Any:
        """处理推理引擎到缓存的路由"""
        if hasattr(target_adapter, method):
            return await getattr(target_adapter, method)(*args, **kwargs)
        else:
            raise AttributeError(f"Method {method} not found in cache adapter")

    async def _handle_cache_to_inference(self, source_adapter: BaseInterface,
                                         target_adapter: BaseInterface,
                                         method: str, *args, **kwargs) -> Any:
        """处理缓存到推理引擎的路由"""
        if hasattr(target_adapter, method):
            return await getattr(target_adapter, method)(*args, **kwargs)
        else:
            raise AttributeError(f"Method {method} not found in inference adapter")

    async def _handle_distributed_to_inference(self, source_adapter: BaseInterface,
                                               target_adapter: BaseInterface,
                                               method: str, *args, **kwargs) -> Any:
        """处理分布式到推理引擎的路由"""
        if hasattr(target_adapter, method):
            return await getattr(target_adapter, method)(*args, **kwargs)
        else:
            raise AttributeError(f"Method {method} not found in inference adapter")

    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            health_status = await self.get_health_status()

            # 检查是否有不健康的适配器
            unhealthy_adapters = []
            for adapter_type, adapters in health_status["adapters"].items():
                for name, status in adapters.items():
                    if status.get("status") != "healthy":
                        unhealthy_adapters.append(f"{adapter_type}.{name}")

            if unhealthy_adapters:
                logger.warning(f"Unhealthy adapters detected: {unhealthy_adapters}")

            # 记录健康检查结果
            if self.config.log_health_check:
                logger.debug(f"Health check completed: {len(unhealthy_adapters)} unhealthy adapters")

        except Exception as e:
            logger.error(f"Health check failed: {e}")


# === 便捷工厂函数 ===

def create_connector(config: Optional[ConnectorConfig] = None, sdk_mode: bool = False) -> XConnector:
    """
    创建 XConnector 实例的便捷函数

    Args:
        config: 连接器配置
        sdk_mode: 是否为SDK模式

    Returns:
        XConnector: 连接器实例
    """
    return XConnector(config, sdk_mode=sdk_mode)


def get_connector(sdk_mode: bool = False) -> XConnector:
    """
    获取全局 XConnector 实例

    Args:
        sdk_mode: 是否为SDK模式

    Returns:
        XConnector: 全局连接器实例
    """
    return XConnector(sdk_mode=sdk_mode)


# === 异步上下文管理器支持 ===

class AsyncXConnector:
    """XConnector 的异步上下文管理器包装"""

    def __init__(self, config: Optional[ConnectorConfig] = None, sdk_mode: bool = False):
        self.connector = XConnector(config, sdk_mode=sdk_mode)

    async def __aenter__(self):
        await self.connector.start()
        return self.connector

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connector.stop()


# === 使用示例 ===

if __name__ == "__main__":
    import asyncio


    async def main():
        # 方式1: 直接使用
        connector = create_connector(sdk_mode=True)
        await connector.start()

        # 获取适配器
        vllm_adapter = connector.vllm
        lmcache_adapter = connector.lmcache

        # 路由消息
        if vllm_adapter and lmcache_adapter:
            result = await connector.route_message(
                "vllm", "lmcache", "store_kv",
                "test_input", torch.randn(10, 10), torch.randn(5, 5)
            )

        # 检查健康状态
        health = await connector.get_health_status()
        print(f"Health status: {health}")

        await connector.stop()


    async def main_with_context():
        # 方式2: 使用上下文管理器
        async with AsyncXConnector(sdk_mode=True) as connector:
            # 使用连接器
            adapters = connector.list_adapters()
            print(f"Available adapters: {adapters}")


    # 运行示例
    asyncio.run(main())