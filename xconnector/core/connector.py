# xconnector/core/connector.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import torch

from xconnector.core.plugin_manager import PluginManager
from xconnector.core.router import Router
from xconnector.interfaces.base_interface import BaseInterface
from xconnector.utils.config import ConnectorConfig
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class AdapterType(Enum):
    """适配器类型枚举"""
    INFERENCE = "inference"
    CACHE = "cache"
    DISTRIBUTED = "distributed"


@dataclass
class AdapterConfig:
    """适配器配置"""
    name: str
    type: AdapterType
    class_path: str
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小优先级越高


class XConnector:
    """
    XConnector 主类 - 支持插件化架构的分布式推理缓存中间件

    功能特性：
    1. 插件化适配器管理
    2. 多对多连接路由
    3. 动态适配器加载
    4. 健康检查和监控
    5. 分布式协调
    """

    _instance = None
    _initialized = False

    def __new__(cls, config: Optional[ConnectorConfig] = None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[ConnectorConfig] = None):
        if self._initialized:
            return

        self.config = config or ConnectorConfig()

        # 原 XConnectorCore 的功能整合到这里
        self.plugin_manager = PluginManager()
        self.router = Router()

        # 适配器注册表
        self.inference_adapters: Dict[str, BaseInterface] = {}
        self.cache_adapters: Dict[str, BaseInterface] = {}
        self.distributed_adapters: Dict[str, BaseInterface] = {}

        # 原 core 的消息队列和任务管理功能
        self.connection_table: Dict[str, asyncio.Queue] = {}
        self.task_table: Dict[str, asyncio.Task] = {}

        # 运行时状态
        self.is_running = False
        self.health_check_task: Optional[asyncio.Task] = None

        # 初始化核心组件
        self._initialize_components()
        self._load_configured_adapters()

        self._initialized = True
        logger.info("XConnector initialized successfully")

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
        """创建端点的消息队列"""
        if endpoint not in self.connection_table:
            self.connection_table[endpoint] = asyncio.Queue(queue_size)
            logger.debug(f"Created endpoint queue: {endpoint}")

    async def send(self, endpoint: str, *args, **kwargs):
        """发送消息到指定端点"""
        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        await self.connection_table[endpoint].put((args, kwargs))

    async def receive(self, endpoint: str) -> Tuple[Tuple, Dict]:
        """从端点接收消息"""
        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        return await self.connection_table[endpoint].get()

    def start_task(self, name: str, coro):
        """启动后台任务"""
        if asyncio.iscoroutine(coro):
            self.task_table[name] = asyncio.create_task(coro)
        else:
            # 如果传入的是协程函数，需要调用它
            self.task_table[name] = asyncio.create_task(coro)
        logger.debug(f"Started task: {name}")

    def stop_task(self, name: str):
        """停止后台任务"""
        if name in self.task_table:
            self.task_table[name].cancel()
            del self.task_table[name]
            logger.debug(f"Stopped task: {name}")

    # === 核心功能 ===
    def _initialize_components(self):
        """初始化核心组件"""
        # 注册内置适配器
        self._register_builtin_adapters()

        # 配置路由规则
        self._setup_routing_rules()

        # 设置健康检查
        if self.config.enable_health_check:
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
            source_type=AdapterType.INFERENCE,
            target_type=AdapterType.CACHE,
            handler=self._handle_inference_to_cache
        )

        # 反向路由：cache -> inference
        self.router.add_route(
            source_type=AdapterType.CACHE,
            target_type=AdapterType.INFERENCE,
            handler=self._handle_cache_to_inference
        )

        # 分布式路由
        self.router.add_route(
            source_type=AdapterType.DISTRIBUTED,
            target_type=AdapterType.INFERENCE,
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
                self.load_adapter(adapter_config)

    # === 公共 API ===

    async def load_adapter(self, adapter_config: AdapterConfig) -> BaseInterface:
        """
        动态加载适配器

        Args:
            adapter_config: 适配器配置

        Returns:
            BaseInterface: 加载的适配器实例
        """
        try:
            # 确保插件管理器返回适配器实例
            adapter_instance = await self.plugin_manager.load_adapter(adapter_config, self.core)

            # 确保我们有一个有效的适配器实例
            if not adapter_instance:
                raise ValueError(f"PluginManager returned None for adapter: {adapter_config.name}")

            # 获取适配器类型的值
            adapter_type_value = adapter_config.type.value

            # 根据类型注册到相应的注册表
            if adapter_type_value == AdapterType.INFERENCE.value:
                # 确保适配器被正确添加到字典
                self.inference_adapters[adapter_config.name] = adapter_instance
            elif adapter_type_value == AdapterType.CACHE.value:
                self.cache_adapters[adapter_config.name] = adapter_instance
            elif adapter_type_value == AdapterType.DISTRIBUTED.value:
                self.distributed_adapters[adapter_config.name] = adapter_instance
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type_value}")

            logger.info(f"Loaded adapter: {adapter_config.name} ({adapter_type_value})")
            return adapter_instance

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_config.name}: {e}")
            raise

    def unload_adapter(self, adapter_name: str, adapter_type: AdapterType) -> Optional[BaseInterface]:
        """
        卸载适配器

        Args:
            adapter_name: 适配器名称
            adapter_type: 适配器类型

        Returns:
            Optional[BaseInterface]: 卸载的适配器实例
        """
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

        # 启动健康检查任务
        if hasattr(self, '_health_check_coro'):
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

        # 停止所有后台任务
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

        # 清理消息队列
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
                "adapters_count": {
                    "inference": len(self.inference_adapters),
                    "cache": len(self.cache_adapters),
                    "distributed": len(self.distributed_adapters)
                },
                "active_tasks": len(self.task_table),
                "active_endpoints": len(self.connection_table)
            },
            "adapters": {},
            "tasks": list(self.task_table.keys()),
            "endpoints": list(self.connection_table.keys())
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
            # 清理任务
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

def create_connector(config: Optional[ConnectorConfig] = None) -> XConnector:
    """
    创建 XConnector 实例的便捷函数

    Args:
        config: 连接器配置

    Returns:
        XConnector: 连接器实例
    """
    return XConnector(config)


def get_connector() -> XConnector:
    """
    获取全局 XConnector 实例

    Returns:
        XConnector: 全局连接器实例
    """
    return XConnector()


# === 异步上下文管理器支持 ===

class AsyncXConnector:
    """XConnector 的异步上下文管理器包装"""

    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.connector = XConnector(config)

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
        connector = create_connector()
        await connector.start()

        # 获取适配器
        vllm_adapter = connector.vllm
        lmcache_adapter = connector.lmcache

        # 路由消息
        if vllm_adapter and lmcache_adapter:
            result = await connector.route_message(
                "vllm", "lmcache", "save_kv_layer",
                "layer1", torch.randn(10, 10), None
            )

        # 检查健康状态
        health = await connector.get_health_status()
        print(f"Health status: {health}")

        await connector.stop()


    async def main_with_context():
        # 方式2: 使用上下文管理器
        async with AsyncXConnector() as connector:
            # 使用连接器
            adapters = connector.list_adapters()
            print(f"Available adapters: {adapters}")


    # 运行示例
    asyncio.run(main())