# integrations/dynamo/sdk/integration.py
"""
Dynamo SDK集成核心模块

提供XConnector与AI-Dynamo的深度集成，支持嵌入式部署和自动化配置。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import os

from xconnector.sdk import XConnectorSDK, SDKConfig, SDKMode
from xconnector.sdk.factory import create_dynamo_sdk
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


@dataclass
class DynamoIntegrationConfig:
    """Dynamo集成配置"""
    enabled: bool = True
    auto_inject: bool = True
    fail_on_error: bool = False

    # XConnector配置
    xconnector_config: Dict[str, Any] = None

    # 集成行为配置
    wrap_kv_methods: bool = True
    wrap_distributed_methods: bool = True
    enable_monitoring: bool = True

    # 错误处理配置
    graceful_degradation: bool = True
    fallback_to_original: bool = True


class DynamoXConnectorIntegration:
    """
    Dynamo与XConnector的SDK集成管理器

    负责XConnector SDK在Dynamo环境中的初始化、配置和生命周期管理
    """

    _instance: Optional['DynamoXConnectorIntegration'] = None

    def __init__(self, config: Optional[Union[Dict[str, Any], DynamoIntegrationConfig]] = None):
        """
        初始化集成管理器

        Args:
            config: 集成配置
        """
        # 配置处理
        if isinstance(config, dict):
            self.config = DynamoIntegrationConfig(**config)
        elif isinstance(config, DynamoIntegrationConfig):
            self.config = config
        else:
            self.config = DynamoIntegrationConfig()

        # 核心组件
        self.sdk: Optional[XConnectorSDK] = None
        self.dynamo_config: Dict[str, Any] = {}

        # 集成状态
        self.initialized = False
        self.injected_workers: List[Any] = []

        # 错误处理
        self._integration_errors: List[str] = []

        logger.info("DynamoXConnectorIntegration initialized")

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> 'DynamoXConnectorIntegration':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def initialize(self, dynamo_config: Dict[str, Any]) -> bool:
        """
        初始化集成

        Args:
            dynamo_config: Dynamo配置字典

        Returns:
            bool: 初始化是否成功
        """
        if self.initialized:
            logger.warning("Integration already initialized")
            return True

        if not self.config.enabled:
            logger.info("XConnector integration is disabled")
            return False

        try:
            logger.info("Initializing XConnector-Dynamo integration...")

            # 保存Dynamo配置
            self.dynamo_config = dynamo_config

            # 创建XConnector SDK
            if not await self._create_sdk():
                return False

            # 初始化SDK
            if not await self._initialize_sdk():
                return False

            # 启动SDK
            if not await self._start_sdk():
                return False

            self.initialized = True
            logger.info("XConnector-Dynamo integration initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Integration initialization failed: {e}"
            self._integration_errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

            if self.config.fail_on_error:
                raise
            return False

    def inject_into_worker(self, worker_instance: Any) -> bool:
        """
        将XConnector功能注入到Dynamo Worker中

        Args:
            worker_instance: Dynamo Worker实例

        Returns:
            bool: 注入是否成功
        """
        if not self.initialized or not self.sdk:
            logger.warning("Integration not initialized, skipping injection")
            return False

        try:
            logger.info(f"Injecting XConnector into worker: {type(worker_instance).__name__}")

            # 添加SDK引用到worker
            worker_instance.xconnector_sdk = self.sdk
            worker_instance.xconnector_enabled = True

            # 包装KV缓存方法
            if self.config.wrap_kv_methods:
                self._wrap_kv_methods(worker_instance)

            # 包装分布式方法
            if self.config.wrap_distributed_methods:
                self._wrap_distributed_methods(worker_instance)

            # 添加监控钩子
            if self.config.enable_monitoring:
                self._add_monitoring_hooks(worker_instance)

            # 记录已注入的worker
            self.injected_workers.append(worker_instance)

            logger.info(f"Successfully injected XConnector into worker")
            return True

        except Exception as e:
            error_msg = f"Worker injection failed: {e}"
            self._integration_errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

            if self.config.fail_on_error:
                raise
            return False

    async def cleanup(self) -> None:
        """清理集成资源"""
        try:
            logger.info("Cleaning up XConnector-Dynamo integration...")

            # 停止SDK
            if self.sdk:
                await self.sdk.stop()

            # 清理注入的workers
            for worker in self.injected_workers:
                self._cleanup_worker_injection(worker)

            self.injected_workers.clear()
            self.initialized = False

            logger.info("Integration cleanup completed")

        except Exception as e:
            logger.error(f"Integration cleanup failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return {
            "initialized": self.initialized,
            "enabled": self.config.enabled,
            "sdk_available": self.sdk is not None,
            "sdk_healthy": self.sdk.is_healthy() if self.sdk else False,
            "injected_workers": len(self.injected_workers),
            "errors": self._integration_errors,
            "config": {
                "auto_inject": self.config.auto_inject,
                "wrap_kv_methods": self.config.wrap_kv_methods,
                "graceful_degradation": self.config.graceful_degradation
            }
        }

    # === 内部实现方法 ===

    async def _create_sdk(self) -> bool:
        """创建XConnector SDK"""
        try:
            # 使用工厂函数创建SDK
            self.sdk = create_dynamo_sdk(
                self.dynamo_config,
                **(self.config.xconnector_config or {})
            )

            logger.info("XConnector SDK created")
            return True

        except Exception as e:
            logger.error(f"SDK creation failed: {e}")
            return False

    async def _initialize_sdk(self) -> bool:
        """初始化SDK"""
        try:
            success = await self.sdk.initialize()
            if success:
                logger.info("XConnector SDK initialized")
            else:
                logger.error("XConnector SDK initialization failed")
            return success

        except Exception as e:
            logger.error(f"SDK initialization failed: {e}")
            return False

    async def _start_sdk(self) -> bool:
        """启动SDK"""
        try:
            success = await self.sdk.start()
            if success:
                logger.info("XConnector SDK started")
            else:
                logger.error("XConnector SDK start failed")
            return success

        except Exception as e:
            logger.error(f"SDK start failed: {e}")
            return False

    def _wrap_kv_methods(self, worker_instance: Any) -> None:
        """包装KV缓存相关方法"""
        from integrations.dynamo.sdk.worker_wrapper import wrap_kv_cache_methods

        try:
            wrap_kv_cache_methods(worker_instance, self.sdk, self.config)
            logger.debug("KV cache methods wrapped successfully")

        except Exception as e:
            logger.error(f"KV methods wrapping failed: {e}")
            if self.config.fail_on_error:
                raise

    def _wrap_distributed_methods(self, worker_instance: Any) -> None:
        """包装分布式相关方法"""
        try:
            # 这里可以添加分布式方法的包装逻辑
            # 目前暂时跳过
            logger.debug("Distributed methods wrapping completed")

        except Exception as e:
            logger.error(f"Distributed methods wrapping failed: {e}")
            if self.config.fail_on_error:
                raise

    def _add_monitoring_hooks(self, worker_instance: Any) -> None:
        """添加监控钩子"""
        try:
            # 添加性能监控钩子
            if not hasattr(worker_instance, '_xconnector_monitoring'):
                worker_instance._xconnector_monitoring = {
                    'request_count': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'errors': 0
                }

            logger.debug("Monitoring hooks added successfully")

        except Exception as e:
            logger.error(f"Monitoring hooks setup failed: {e}")

    def _cleanup_worker_injection(self, worker_instance: Any) -> None:
        """清理worker注入"""
        try:
            # 移除XConnector相关属性
            if hasattr(worker_instance, 'xconnector_sdk'):
                delattr(worker_instance, 'xconnector_sdk')

            if hasattr(worker_instance, 'xconnector_enabled'):
                delattr(worker_instance, 'xconnector_enabled')

            logger.debug(f"Cleaned up injection for worker: {type(worker_instance).__name__}")

        except Exception as e:
            logger.warning(f"Worker cleanup failed: {e}")


# === 便捷函数 ===

def setup_xconnector_integration(
        worker_instance: Any,
        dynamo_config: Dict[str, Any],
        integration_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    便捷的集成设置函数

    Args:
        worker_instance: Dynamo Worker实例
        dynamo_config: Dynamo配置
        integration_config: 集成配置

    Returns:
        bool: 设置是否成功

    Example:
        >>> from integrations.dynamo.sdk.integration import setup_xconnector_integration
        >>>
        >>> def initialize_worker(config):
        ...     worker = VLLMWorker(config)
        ...
        ...     # 集成XConnector
        ...     if config.get('xconnector', {}).get('enabled', False):
        ...         setup_xconnector_integration(worker, config)
        ...
        ...     return worker
    """
    try:
        # 获取集成管理器实例
        integration = DynamoXConnectorIntegration.get_instance(integration_config)

        # 初始化集成（如果还未初始化）
        if not integration.initialized:
            # 由于这是同步函数，我们需要在异步上下文中调用
            import asyncio

            # 检查是否在事件循环中
            try:
                loop = asyncio.get_running_loop()
                # 如果在事件循环中，创建一个任务
                task = loop.create_task(integration.initialize(dynamo_config))
                # 注意：这里不能直接await，需要调用方处理
                logger.info("Created initialization task, will complete asynchronously")
            except RuntimeError:
                # 如果不在事件循环中，直接运行
                success = asyncio.run(integration.initialize(dynamo_config))
                if not success:
                    logger.error("Integration initialization failed")
                    return False

        # 注入到worker
        return integration.inject_into_worker(worker_instance)

    except Exception as e:
        logger.error(f"XConnector integration setup failed: {e}")
        return False


async def async_setup_xconnector_integration(
        worker_instance: Any,
        dynamo_config: Dict[str, Any],
        integration_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    异步版本的集成设置函数

    Args:
        worker_instance: Dynamo Worker实例
        dynamo_config: Dynamo配置
        integration_config: 集成配置

    Returns:
        bool: 设置是否成功
    """
    try:
        # 获取集成管理器实例
        integration = DynamoXConnectorIntegration.get_instance(integration_config)

        # 初始化集成
        if not integration.initialized:
            success = await integration.initialize(dynamo_config)
            if not success:
                logger.error("Integration initialization failed")
                return False

        # 注入到worker
        return integration.inject_into_worker(worker_instance)

    except Exception as e:
        logger.error(f"Async XConnector integration setup failed: {e}")
        return False


def get_integration_instance() -> Optional[DynamoXConnectorIntegration]:
    """获取集成管理器实例"""
    return DynamoXConnectorIntegration._instance


# === 环境检测和自动配置 ===

def detect_dynamo_environment() -> Dict[str, Any]:
    """检测Dynamo环境并返回建议配置"""
    config = {}

    # 检查环境变量
    if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
        config["enabled"] = True

    if os.getenv("XCONNECTOR_AUTO_INJECT", "").lower() == "true":
        config["auto_inject"] = True

    if os.getenv("XCONNECTOR_FAIL_ON_ERROR", "").lower() == "true":
        config["fail_on_error"] = True

    # 检测Dynamo版本
    try:
        import dynamo
        if hasattr(dynamo, '__version__'):
            config["dynamo_version"] = dynamo.__version__
        logger.info(f"Detected Dynamo environment: {config}")
    except ImportError:
        logger.warning("Dynamo not detected")

    return config


# === 导出 ===

__all__ = [
    'DynamoXConnectorIntegration',
    'DynamoIntegrationConfig',
    'setup_xconnector_integration',
    'async_setup_xconnector_integration',
    'get_integration_instance',
    'detect_dynamo_environment'
]