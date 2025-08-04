# xconnector/sdk/__init__.py
"""
XConnector SDK - 嵌入式集成模式

提供直接集成到推理框架（如Dynamo）的SDK接口，
替代原有的独立服务模式，实现更高效的进程内通信。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from xconnector.utils.xconnector_logging import get_logger
from xconnector.utils.xconnector_logging import get_logger
from xconnector.core.connector import XConnector
from xconnector.config import ConnectorConfig, AdapterConfig, AdapterType, SDKConfig, SDKMode

logger = get_logger(__name__)

class XConnectorSDK:
    """
    XConnector SDK主类

    提供嵌入式集成接口，支持直接在推理框架中使用XConnector功能
    """

    def __init__(self, config: Optional[Union[SDKConfig, Dict[str, Any]]] = None):
        """
        初始化SDK

        Args:
            config: SDK配置，可以是SDKConfig对象或字典
        """
        # 配置处理
        if isinstance(config, dict):
            self.config = SDKConfig(**config)
        elif isinstance(config, SDKConfig):
            self.config = config
        else:
            self.config = SDKConfig()

        # 核心组件
        self.connector: Optional[XConnector] = None
        self._handlers: Dict[str, Any] = {}

        # 状态管理
        self.initialized = False
        self.started = False

        # 错误处理
        self._error_count = 0
        self._last_error: Optional[Exception] = None

        logger.info(f"XConnectorSDK initialized in {self.config.mode.value} mode")

    async def initialize(self) -> bool:
        """
        初始化SDK和所有组件

        Returns:
            bool: 初始化是否成功
        """
        if self.initialized:
            logger.warning("SDK already initialized")
            return True

        try:
            logger.info("Initializing XConnector SDK...")

            # 创建核心连接器
            connector_config = self._create_connector_config()
            self.connector = XConnector(connector_config)

            # 初始化连接器
            if not await self._initialize_connector():
                return False

            # 加载适配器
            if not await self._load_adapters():
                return False

            # 初始化处理器
            if not await self._initialize_handlers():
                return False

            self.initialized = True
            logger.info("XConnector SDK initialized successfully")
            return True

        except Exception as e:
            self._handle_error(e, "SDK initialization failed")
            return False

    async def start(self) -> bool:
        """
        启动SDK服务

        Returns:
            bool: 启动是否成功
        """
        if not self.initialized:
            logger.error("SDK not initialized. Call initialize() first.")
            return False

        if self.started:
            logger.warning("SDK already started")
            return True

        try:
            logger.info("Starting XConnector SDK...")

            # 启动核心连接器
            await self.connector.start()

            # 启动处理器
            for name, handler in self._handlers.items():
                if hasattr(handler, 'start'):
                    await handler.start()

            self.started = True
            logger.info("XConnector SDK started successfully")
            return True

        except Exception as e:
            self._handle_error(e, "SDK start failed")
            return False

    async def stop(self) -> bool:
        """
        停止SDK服务

        Returns:
            bool: 停止是否成功
        """
        if not self.started:
            return True

        try:
            logger.info("Stopping XConnector SDK...")

            # 停止处理器
            for name, handler in self._handlers.items():
                if hasattr(handler, 'stop'):
                    try:
                        await handler.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping handler {name}: {e}")

            # 停止核心连接器
            if self.connector:
                await self.connector.stop()

            self.started = False
            logger.info("XConnector SDK stopped successfully")
            return True

        except Exception as e:
            self._handle_error(e, "SDK stop failed")
            return False

    def get_kv_handler(self):
        """获取KV缓存处理器"""
        if not self.config.enable_kv_cache:
            raise RuntimeError("KV cache is disabled in SDK config")

        return self._handlers.get('kv_cache')

    def get_distributed_handler(self):
        """获取分布式处理器"""
        if not self.config.enable_distributed:
            raise RuntimeError("Distributed processing is disabled in SDK config")

        return self._handlers.get('distributed')

    def get_monitoring_handler(self):
        """获取监控处理器"""
        if not self.config.enable_monitoring:
            raise RuntimeError("Monitoring is disabled in SDK config")

        return self._handlers.get('monitoring')

    async def route_message(self, source: str, target: str, method: str, *args, **kwargs) -> Any:
        """
        路由消息（兼容原有接口）

        Args:
            source: 源适配器名称
            target: 目标适配器名称
            method: 方法名称
            *args, **kwargs: 方法参数

        Returns:
            Any: 执行结果
        """
        if not self.started:
            raise RuntimeError("SDK not started. Call start() first.")

        try:
            return await self.connector.route_message(source, target, method, *args, **kwargs)
        except Exception as e:
            self._handle_error(e, f"Message routing failed: {source}->{target}::{method}")

            # 优雅降级
            if self.config.error_handling.get("graceful_degradation", True):
                logger.warning(f"Routing failed, attempting fallback for {source}->{target}::{method}")
                return await self._fallback_route(source, target, method, *args, **kwargs)
            else:
                raise

    def is_healthy(self) -> bool:
        """检查SDK健康状态"""
        if not self.initialized or not self.started:
            return False

        if self.connector and not self.connector.is_running:
            return False

        # 检查错误率
        if self._error_count > 10:  # 简单的错误阈值
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """获取SDK状态信息"""
        return {
            "initialized": self.initialized,
            "started": self.started,
            "healthy": self.is_healthy(),
            "mode": self.config.mode.value,
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None,
            "handlers": list(self._handlers.keys()),
            "connector_status": self.connector.get_status() if self.connector else None
        }

    # === 内部方法 ===

    def _create_connector_config(self) -> ConnectorConfig:
        """创建连接器配置"""
        # 转换SDK配置为连接器配置
        adapter_configs = []

        for adapter_cfg in self.config.adapters:
            adapter_config = AdapterConfig(
                name=adapter_cfg["name"],
                type=AdapterType(adapter_cfg["type"]),
                class_path=adapter_cfg["class_path"],
                config=adapter_cfg.get("config", {}),
                enabled=adapter_cfg.get("enabled", True),
                priority=adapter_cfg.get("priority", 0)
            )
            adapter_configs.append(adapter_config)

        # 创建连接器配置
        connector_config = ConnectorConfig()
        connector_config.adapters = adapter_configs

        return connector_config

    async def _initialize_connector(self) -> bool:
        """初始化核心连接器"""
        try:
            # 这里可以添加SDK特定的初始化逻辑
            return True
        except Exception as e:
            logger.error(f"Connector initialization failed: {e}")
            return False

    async def _load_adapters(self) -> bool:
        """加载适配器"""
        if not self.config.adapters:
            logger.info("No adapters configured")
            return True

        try:
            for adapter_cfg in self.config.adapters:
                adapter_config = AdapterConfig(
                    name=adapter_cfg["name"],
                    type=AdapterType(adapter_cfg["type"]),
                    class_path=adapter_cfg["class_path"],
                    config=adapter_cfg.get("config", {}),
                    enabled=adapter_cfg.get("enabled", True)
                )

                if adapter_config.enabled:
                    await self.connector.load_adapter(adapter_config)
                    logger.info(f"Loaded adapter: {adapter_config.name}")

            return True

        except Exception as e:
            logger.error(f"Adapter loading failed: {e}")
            return False

    async def _initialize_handlers(self) -> bool:
        """初始化功能处理器"""
        try:
            # 导入处理器（延迟导入避免循环依赖）
            from xconnector.sdk.handlers import (
                KVCacheHandler,
                DistributedHandler,
                MonitoringHandler
            )

            # 初始化KV缓存处理器
            if self.config.enable_kv_cache:
                self._handlers['kv_cache'] = KVCacheHandler(self)
                await self._handlers['kv_cache'].initialize()

            # 初始化分布式处理器
            if self.config.enable_distributed:
                self._handlers['distributed'] = DistributedHandler(self)
                await self._handlers['distributed'].initialize()

            # 初始化监控处理器
            if self.config.enable_monitoring:
                self._handlers['monitoring'] = MonitoringHandler(self)
                await self._handlers['monitoring'].initialize()

            return True

        except Exception as e:
            logger.error(f"Handlers initialization failed: {e}")
            return False

    async def _fallback_route(self, source: str, target: str, method: str, *args, **kwargs) -> Any:
        """降级路由处理"""
        if not self.config.error_handling.get("fallback_enabled", True):
            raise RuntimeError("Fallback routing is disabled")

        # 简单的降级逻辑：返回空结果
        logger.warning(f"Using fallback for {source}->{target}::{method}")

        if method in ["retrieve_kv", "recv_kv_caches"]:
            return {"found": False, "data": None}
        elif method in ["store_kv", "send_kv_caches"]:
            return True
        else:
            return None

    def _handle_error(self, error: Exception, context: str = ""):
        """处理错误"""
        self._error_count += 1
        self._last_error = error

        error_msg = f"{context}: {error}" if context else str(error)
        logger.error(error_msg, exc_info=True)

        # 错误隔离
        if self.config.error_handling.get("error_isolation", True):
            # 这里可以添加错误隔离逻辑
            pass

    # === 上下文管理器支持 ===

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()


# === 便捷导出 ===

__all__ = [
    'XConnectorSDK',
    'SDKConfig',
    'SDKMode',
]