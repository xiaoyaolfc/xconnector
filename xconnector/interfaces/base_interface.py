# xconnector/interfaces/base_interface.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from xconnector.utils.logging import get_logger

logger = get_logger(__name__)


class AdapterStatus(Enum):
    """适配器状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class AdapterMetrics:
    """适配器指标数据"""
    timestamp: datetime
    request_count: int
    error_count: int
    avg_response_time: float
    memory_usage: float
    cpu_usage: float
    custom_metrics: Dict[str, Any]


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


@dataclass
class Capability:
    """功能能力描述"""
    name: str
    description: str
    version: str
    supported: bool
    parameters: Dict[str, Any]


class BaseInterface(ABC):
    """
    XConnector 适配器基础接口

    所有适配器必须继承此接口，并实现相应的抽象方法
    """

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        """
        初始化适配器

        Args:
            core_instance: XConnector 核心实例
            config: 适配器配置
        """
        self.core = core_instance
        self.config = config or {}
        self.status = AdapterStatus.INITIALIZING
        self.last_health_check = None
        self.metrics_history: List[AdapterMetrics] = []
        self.error_log: List[Dict[str, Any]] = []

        # 适配器元信息
        self.adapter_name = self.__class__.__name__
        self.adapter_version = getattr(self, '__version__', '1.0.0')
        self.adapter_description = getattr(self, '__doc__', '')
        self.adapter_author = getattr(self, '__author__', 'Unknown')
        self.adapter_dependencies = getattr(self, '__dependencies__', [])

        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}

        # 初始化日志
        self.logger = get_logger(f"{__name__}.{self.adapter_name}")

        self.logger.info(f"Initializing adapter: {self.adapter_name}")

    # === 抽象方法 - 必须实现 ===

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化适配器

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def start(self) -> bool:
        """
        启动适配器

        Returns:
            bool: 启动是否成功
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        停止适配器

        Returns:
            bool: 停止是否成功
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Capability]:
        """
        返回适配器支持的功能

        Returns:
            Dict[str, Capability]: 功能能力字典
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        健康检查

        Returns:
            HealthCheckResult: 健康检查结果
        """
        pass

    @abstractmethod
    def get_metrics(self) -> AdapterMetrics:
        """
        获取监控指标

        Returns:
            AdapterMetrics: 监控指标数据
        """
        pass

    # === 可选实现的方法 ===

    async def pause(self) -> bool:
        """
        暂停适配器

        Returns:
            bool: 暂停是否成功
        """
        try:
            self.status = AdapterStatus.PAUSED
            self.logger.info(f"Adapter {self.adapter_name} paused")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pause adapter {self.adapter_name}: {e}")
            return False

    async def resume(self) -> bool:
        """
        恢复适配器

        Returns:
            bool: 恢复是否成功
        """
        try:
            self.status = AdapterStatus.RUNNING
            self.logger.info(f"Adapter {self.adapter_name} resumed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to resume adapter {self.adapter_name}: {e}")
            return False

    async def restart(self) -> bool:
        """
        重启适配器

        Returns:
            bool: 重启是否成功
        """
        try:
            await self.stop()
            await asyncio.sleep(1)  # 等待完全停止
            return await self.start()
        except Exception as e:
            self.logger.error(f"Failed to restart adapter {self.adapter_name}: {e}")
            return False

    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.status = AdapterStatus.STOPPED
            self.event_callbacks.clear()
            self.logger.info(f"Adapter {self.adapter_name} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup of adapter {self.adapter_name}: {e}")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证配置

        Args:
            config: 配置字典

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误消息)
        """
        try:
            # 默认实现：检查必需的配置项
            required_fields = getattr(self, '_required_config_fields', [])

            for field in required_fields:
                if field not in config:
                    return False, f"Missing required config field: {field}"

            return True, None
        except Exception as e:
            return False, f"Config validation error: {e}"

    def get_status(self) -> AdapterStatus:
        """获取适配器状态"""
        return self.status

    def get_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            "name": self.adapter_name,
            "version": self.adapter_version,
            "description": self.adapter_description,
            "author": self.adapter_author,
            "dependencies": self.adapter_dependencies,
            "status": self.status.value,
            "config": self.config,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }

    def get_error_log(self) -> List[Dict[str, Any]]:
        """获取错误日志"""
        return self.error_log.copy()

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """记录错误"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }

        self.error_log.append(error_entry)

        # 限制错误日志大小
        if len(self.error_log) > 100:
            self.error_log.pop(0)

        self.logger.error(f"Error in adapter {self.adapter_name}: {error}", exc_info=True)

    # === 事件系统 ===

    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        注册事件回调

        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []

        self.event_callbacks[event_name].append(callback)
        self.logger.debug(f"Registered callback for event: {event_name}")

    def unregister_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        取消注册事件回调

        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name in self.event_callbacks:
            try:
                self.event_callbacks[event_name].remove(callback)
                self.logger.debug(f"Unregistered callback for event: {event_name}")
            except ValueError:
                pass

    async def emit_event(self, event_name: str, **kwargs) -> None:
        """
        触发事件

        Args:
            event_name: 事件名称
            **kwargs: 事件参数
        """
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self, event_name, **kwargs)
                    else:
                        callback(self, event_name, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_name}: {e}")

    # === 指标收集 ===

    def update_metrics(self, metrics: AdapterMetrics) -> None:
        """更新指标历史"""
        self.metrics_history.append(metrics)

        # 限制历史记录大小
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

    def get_metrics_history(self, limit: int = 100) -> List[AdapterMetrics]:
        """获取指标历史"""
        return self.metrics_history[-limit:]

    def get_average_metrics(self, duration_minutes: int = 60) -> Optional[Dict[str, float]]:
        """获取指定时间段内的平均指标"""
        if not self.metrics_history:
            return None

        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp.timestamp() > cutoff_time
        ]

        if not recent_metrics:
            return None

        return {
            "avg_response_time": sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "total_requests": sum(m.request_count for m in recent_metrics),
            "total_errors": sum(m.error_count for m in recent_metrics)
        }

    # === 配置管理 ===

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新配置

        Args:
            new_config: 新配置

        Returns:
            bool: 更新是否成功
        """
        try:
            # 验证新配置
            is_valid, error_message = self.validate_config(new_config)
            if not is_valid:
                self.logger.error(f"Invalid config: {error_message}")
                return False

            # 备份旧配置
            old_config = self.config.copy()

            # 更新配置
            self.config.update(new_config)

            # 触发配置更新事件
            asyncio.create_task(self.emit_event("config_updated",
                                                old_config=old_config,
                                                new_config=self.config))

            self.logger.info(f"Config updated for adapter {self.adapter_name}")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "update_config"})
            return False

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    # === 钩子方法 ===

    async def before_start(self) -> None:
        """启动前钩子"""
        pass

    async def after_start(self) -> None:
        """启动后钩子"""
        pass

    async def before_stop(self) -> None:
        """停止前钩子"""
        pass

    async def after_stop(self) -> None:
        """停止后钩子"""
        pass

    # === 工具方法 ===

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return (self.status in [AdapterStatus.READY, AdapterStatus.RUNNING] and
                self.last_health_check is not None)

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.status == AdapterStatus.RUNNING

    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self.status in [AdapterStatus.READY, AdapterStatus.RUNNING]

    def __str__(self) -> str:
        return f"{self.adapter_name} (v{self.adapter_version})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.adapter_name} status={self.status.value}>"