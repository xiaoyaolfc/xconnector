# xconnector/adapters/base_adapter.py
import asyncio
import psutil
import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from xconnector.interfaces.base_interface import (
    BaseInterface,
    AdapterStatus,
    HealthStatus,
    AdapterMetrics,
    HealthCheckResult,
    Capability
)
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class BaseAdapter(BaseInterface):
    """
    基础适配器实现类

    提供通用的适配器功能实现，具体适配器可以继承此类
    并重写特定的方法来实现自定义逻辑
    """

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # 请求统计
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        self._start_time = time.time()

        # 进程监控
        self._process = psutil.Process()

        # 初始化完成
        self.logger.info(f"BaseAdapter {self.adapter_name} initialized")

    # === 通用实现方法 ===

    async def initialize(self) -> bool:
        """
        默认初始化实现

        子类可以重写此方法来添加特定的初始化逻辑
        """
        try:
            self.status = AdapterStatus.INITIALIZING
            await self.emit_event("initializing")

            # 验证配置
            is_valid, error_msg = self.validate_config(self.config)
            if not is_valid:
                self.logger.error(f"Config validation failed: {error_msg}")
                self.status = AdapterStatus.ERROR
                return False

            # 调用子类初始化
            if hasattr(self, '_initialize_impl'):
                success = await self._initialize_impl()
                if not success:
                    self.status = AdapterStatus.ERROR
                    return False

            self.status = AdapterStatus.READY
            await self.emit_event("initialized")
            self.logger.info(f"Adapter {self.adapter_name} initialized successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "initialize"})
            self.status = AdapterStatus.ERROR
            return False

    async def start(self) -> bool:
        """
        默认启动实现
        """
        try:
            if self.status == AdapterStatus.RUNNING:
                self.logger.warning(f"Adapter {self.adapter_name} is already running")
                return True

            await self.before_start()
            self.status = AdapterStatus.RUNNING

            # 调用子类启动逻辑
            if hasattr(self, '_start_impl'):
                success = await self._start_impl()
                if not success:
                    self.status = AdapterStatus.ERROR
                    return False

            await self.after_start()
            await self.emit_event("started")
            self.logger.info(f"Adapter {self.adapter_name} started successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "start"})
            self.status = AdapterStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        默认停止实现
        """
        try:
            if self.status == AdapterStatus.STOPPED:
                self.logger.warning(f"Adapter {self.adapter_name} is already stopped")
                return True

            await self.before_stop()
            self.status = AdapterStatus.STOPPING

            # 调用子类停止逻辑
            if hasattr(self, '_stop_impl'):
                success = await self._stop_impl()
                if not success:
                    self.logger.warning(f"Stop implementation returned False for {self.adapter_name}")

            self.status = AdapterStatus.STOPPED
            await self.after_stop()
            await self.emit_event("stopped")
            self.logger.info(f"Adapter {self.adapter_name} stopped successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "stop"})
            self.status = AdapterStatus.ERROR
            return False

    def get_capabilities(self) -> Dict[str, Capability]:
        """
        默认功能能力实现

        子类应该重写此方法来定义具体的功能能力
        """
        return {
            "basic": Capability(
                name="basic",
                description="Basic adapter functionality",
                version="1.0.0",
                supported=True,
                parameters={}
            )
        }

    async def health_check(self) -> HealthCheckResult:
        """
        默认健康检查实现
        """
        try:
            # 检查适配器状态
            if self.status == AdapterStatus.ERROR:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Adapter is in error state",
                    timestamp=datetime.now()
                )

            if self.status not in [AdapterStatus.READY, AdapterStatus.RUNNING]:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Adapter status: {self.status.value}",
                    timestamp=datetime.now()
                )

            # 调用子类健康检查
            if hasattr(self, '_health_check_impl'):
                result = await self._health_check_impl()
                if result:
                    self.last_health_check = datetime.now()
                    return result

            # 默认健康检查
            self.last_health_check = datetime.now()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Adapter is healthy",
                timestamp=self.last_health_check,
                details={
                    "status": self.status.value,
                    "uptime": time.time() - self._start_time,
                    "request_count": self._request_count,
                    "error_rate": self._error_count / max(self._request_count, 1)
                }
            )

        except Exception as e:
            self.log_error(e, {"operation": "health_check"})
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.now()
            )

    def get_metrics(self) -> AdapterMetrics:
        """
        默认指标获取实现
        """
        try:
            # 获取系统资源使用情况
            memory_info = self._process.memory_info()
            cpu_percent = self._process.cpu_percent()

            # 计算平均响应时间
            avg_response_time = (
                    self._total_response_time / max(self._request_count, 1)
            )

            metrics = AdapterMetrics(
                timestamp=datetime.now(),
                request_count=self._request_count,
                error_count=self._error_count,
                avg_response_time=avg_response_time,
                memory_usage=memory_info.rss / 1024 / 1024,  # MB
                cpu_usage=cpu_percent,
                custom_metrics=self._get_custom_metrics()
            )

            self.update_metrics(metrics)
            return metrics

        except Exception as e:
            self.log_error(e, {"operation": "get_metrics"})
            # 返回空的指标对象
            return AdapterMetrics(
                timestamp=datetime.now(),
                request_count=0,
                error_count=1,
                avg_response_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                custom_metrics={"error": str(e)}
            )

    # === 工具方法 ===

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """
        获取自定义指标

        子类可以重写此方法来提供特定的指标
        """
        return {
            "uptime": time.time() - self._start_time,
            "adapter_version": self.adapter_version
        }

    def _record_request(self, response_time: float = 0.0, success: bool = True):
        """记录请求统计"""
        self._request_count += 1
        self._total_response_time += response_time

        if not success:
            self._error_count += 1

    async def _execute_with_metrics(self, func, *args, **kwargs):
        """
        执行函数并记录指标

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果
        """
        start_time = time.time()
        success = True

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        except Exception as e:
            success = False
            self.log_error(e, {"function": func.__name__})
            raise

        finally:
            response_time = time.time() - start_time
            self._record_request(response_time, success)

    # === 抽象方法（可选实现） ===

    async def _initialize_impl(self) -> bool:
        """
        子类可以实现此方法来添加特定的初始化逻辑

        Returns:
            bool: 初始化是否成功
        """
        return True

    async def _start_impl(self) -> bool:
        """
        子类可以实现此方法来添加特定的启动逻辑

        Returns:
            bool: 启动是否成功
        """
        return True

    async def _stop_impl(self) -> bool:
        """
        子类可以实现此方法来添加特定的停止逻辑

        Returns:
            bool: 停止是否成功
        """
        return True

    async def _health_check_impl(self) -> Optional[HealthCheckResult]:
        """
        子类可以实现此方法来添加特定的健康检查逻辑

        Returns:
            Optional[HealthCheckResult]: 健康检查结果，返回None使用默认检查
        """
        return None


class AsyncContextAdapter(BaseAdapter):
    """
    支持异步上下文管理的适配器基类
    """

    async def __aenter__(self):
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        self.cleanup()