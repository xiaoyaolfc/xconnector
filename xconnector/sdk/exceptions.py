# xconnector/sdk/exceptions.py
"""
XConnector SDK 异常定义

定义SDK专用的异常类型，提供清晰的错误处理
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorCode(Enum):
    """错误代码枚举"""

    # SDK通用错误 (1000-1099)
    SDK_INITIALIZATION_FAILED = 1000
    SDK_NOT_INITIALIZED = 1001
    SDK_ALREADY_STARTED = 1002
    SDK_NOT_STARTED = 1003
    SDK_CONFIGURATION_ERROR = 1004
    SDK_MODE_NOT_SUPPORTED = 1005

    # 适配器错误 (1100-1199)
    ADAPTER_NOT_FOUND = 1100
    ADAPTER_LOAD_FAILED = 1101
    ADAPTER_START_FAILED = 1102
    ADAPTER_STOP_FAILED = 1103
    ADAPTER_CONFIG_INVALID = 1104
    ADAPTER_DEPENDENCY_MISSING = 1105
    ADAPTER_ALREADY_EXISTS = 1106

    # 路由错误 (1200-1299)
    ROUTE_NOT_FOUND = 1200
    ROUTE_EXECUTION_FAILED = 1201
    ROUTE_TIMEOUT = 1202
    ROUTE_CIRCUIT_BREAKER_OPEN = 1203
    ROUTE_LOAD_BALANCER_ERROR = 1204

    # KV缓存错误 (1300-1399)
    CACHE_HANDLER_NOT_AVAILABLE = 1300
    CACHE_RETRIEVE_FAILED = 1301
    CACHE_STORE_FAILED = 1302
    CACHE_CLEANUP_FAILED = 1303
    CACHE_BACKEND_ERROR = 1304

    # 集成错误 (1400-1499)
    INTEGRATION_FAILED = 1400
    VLLM_INTEGRATION_ERROR = 1401
    DYNAMO_INTEGRATION_ERROR = 1402
    LMCACHE_INTEGRATION_ERROR = 1403
    CONFIG_CONVERSION_ERROR = 1404

    # 健康检查错误 (1500-1599)
    HEALTH_CHECK_FAILED = 1500
    ADAPTER_UNHEALTHY = 1501
    SERVICE_DEGRADED = 1502


class XConnectorSDKException(Exception):
    """XConnector SDK基础异常类"""

    def __init__(
            self,
            message: str,
            error_code: Optional[ErrorCode] = None,
            details: Optional[Dict[str, Any]] = None,
            cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.error_code:
            return f"[{self.error_code.name}] {base_msg}"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code.name if self.error_code else None,
            "error_code_value": self.error_code.value if self.error_code else None,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


# === SDK初始化和生命周期异常 ===

class SDKInitializationError(XConnectorSDKException):
    """SDK初始化失败异常"""

    def __init__(self, message: str = "SDK initialization failed", **kwargs):
        super().__init__(message, ErrorCode.SDK_INITIALIZATION_FAILED, **kwargs)


class SDKNotInitializedError(XConnectorSDKException):
    """SDK未初始化异常"""

    def __init__(self, message: str = "SDK not initialized", **kwargs):
        super().__init__(message, ErrorCode.SDK_NOT_INITIALIZED, **kwargs)


class SDKAlreadyStartedError(XConnectorSDKException):
    """SDK已启动异常"""

    def __init__(self, message: str = "SDK already started", **kwargs):
        super().__init__(message, ErrorCode.SDK_ALREADY_STARTED, **kwargs)


class SDKNotStartedError(XConnectorSDKException):
    """SDK未启动异常"""

    def __init__(self, message: str = "SDK not started", **kwargs):
        super().__init__(message, ErrorCode.SDK_NOT_STARTED, **kwargs)


class SDKConfigurationError(XConnectorSDKException):
    """SDK配置错误异常"""

    def __init__(self, message: str = "SDK configuration error", **kwargs):
        super().__init__(message, ErrorCode.SDK_CONFIGURATION_ERROR, **kwargs)


# === 适配器相关异常 ===

class AdapterError(XConnectorSDKException):
    """适配器基础异常"""
    pass


class AdapterNotFoundError(AdapterError):
    """适配器未找到异常"""

    def __init__(self, adapter_name: str, adapter_type: str = "", **kwargs):
        message = f"Adapter '{adapter_name}' not found"
        if adapter_type:
            message += f" (type: {adapter_type})"
        super().__init__(message, ErrorCode.ADAPTER_NOT_FOUND,
                         details={"adapter_name": adapter_name, "adapter_type": adapter_type}, **kwargs)


class AdapterLoadError(AdapterError):
    """适配器加载失败异常"""

    def __init__(self, adapter_name: str, reason: str = "", **kwargs):
        message = f"Failed to load adapter '{adapter_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, ErrorCode.ADAPTER_LOAD_FAILED,
                         details={"adapter_name": adapter_name, "reason": reason}, **kwargs)


class AdapterConfigError(AdapterError):
    """适配器配置错误异常"""

    def __init__(self, adapter_name: str, config_errors: List[str] = None, **kwargs):
        message = f"Invalid configuration for adapter '{adapter_name}'"
        if config_errors:
            message += f": {', '.join(config_errors)}"
        super().__init__(message, ErrorCode.ADAPTER_CONFIG_INVALID,
                         details={"adapter_name": adapter_name, "config_errors": config_errors or []}, **kwargs)


class AdapterDependencyError(AdapterError):
    """适配器依赖缺失异常"""

    def __init__(self, adapter_name: str, missing_dependencies: List[str] = None, **kwargs):
        message = f"Missing dependencies for adapter '{adapter_name}'"
        if missing_dependencies:
            message += f": {', '.join(missing_dependencies)}"
        super().__init__(message, ErrorCode.ADAPTER_DEPENDENCY_MISSING,
                         details={"adapter_name": adapter_name, "missing_dependencies": missing_dependencies or []},
                         **kwargs)


# === 路由相关异常 ===

class RouteError(XConnectorSDKException):
    """路由基础异常"""
    pass


class RouteNotFoundError(RouteError):
    """路由未找到异常"""

    def __init__(self, source: str, target: str, method: str = "", **kwargs):
        message = f"Route not found: {source} -> {target}"
        if method:
            message += f"::{method}"
        super().__init__(message, ErrorCode.ROUTE_NOT_FOUND,
                         details={"source": source, "target": target, "method": method}, **kwargs)


class RouteExecutionError(RouteError):
    """路由执行失败异常"""

    def __init__(self, source: str, target: str, method: str = "", reason: str = "", **kwargs):
        message = f"Route execution failed: {source} -> {target}"
        if method:
            message += f"::{method}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, ErrorCode.ROUTE_EXECUTION_FAILED,
                         details={"source": source, "target": target, "method": method, "reason": reason}, **kwargs)


class RouteTimeoutError(RouteError):
    """路由超时异常"""

    def __init__(self, source: str, target: str, timeout: float, **kwargs):
        message = f"Route timeout: {source} -> {target} (timeout: {timeout}s)"
        super().__init__(message, ErrorCode.ROUTE_TIMEOUT,
                         details={"source": source, "target": target, "timeout": timeout}, **kwargs)


# === KV缓存相关异常 ===

class CacheError(XConnectorSDKException):
    """缓存基础异常"""
    pass


class CacheHandlerNotAvailableError(CacheError):
    """缓存处理器不可用异常"""

    def __init__(self, message: str = "KV cache handler not available", **kwargs):
        super().__init__(message, ErrorCode.CACHE_HANDLER_NOT_AVAILABLE, **kwargs)


class CacheRetrieveError(CacheError):
    """缓存检索失败异常"""

    def __init__(self, cache_key: str = "", reason: str = "", **kwargs):
        message = "Cache retrieve failed"
        if cache_key:
            message += f" (key: {cache_key})"
        if reason:
            message += f": {reason}"
        super().__init__(message, ErrorCode.CACHE_RETRIEVE_FAILED,
                         details={"cache_key": cache_key, "reason": reason}, **kwargs)


class CacheStoreError(CacheError):
    """缓存存储失败异常"""

    def __init__(self, cache_key: str = "", reason: str = "", **kwargs):
        message = "Cache store failed"
        if cache_key:
            message += f" (key: {cache_key})"
        if reason:
            message += f": {reason}"
        super().__init__(message, ErrorCode.CACHE_STORE_FAILED,
                         details={"cache_key": cache_key, "reason": reason}, **kwargs)


# === 集成相关异常 ===

class IntegrationError(XConnectorSDKException):
    """集成基础异常"""
    pass


class VLLMIntegrationError(IntegrationError):
    """vLLM集成异常"""

    def __init__(self, message: str = "vLLM integration failed", **kwargs):
        super().__init__(message, ErrorCode.VLLM_INTEGRATION_ERROR, **kwargs)


class DynamoIntegrationError(IntegrationError):
    """Dynamo集成异常"""

    def __init__(self, message: str = "Dynamo integration failed", **kwargs):
        super().__init__(message, ErrorCode.DYNAMO_INTEGRATION_ERROR, **kwargs)


class LMCacheIntegrationError(IntegrationError):
    """LMCache集成异常"""

    def __init__(self, message: str = "LMCache integration failed", **kwargs):
        super().__init__(message, ErrorCode.LMCACHE_INTEGRATION_ERROR, **kwargs)


class ConfigConversionError(IntegrationError):
    """配置转换异常"""

    def __init__(self, source_format: str, target_format: str, reason: str = "", **kwargs):
        message = f"Config conversion failed: {source_format} -> {target_format}"
        if reason:
            message += f": {reason}"
        super().__init__(message, ErrorCode.CONFIG_CONVERSION_ERROR,
                         details={"source_format": source_format, "target_format": target_format, "reason": reason},
                         **kwargs)


# === 健康检查相关异常 ===

class HealthCheckError(XConnectorSDKException):
    """健康检查异常"""

    def __init__(self, component: str = "", reason: str = "", **kwargs):
        message = "Health check failed"
        if component:
            message += f" for {component}"
        if reason:
            message += f": {reason}"
        super().__init__(message, ErrorCode.HEALTH_CHECK_FAILED,
                         details={"component": component, "reason": reason}, **kwargs)


# === 异常处理工具 ===

class ExceptionHandler:
    """异常处理工具类"""

    @staticmethod
    def handle_gracefully(func, fallback_result=None, log_error=True):
        """优雅处理异常的装饰器"""

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except XConnectorSDKException as e:
                if log_error:
                    from xconnector.utils.xconnector_logging import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"SDK exception in {func.__name__}: {e}")
                return fallback_result
            except Exception as e:
                if log_error:
                    from xconnector.utils.xconnector_logging import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Unexpected exception in {func.__name__}: {e}")
                return fallback_result

        return wrapper

    @staticmethod
    def convert_to_sdk_exception(e: Exception, default_error_code: ErrorCode = None) -> XConnectorSDKException:
        """将普通异常转换为SDK异常"""
        if isinstance(e, XConnectorSDKException):
            return e

        # 根据异常类型进行转换
        if isinstance(e, ImportError):
            return AdapterDependencyError("unknown", missing_dependencies=[str(e)], cause=e)
        elif isinstance(e, FileNotFoundError):
            return SDKConfigurationError(f"Configuration file not found: {e}", cause=e)
        elif isinstance(e, TimeoutError):
            return RouteTimeoutError("unknown", "unknown", 0, cause=e)
        else:
            # 默认转换
            return XConnectorSDKException(
                f"Unexpected error: {str(e)}",
                error_code=default_error_code,
                cause=e
            )


# === 异常上下文管理器 ===

class SDKExceptionContext:
    """SDK异常上下文管理器"""

    def __init__(self, operation: str, component: str = "", graceful: bool = True):
        self.operation = operation
        self.component = component
        self.graceful = graceful

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        # 如果已经是SDK异常，直接传递
        if isinstance(exc_val, XConnectorSDKException):
            return False

        # 转换为SDK异常
        sdk_exception = ExceptionHandler.convert_to_sdk_exception(exc_val)

        # 添加操作上下文
        if self.component:
            sdk_exception.details.update({
                "operation": self.operation,
                "component": self.component
            })

        if self.graceful:
            # 优雅处理：记录日志但不抛出异常
            from xconnector.utils.xconnector_logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"Operation '{self.operation}' failed: {sdk_exception}")
            return True  # 抑制异常
        else:
            # 抛出转换后的SDK异常
            raise sdk_exception from exc_val


# === 批量异常处理 ===

class BatchExceptionCollector:
    """批量异常收集器"""

    def __init__(self):
        self.exceptions: List[XConnectorSDKException] = []

    def add_exception(self, exception: Exception, context: str = ""):
        """添加异常"""
        sdk_exception = ExceptionHandler.convert_to_sdk_exception(exception)
        if context:
            sdk_exception.details["context"] = context
        self.exceptions.append(sdk_exception)

    def has_exceptions(self) -> bool:
        """是否有异常"""
        return len(self.exceptions) > 0

    def get_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        return {
            "total_count": len(self.exceptions),
            "error_codes": [e.error_code.name for e in self.exceptions if e.error_code],
            "messages": [str(e) for e in self.exceptions]
        }

    def raise_if_any(self, summary_message: str = "Multiple errors occurred"):
        """如果有异常则抛出"""
        if self.has_exceptions():
            raise XConnectorSDKException(
                summary_message,
                details={
                    "exception_count": len(self.exceptions),
                    "exceptions": [e.to_dict() for e in self.exceptions]
                }
            )


# === 特定场景的异常处理器 ===

def handle_adapter_operation(operation_name: str):
    """适配器操作异常处理装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                adapter_name = getattr(args[0], 'adapter_name', 'unknown') if args else 'unknown'

                if isinstance(e, XConnectorSDKException):
                    raise e

                # 根据操作类型转换异常
                if 'load' in operation_name.lower():
                    raise AdapterLoadError(adapter_name, str(e), cause=e)
                elif 'start' in operation_name.lower():
                    raise AdapterError(f"Failed to start adapter '{adapter_name}': {e}",
                                       ErrorCode.ADAPTER_START_FAILED, cause=e)
                elif 'stop' in operation_name.lower():
                    raise AdapterError(f"Failed to stop adapter '{adapter_name}': {e}",
                                       ErrorCode.ADAPTER_STOP_FAILED, cause=e)
                else:
                    raise AdapterError(f"Adapter operation '{operation_name}' failed for '{adapter_name}': {e}",
                                       cause=e)

        return wrapper

    return decorator


def handle_cache_operation(operation_name: str):
    """缓存操作异常处理装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, XConnectorSDKException):
                    raise e

                # 根据操作类型转换异常
                if 'retrieve' in operation_name.lower():
                    raise CacheRetrieveError(reason=str(e), cause=e)
                elif 'store' in operation_name.lower():
                    raise CacheStoreError(reason=str(e), cause=e)
                else:
                    raise CacheError(f"Cache operation '{operation_name}' failed: {e}", cause=e)

        return wrapper

    return decorator


# === 快速异常创建函数 ===

def sdk_not_initialized(operation: str = "") -> SDKNotInitializedError:
    """创建SDK未初始化异常"""
    message = "SDK not initialized"
    if operation:
        message += f" for operation: {operation}"
    return SDKNotInitializedError(message)


def adapter_not_found(adapter_name: str, adapter_type: str = "") -> AdapterNotFoundError:
    """创建适配器未找到异常"""
    return AdapterNotFoundError(adapter_name, adapter_type)


def route_not_found(source: str, target: str, method: str = "") -> RouteNotFoundError:
    """创建路由未找到异常"""
    return RouteNotFoundError(source, target, method)


def cache_handler_unavailable(reason: str = "") -> CacheHandlerNotAvailableError:
    """创建缓存处理器不可用异常"""
    message = "KV cache handler not available"
    if reason:
        message += f": {reason}"
    return CacheHandlerNotAvailableError(message)


# === 异常统计 ===

class ExceptionStats:
    """异常统计器"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.total_exceptions = 0

    def record_exception(self, exception: XConnectorSDKException):
        """记录异常"""
        self.total_exceptions += 1
        error_name = exception.__class__.__name__
        self.error_counts[error_name] = self.error_counts.get(error_name, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_exceptions": self.total_exceptions,
            "error_counts": self.error_counts.copy(),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }

    def reset(self):
        """重置统计"""
        self.error_counts.clear()
        self.total_exceptions = 0


# 全局异常统计实例
_global_exception_stats = ExceptionStats()


def get_exception_stats() -> ExceptionStats:
    """获取全局异常统计实例"""
    return _global_exception_stats


# === 使用示例 ===

if __name__ == "__main__":
    # 示例1: 基本异常使用
    try:
        raise AdapterNotFoundError("test_adapter", "cache")
    except XConnectorSDKException as e:
        print(f"Caught SDK exception: {e}")
        print(f"Exception dict: {e.to_dict()}")

    # 示例2: 异常上下文管理器
    with SDKExceptionContext("test_operation", "test_component", graceful=True):
        raise ValueError("Test error")  # 会被自动转换和处理

    # 示例3: 批量异常收集
    collector = BatchExceptionCollector()
    collector.add_exception(ValueError("Error 1"), "context 1")
    collector.add_exception(RuntimeError("Error 2"), "context 2")

    if collector.has_exceptions():
        print(f"Collected exceptions: {collector.get_summary()}")

    # 示例4: 异常统计
    stats = get_exception_stats()
    stats.record_exception(AdapterNotFoundError("test"))
    print(f"Exception stats: {stats.get_stats()}")