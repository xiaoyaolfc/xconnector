# integrations/dynamo/sdk/worker_wrapper.py
"""
Dynamo Worker包装器

提供对Dynamo Worker的方法包装，实现XConnector功能的透明集成。
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Tuple, List, Dict
import torch

from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


def wrap_kv_cache_methods(worker_instance: Any, sdk: Any, config: Any) -> None:
    """
    包装Worker的KV缓存相关方法

    Args:
        worker_instance: Dynamo Worker实例
        sdk: XConnectorSDK实例
        config: 集成配置
    """
    # 检查worker是否有KV缓存方法
    has_recv_kv = hasattr(worker_instance, 'recv_kv_caches')
    has_send_kv = hasattr(worker_instance, 'send_kv_caches')

    if not (has_recv_kv or has_send_kv):
        logger.warning("Worker doesn't have KV cache methods to wrap")
        return

    logger.info("Wrapping KV cache methods...")

    # 包装recv_kv_caches方法
    if has_recv_kv:
        original_recv = worker_instance.recv_kv_caches
        worker_instance._original_recv_kv_caches = original_recv
        worker_instance.recv_kv_caches = create_recv_kv_wrapper(
            original_recv, sdk, config
        )
        logger.debug("Wrapped recv_kv_caches method")

    # 包装send_kv_caches方法
    if has_send_kv:
        original_send = worker_instance.send_kv_caches
        worker_instance._original_send_kv_caches = original_send
        worker_instance.send_kv_caches = create_send_kv_wrapper(
            original_send, sdk, config
        )
        logger.debug("Wrapped send_kv_caches method")


def create_recv_kv_wrapper(
        original_method: Callable,
        sdk: Any,
        config: Any
) -> Callable:
    """
    创建recv_kv_caches方法的包装器

    Args:
        original_method: 原始方法
        sdk: XConnectorSDK实例
        config: 集成配置

    Returns:
        Callable: 包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_recv_kv_caches(*args, **kwargs):
        """
        包装的KV接收方法

        尝试从XConnector缓存中检索KV，如果未命中则回退到原始方法
        """
        start_time = time.time()

        try:
            # 提取参数
            model_executable = args[0] if len(args) > 0 else kwargs.get('model_executable')
            model_input = args[1] if len(args) > 1 else kwargs.get('model_input')
            kv_caches = args[2] if len(args) > 2 else kwargs.get('kv_caches')

            if model_input is None or kv_caches is None:
                logger.debug("Missing model_input or kv_caches, falling back to original method")
                return await original_method(*args, **kwargs)

            # 尝试从XConnector缓存检索
            kv_handler = sdk.get_kv_handler()
            if kv_handler:
                cache_result = await kv_handler.retrieve_kv(model_input, kv_caches)

                if cache_result.get("found", False):
                    # 缓存命中
                    logger.debug("XConnector cache hit")

                    # 更新监控信息
                    _update_monitoring(kwargs.get('worker_instance'), 'cache_hit')

                    # 返回缓存结果
                    return (
                        cache_result.get("hidden_states"),
                        cache_result.get("skip_forward", False),
                        cache_result.get("updated_input", model_input)
                    )
                else:
                    # 缓存未命中，记录原因
                    logger.debug(f"XConnector cache miss: {cache_result.get('reason', 'unknown')}")
                    _update_monitoring(kwargs.get('worker_instance'), 'cache_miss')

            # 回退到原始方法
            logger.debug("Falling back to original recv_kv_caches method")
            result = await original_method(*args, **kwargs)

            # 记录性能信息
            elapsed_time = time.time() - start_time
            logger.debug(f"recv_kv_caches completed in {elapsed_time:.3f}s")

            return result

        except Exception as e:
            # 错误处理
            _update_monitoring(kwargs.get('worker_instance'), 'error')

            if config.graceful_degradation:
                logger.warning(f"XConnector recv_kv failed, falling back: {e}")
                return await original_method(*args, **kwargs)
            else:
                logger.error(f"XConnector recv_kv failed: {e}")
                raise

    return wrapped_recv_kv_caches


def create_send_kv_wrapper(
        original_method: Callable,
        sdk: Any,
        config: Any
) -> Callable:
    """
    创建send_kv_caches方法的包装器

    Args:
        original_method: 原始方法
        sdk: XConnectorSDK实例
        config: 集成配置

    Returns:
        Callable: 包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_send_kv_caches(*args, **kwargs):
        """
        包装的KV发送方法

        先调用原始方法，然后将结果存储到XConnector缓存
        """
        start_time = time.time()

        try:
            # 首先调用原始方法
            result = await original_method(*args, **kwargs)

            # 提取参数用于缓存存储
            model_executable = args[0] if len(args) > 0 else kwargs.get('model_executable')
            model_input = args[1] if len(args) > 1 else kwargs.get('model_input')
            kv_caches = args[2] if len(args) > 2 else kwargs.get('kv_caches')
            hidden_states = args[3] if len(args) > 3 else kwargs.get('hidden_or_intermediate_states')

            if model_input is not None and kv_caches is not None:
                # 尝试存储到XConnector缓存
                kv_handler = sdk.get_kv_handler()
                if kv_handler:
                    try:
                        success = await kv_handler.store_kv(
                            model_input,
                            kv_caches,
                            hidden_states,
                            metadata={
                                'timestamp': time.time(),
                                'worker_type': type(kwargs.get('worker_instance', {})).__name__
                            }
                        )

                        if success:
                            logger.debug("Successfully stored KV to XConnector cache")
                        else:
                            logger.debug("Failed to store KV to XConnector cache")

                    except Exception as e:
                        logger.warning(f"XConnector KV storage failed: {e}")
                        _update_monitoring(kwargs.get('worker_instance'), 'storage_error')

            # 记录性能信息
            elapsed_time = time.time() - start_time
            logger.debug(f"send_kv_caches completed in {elapsed_time:.3f}s")

            return result

        except Exception as e:
            _update_monitoring(kwargs.get('worker_instance'), 'error')

            if config.graceful_degradation:
                logger.warning(f"XConnector send_kv failed, continuing: {e}")
                # 即使XConnector存储失败，也返回原始方法的结果
                try:
                    return await original_method(*args, **kwargs)
                except Exception as original_error:
                    logger.error(f"Original send_kv also failed: {original_error}")
                    raise original_error
            else:
                logger.error(f"XConnector send_kv failed: {e}")
                raise

    return wrapped_send_kv_caches


def wrap_distributed_methods(worker_instance: Any, sdk: Any, config: Any) -> None:
    """
    包装分布式相关方法

    Args:
        worker_instance: Dynamo Worker实例
        sdk: XConnectorSDK实例
        config: 集成配置
    """
    # 检查是否有分布式相关方法需要包装
    distributed_methods = [
        'get_finished',
        'handle_finished_requests',
        'coordinate_workers'
    ]

    methods_to_wrap = []
    for method_name in distributed_methods:
        if hasattr(worker_instance, method_name):
            methods_to_wrap.append(method_name)

    if not methods_to_wrap:
        logger.debug("No distributed methods found to wrap")
        return

    logger.info(f"Wrapping distributed methods: {methods_to_wrap}")

    # 包装get_finished方法
    if hasattr(worker_instance, 'get_finished'):
        original_get_finished = worker_instance.get_finished
        worker_instance._original_get_finished = original_get_finished
        worker_instance.get_finished = create_get_finished_wrapper(
            original_get_finished, sdk, config
        )
        logger.debug("Wrapped get_finished method")


def create_get_finished_wrapper(
        original_method: Callable,
        sdk: Any,
        config: Any
) -> Callable:
    """
    创建get_finished方法的包装器

    Args:
        original_method: 原始方法
        sdk: XConnectorSDK实例
        config: 集成配置

    Returns:
        Callable: 包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_get_finished(*args, **kwargs):
        """
        包装的完成请求处理方法

        处理完成的请求并清理相关缓存
        """
        try:
            # 调用原始方法
            result = await original_method(*args, **kwargs)

            # 提取完成的请求ID
            finished_req_ids = args[0] if len(args) > 0 else kwargs.get('finished_req_ids', set())

            if finished_req_ids:
                # 清理XConnector缓存
                kv_handler = sdk.get_kv_handler()
                if kv_handler:
                    try:
                        cleaned_count = await kv_handler.cleanup_finished(list(finished_req_ids))
                        logger.debug(
                            f"Cleaned up {cleaned_count} cache entries for {len(finished_req_ids)} finished requests")
                    except Exception as e:
                        logger.warning(f"Cache cleanup failed: {e}")

            return result

        except Exception as e:
            if config.graceful_degradation:
                logger.warning(f"XConnector get_finished wrapper failed, falling back: {e}")
                return await original_method(*args, **kwargs)
            else:
                logger.error(f"XConnector get_finished wrapper failed: {e}")
                raise

    return wrapped_get_finished


def _update_monitoring(worker_instance: Any, event_type: str) -> None:
    """
    更新监控信息

    Args:
        worker_instance: Worker实例
        event_type: 事件类型
    """
    try:
        if worker_instance and hasattr(worker_instance, '_xconnector_monitoring'):
            monitoring = worker_instance._xconnector_monitoring

            if event_type == 'cache_hit':
                monitoring['cache_hits'] += 1
            elif event_type == 'cache_miss':
                monitoring['cache_misses'] += 1
            elif event_type in ['error', 'storage_error']:
                monitoring['errors'] += 1

            monitoring['request_count'] += 1

    except Exception as e:
        # 监控更新失败不应该影响主流程
        logger.debug(f"Monitoring update failed: {e}")


def unwrap_worker_methods(worker_instance: Any) -> None:
    """
    恢复Worker的原始方法

    Args:
        worker_instance: Dynamo Worker实例
    """
    try:
        # 恢复KV缓存方法
        if hasattr(worker_instance, '_original_recv_kv_caches'):
            worker_instance.recv_kv_caches = worker_instance._original_recv_kv_caches
            delattr(worker_instance, '_original_recv_kv_caches')
            logger.debug("Restored original recv_kv_caches method")

        if hasattr(worker_instance, '_original_send_kv_caches'):
            worker_instance.send_kv_caches = worker_instance._original_send_kv_caches
            delattr(worker_instance, '_original_send_kv_caches')
            logger.debug("Restored original send_kv_caches method")

        # 恢复分布式方法
        if hasattr(worker_instance, '_original_get_finished'):
            worker_instance.get_finished = worker_instance._original_get_finished
            delattr(worker_instance, '_original_get_finished')
            logger.debug("Restored original get_finished method")

        logger.info("Successfully unwrapped worker methods")

    except Exception as e:
        logger.error(f"Failed to unwrap worker methods: {e}")


def get_worker_monitoring_stats(worker_instance: Any) -> Dict[str, Any]:
    """
    获取Worker的监控统计信息

    Args:
        worker_instance: Dynamo Worker实例

    Returns:
        Dict: 监控统计信息
    """
    if not hasattr(worker_instance, '_xconnector_monitoring'):
        return {}

    monitoring = worker_instance._xconnector_monitoring
    total_requests = monitoring.get('request_count', 0)
    cache_hits = monitoring.get('cache_hits', 0)

    hit_rate = (cache_hits / max(total_requests, 1)) * 100

    return {
        'total_requests': total_requests,
        'cache_hits': cache_hits,
        'cache_misses': monitoring.get('cache_misses', 0),
        'errors': monitoring.get('errors', 0),
        'hit_rate': f"{hit_rate:.2f}%",
        'worker_type': type(worker_instance).__name__
    }


# === 装饰器支持 ===

def xconnector_integration(
        enable_kv_cache: bool = True,
        enable_distributed: bool = True,
        graceful_degradation: bool = True
):
    """
    用于Worker类的XConnector集成装饰器

    Args:
        enable_kv_cache: 是否启用KV缓存集成
        enable_distributed: 是否启用分布式集成
        graceful_degradation: 是否启用优雅降级

    Returns:
        装饰器函数

    Example:
        >>> @xconnector_integration(enable_kv_cache=True)
        ... class MyVLLMWorker(VLLMWorker):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         # Worker将自动集成XConnector功能
    """

    def decorator(worker_class):
        original_init = worker_class.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # 调用原始初始化
            original_init(self, *args, **kwargs)

            # 尝试自动集成XConnector
            try:
                from integrations.dynamo.sdk.integration import get_integration_instance

                integration = get_integration_instance()
                if integration and integration.initialized:
                    # 创建集成配置
                    config = type('Config', (), {
                        'wrap_kv_methods': enable_kv_cache,
                        'wrap_distributed_methods': enable_distributed,
                        'graceful_degradation': graceful_degradation,
                        'fail_on_error': False
                    })()

                    # 包装方法
                    if enable_kv_cache:
                        wrap_kv_cache_methods(self, integration.sdk, config)

                    if enable_distributed:
                        wrap_distributed_methods(self, integration.sdk, config)

                    logger.info(f"XConnector integration applied to {worker_class.__name__}")
                else:
                    logger.warning("XConnector integration not available, worker will function normally")

            except Exception as e:
                logger.warning(f"XConnector integration failed for {worker_class.__name__}: {e}")
                # 继续正常工作，不抛出异常

        worker_class.__init__ = wrapped_init
        return worker_class

    return decorator


# === 辅助函数 ===

def check_worker_compatibility(worker_instance: Any) -> Dict[str, bool]:
    """
    检查Worker与XConnector的兼容性

    Args:
        worker_instance: Dynamo Worker实例

    Returns:
        Dict: 兼容性检查结果
    """
    compatibility = {
        'has_recv_kv_caches': hasattr(worker_instance, 'recv_kv_caches'),
        'has_send_kv_caches': hasattr(worker_instance, 'send_kv_caches'),
        'has_get_finished': hasattr(worker_instance, 'get_finished'),
        'is_vllm_worker': 'VLLM' in type(worker_instance).__name__.upper(),
        'is_async_compatible': asyncio.iscoroutinefunction(
            getattr(worker_instance, 'recv_kv_caches', None)
        ),
    }

    # 计算总体兼容性分数
    compatible_features = sum(compatibility.values())
    total_features = len(compatibility)
    compatibility['compatibility_score'] = f"{compatible_features}/{total_features}"
    compatibility['is_compatible'] = compatible_features >= 2  # 至少需要2个特性

    return compatibility


def create_compatibility_report(worker_instance: Any) -> str:
    """
    创建兼容性报告

    Args:
        worker_instance: Dynamo Worker实例

    Returns:
        str: 兼容性报告
    """
    compatibility = check_worker_compatibility(worker_instance)
    worker_type = type(worker_instance).__name__

    report = f"XConnector Compatibility Report for {worker_type}:\n"
    report += f"  Overall Score: {compatibility['compatibility_score']}\n"
    report += f"  Compatible: {'✓' if compatibility['is_compatible'] else '✗'}\n"
    report += "\n  Feature Support:\n"

    feature_names = {
        'has_recv_kv_caches': 'KV Cache Receive',
        'has_send_kv_caches': 'KV Cache Send',
        'has_get_finished': 'Finished Request Handling',
        'is_vllm_worker': 'vLLM Worker Type',
        'is_async_compatible': 'Async Method Support'
    }

    for feature, supported in compatibility.items():
        if feature in feature_names:
            status = '✓' if supported else '✗'
            report += f"    {feature_names[feature]}: {status}\n"

    # 添加建议
    if not compatibility['is_compatible']:
        report += "\n  Recommendations:\n"
        if not compatibility['has_recv_kv_caches']:
            report += "    - Implement recv_kv_caches method for cache retrieval\n"
        if not compatibility['has_send_kv_caches']:
            report += "    - Implement send_kv_caches method for cache storage\n"
        if not compatibility['is_async_compatible']:
            report += "    - Ensure KV cache methods are async for better integration\n"

    return report


def validate_method_signature(method: Callable, expected_params: List[str]) -> bool:
    """
    验证方法签名是否符合预期

    Args:
        method: 要验证的方法
        expected_params: 期望的参数列表

    Returns:
        bool: 签名是否匹配
    """
    try:
        import inspect

        signature = inspect.signature(method)
        param_names = list(signature.parameters.keys())

        # 检查是否包含所有期望的参数
        for param in expected_params:
            if param not in param_names:
                return False

        return True

    except Exception as e:
        logger.debug(f"Method signature validation failed: {e}")
        return False


# === 性能监控 ===

class PerformanceMonitor:
    """性能监控类"""

    def __init__(self):
        self.stats = {
            'method_calls': {},
            'execution_times': {},
            'cache_performance': {
                'hits': 0,
                'misses': 0,
                'errors': 0
            }
        }

    def record_method_call(self, method_name: str, execution_time: float):
        """记录方法调用"""
        if method_name not in self.stats['method_calls']:
            self.stats['method_calls'][method_name] = 0
            self.stats['execution_times'][method_name] = []

        self.stats['method_calls'][method_name] += 1
        self.stats['execution_times'][method_name].append(execution_time)

        # 保持最近100次记录
        if len(self.stats['execution_times'][method_name]) > 100:
            self.stats['execution_times'][method_name].pop(0)

    def record_cache_event(self, event_type: str):
        """记录缓存事件"""
        if event_type in self.stats['cache_performance']:
            self.stats['cache_performance'][event_type] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'method_calls': self.stats['method_calls'].copy(),
            'cache_performance': self.stats['cache_performance'].copy(),
            'avg_execution_times': {}
        }

        # 计算平均执行时间
        for method, times in self.stats['execution_times'].items():
            if times:
                summary['avg_execution_times'][method] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }

        # 计算缓存命中率
        cache_stats = self.stats['cache_performance']
        total_cache_ops = cache_stats['hits'] + cache_stats['misses']
        if total_cache_ops > 0:
            hit_rate = (cache_stats['hits'] / total_cache_ops) * 100
            summary['cache_performance']['hit_rate'] = f"{hit_rate:.2f}%"

        return summary


# 全局性能监控实例
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控实例"""
    return _performance_monitor


# === 导出 ===

__all__ = [
    'wrap_kv_cache_methods',
    'wrap_distributed_methods',
    'create_recv_kv_wrapper',
    'create_send_kv_wrapper',
    'create_get_finished_wrapper',
    'unwrap_worker_methods',
    'get_worker_monitoring_stats',
    'xconnector_integration',
    'check_worker_compatibility',
    'create_compatibility_report',
    'validate_method_signature',
    'PerformanceMonitor',
    'get_performance_monitor'
]