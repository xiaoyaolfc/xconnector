# xconnector/integrations/dynamo/worker_injector.py
"""
Worker类Monkey Patch注入器

负责：
1. 检测和识别Dynamo Worker类
2. Monkey patch Worker的关键方法
3. 注入XConnector功能
4. 设置import钩子处理未来导入的类

设计原则：
- 最小侵入：只patch必要的方法
- 高效执行：最小性能开销
- 容错优先：出错时回退到原方法
- 简单直接：避免复杂的hook机制
"""

import sys
import logging
import inspect
import functools
from typing import Any, Dict, List, Optional, Callable, Set
import asyncio

# 获取logger
logger = logging.getLogger('xconnector.dynamo.worker_injector')

# 已经patch过的类集合，避免重复patch
_patched_classes: Set[type] = set()
_original_methods: Dict[str, Dict[str, Callable]] = {}


def _is_worker_class(cls: type) -> bool:
    """
    检测是否是Worker类

    Args:
        cls: 要检测的类

    Returns:
        bool: 是否是Worker类
    """
    try:
        class_name = cls.__name__.lower()
        module_name = getattr(cls.__module__, '__name__', '').lower() if cls.__module__ else ''

        # 检查类名模式
        worker_patterns = ['worker', 'vllm', 'prefill', 'decode']
        if any(pattern in class_name for pattern in worker_patterns):
            return True

        # 检查模块名模式
        if any(pattern in module_name for pattern in ['worker', 'vllm', 'dynamo']):
            return True

        # 检查是否有关键方法
        key_methods = ['recv_kv_caches', 'send_kv_caches', 'get_finished']
        if any(hasattr(cls, method) for method in key_methods):
            return True

        return False

    except Exception as e:
        logger.debug(f"Error detecting worker class {cls}: {e}")
        return False


def _has_kv_methods(cls: type) -> bool:
    """检查类是否有KV缓存相关方法"""
    kv_methods = ['recv_kv_caches', 'send_kv_caches']
    return any(hasattr(cls, method) for method in kv_methods)


def _create_recv_kv_wrapper(original_method: Callable, sdk) -> Callable:
    """
    创建recv_kv_caches方法的包装器

    Args:
        original_method: 原始方法
        sdk: MinimalXConnectorSDK实例

    Returns:
        包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_recv_kv_caches(self, *args, **kwargs):
        """
        包装的KV接收方法

        尝试从XConnector缓存检索，失败时回退到原方法
        """
        try:
            # 检查SDK和KV处理器是否可用
            if not sdk or not sdk.is_ready():
                logger.debug("SDK not ready, using original method")
                return await _call_original_method(original_method, self, *args, **kwargs)

            kv_handler = sdk.get_kv_handler()
            if not kv_handler:
                logger.debug("KV handler not available, using original method")
                return await _call_original_method(original_method, self, *args, **kwargs)

            # 提取参数
            model_input = _extract_model_input(args, kwargs)
            kv_caches = _extract_kv_caches(args, kwargs)

            if model_input is None or kv_caches is None:
                logger.debug("Cannot extract parameters, using original method")
                return await _call_original_method(original_method, self, *args, **kwargs)

            # 尝试从缓存检索
            cache_result = await kv_handler.retrieve_kv(model_input, kv_caches)

            if cache_result.get("found"):
                logger.debug("XConnector cache hit")

                # 返回缓存结果
                hidden_states = cache_result.get("hidden_states")
                skip_forward = cache_result.get("skip_forward", False)
                updated_input = cache_result.get("updated_input", model_input)

                return hidden_states, skip_forward, updated_input

            # 缓存未命中，使用原方法
            logger.debug("XConnector cache miss, using original method")
            return await _call_original_method(original_method, self, *args, **kwargs)

        except Exception as e:
            logger.debug(f"XConnector recv_kv failed: {e}, using original method")
            return await _call_original_method(original_method, self, *args, **kwargs)

    return wrapped_recv_kv_caches


def _create_send_kv_wrapper(original_method: Callable, sdk) -> Callable:
    """
    创建send_kv_caches方法的包装器

    Args:
        original_method: 原始方法
        sdk: MinimalXConnectorSDK实例

    Returns:
        包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_send_kv_caches(self, *args, **kwargs):
        """
        包装的KV发送方法

        先调用原方法，然后尝试存储到XConnector缓存
        """
        try:
            # 先调用原方法
            result = await _call_original_method(original_method, self, *args, **kwargs)

            # 检查SDK和KV处理器是否可用
            if not sdk or not sdk.is_ready():
                return result

            kv_handler = sdk.get_kv_handler()
            if not kv_handler:
                return result

            # 提取参数
            model_input = _extract_model_input(args, kwargs)
            kv_caches = _extract_kv_caches(args, kwargs)
            hidden_states = _extract_hidden_states(args, kwargs)

            if model_input is None or kv_caches is None:
                return result

            # 尝试存储到缓存
            await kv_handler.store_kv(model_input, kv_caches, hidden_states)
            logger.debug("Successfully stored KV to XConnector cache")

            return result

        except Exception as e:
            logger.debug(f"XConnector send_kv failed: {e}")
            # 即使存储失败，也返回原方法的结果
            return await _call_original_method(original_method, self, *args, **kwargs)

    return wrapped_send_kv_caches


def _create_get_finished_wrapper(original_method: Callable, sdk) -> Callable:
    """
    创建get_finished方法的包装器（可选）

    Args:
        original_method: 原始方法
        sdk: MinimalXConnectorSDK实例

    Returns:
        包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_get_finished(self, *args, **kwargs):
        """
        包装的完成请求处理方法

        处理完成的请求并清理缓存
        """
        try:
            # 调用原方法
            result = await _call_original_method(original_method, self, *args, **kwargs)

            # 提取完成的请求ID
            finished_req_ids = _extract_finished_request_ids(args, kwargs, result)

            if finished_req_ids and sdk and sdk.is_ready():
                kv_handler = sdk.get_kv_handler()
                if kv_handler:
                    try:
                        cleaned = await kv_handler.cleanup_finished(list(finished_req_ids))
                        logger.debug(f"Cleaned {cleaned} cache entries for finished requests")
                    except Exception as e:
                        logger.debug(f"Cache cleanup failed: {e}")

            return result

        except Exception as e:
            logger.debug(f"XConnector get_finished wrapper failed: {e}")
            return await _call_original_method(original_method, self, *args, **kwargs)

    return wrapped_get_finished


async def _call_original_method(method: Callable, instance, *args, **kwargs):
    """
    调用原始方法（处理异步/同步）

    Args:
        method: 要调用的方法
        instance: 实例
        *args, **kwargs: 参数

    Returns:
        方法执行结果
    """
    try:
        if asyncio.iscoroutinefunction(method):
            return await method(instance, *args, **kwargs)
        else:
            return method(instance, *args, **kwargs)
    except Exception as e:
        logger.error(f"Original method call failed: {e}")
        raise


def _extract_model_input(args: tuple, kwargs: dict) -> Any:
    """从参数中提取model_input"""
    # 通常是第二个参数（第一个是self）
    if len(args) > 1:
        return args[1]
    return kwargs.get('model_input')


def _extract_kv_caches(args: tuple, kwargs: dict) -> Any:
    """从参数中提取kv_caches"""
    # 通常是第三个参数
    if len(args) > 2:
        return args[2]
    return kwargs.get('kv_caches')


def _extract_hidden_states(args: tuple, kwargs: dict) -> Any:
    """从参数中提取hidden_states"""
    # 通常是第四个参数
    if len(args) > 3:
        return args[3]
    return kwargs.get('hidden_or_intermediate_states') or kwargs.get('hidden_states')


def _extract_finished_request_ids(args: tuple, kwargs: dict, result: Any) -> Optional[Set]:
    """从参数或结果中提取完成的请求ID"""
    # 尝试从参数中提取
    if args:
        if isinstance(args[0], (set, list)):
            return set(args[0])

    finished_ids = kwargs.get('finished_req_ids')
    if finished_ids:
        return set(finished_ids)

    # 尝试从结果中提取
    if isinstance(result, tuple) and len(result) > 0:
        if isinstance(result[0], (set, list)):
            return set(result[0])

    return None


def patch_worker_class(worker_class: type, sdk) -> bool:
    """
    对单个Worker类进行patch

    Args:
        worker_class: 要patch的Worker类
        sdk: MinimalXConnectorSDK实例

    Returns:
        bool: patch是否成功
    """
    if worker_class in _patched_classes:
        logger.debug(f"Class {worker_class.__name__} already patched")
        return True

    try:
        class_name = worker_class.__name__
        logger.debug(f"Patching worker class: {class_name}")

        # 保存原始方法
        _original_methods[class_name] = {}
        patched_methods = []

        # Patch recv_kv_caches方法
        if hasattr(worker_class, 'recv_kv_caches'):
            original_recv = getattr(worker_class, 'recv_kv_caches')
            _original_methods[class_name]['recv_kv_caches'] = original_recv

            wrapped_recv = _create_recv_kv_wrapper(original_recv, sdk)
            setattr(worker_class, 'recv_kv_caches', wrapped_recv)
            patched_methods.append('recv_kv_caches')

        # Patch send_kv_caches方法
        if hasattr(worker_class, 'send_kv_caches'):
            original_send = getattr(worker_class, 'send_kv_caches')
            _original_methods[class_name]['send_kv_caches'] = original_send

            wrapped_send = _create_send_kv_wrapper(original_send, sdk)
            setattr(worker_class, 'send_kv_caches', wrapped_send)
            patched_methods.append('send_kv_caches')

        # Patch get_finished方法（可选）
        if hasattr(worker_class, 'get_finished'):
            original_finished = getattr(worker_class, 'get_finished')
            _original_methods[class_name]['get_finished'] = original_finished

            wrapped_finished = _create_get_finished_wrapper(original_finished, sdk)
            setattr(worker_class, 'get_finished', wrapped_finished)
            patched_methods.append('get_finished')

        # 添加XConnector属性
        if not hasattr(worker_class, '_xconnector_sdk'):
            setattr(worker_class, '_xconnector_sdk', sdk)

        _patched_classes.add(worker_class)

        if patched_methods:
            logger.info(f"✓ Patched {class_name} methods: {patched_methods}")
            return True
        else:
            logger.debug(f"No methods to patch in {class_name}")
            return False

    except Exception as e:
        logger.error(f"Failed to patch {worker_class.__name__}: {e}")
        return False


def patch_existing_workers(sdk) -> int:
    """
    Patch所有已经导入的Worker类

    Args:
        sdk: MinimalXConnectorSDK实例

    Returns:
        int: 成功patch的类数量
    """
    patched_count = 0

    try:
        # 遍历所有已导入的模块
        for module_name, module in sys.modules.items():
            if not module:
                continue

            try:
                # 遍历模块中的所有类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # 检查是否是类且是Worker类
                    if (inspect.isclass(attr) and
                            _is_worker_class(attr) and
                            _has_kv_methods(attr)):

                        if patch_worker_class(attr, sdk):
                            patched_count += 1

            except Exception as e:
                logger.debug(f"Error processing module {module_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error patching existing workers: {e}")

    return patched_count


def setup_import_hooks(sdk):
    """
    设置import钩子，自动patch未来导入的Worker类

    Args:
        sdk: MinimalXConnectorSDK实例
    """
    try:
        # 保存原始的__import__函数
        original_import = __builtins__.__import__

        def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
            """带hook的import函数"""
            try:
                # 调用原始import
                module = original_import(name, globals, locals, fromlist, level)

                # 检查新导入的模块是否包含Worker类
                if module and hasattr(module, '__dict__'):
                    for attr_name, attr in module.__dict__.items():
                        if (inspect.isclass(attr) and
                                _is_worker_class(attr) and
                                _has_kv_methods(attr) and
                                attr not in _patched_classes):
                            logger.debug(f"Auto-patching newly imported worker: {attr.__name__}")
                            patch_worker_class(attr, sdk)

                return module

            except Exception as e:
                logger.debug(f"Error in import hook: {e}")
                return original_import(name, globals, locals, fromlist, level)

        # 安装hook
        __builtins__.__import__ = hooked_import
        logger.debug("✓ Import hooks installed")

    except Exception as e:
        logger.error(f"Failed to setup import hooks: {e}")


def unpatch_worker_class(worker_class: type) -> bool:
    """
    恢复Worker类的原始方法（用于清理）

    Args:
        worker_class: 要恢复的Worker类

    Returns:
        bool: 恢复是否成功
    """
    try:
        class_name = worker_class.__name__

        if class_name not in _original_methods:
            return False

        original_methods = _original_methods[class_name]

        for method_name, original_method in original_methods.items():
            setattr(worker_class, method_name, original_method)

        _patched_classes.discard(worker_class)
        del _original_methods[class_name]

        logger.debug(f"Unpatched worker class: {class_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to unpatch {worker_class.__name__}: {e}")
        return False


def get_patch_status() -> Dict[str, Any]:
    """获取patch状态信息"""
    return {
        "patched_classes_count": len(_patched_classes),
        "patched_classes": [cls.__name__ for cls in _patched_classes],
        "original_methods_saved": len(_original_methods)
    }


# 导出
__all__ = [
    'patch_worker_class',
    'patch_existing_workers',
    'setup_import_hooks',
    'unpatch_worker_class',
    'get_patch_status'
]