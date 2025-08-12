# integrations/dynamo/worker_injector.py
"""
Worker类Monkey Patch注入器 (修复版)

修复了 import hooks 中的 __import__ 访问问题
"""

import sys
import logging
import inspect
import functools
from typing import Any, Dict, List, Optional, Callable, Set
import asyncio
import builtins

# 获取logger
logger = logging.getLogger('xconnector.dynamo.worker_injector')

# 已经patch过的类集合，避免重复patch
_patched_classes: Set[type] = set()
_original_methods: Dict[str, Dict[str, Callable]] = {}
_original_import: Optional[Callable] = None


def _is_worker_class(cls: type) -> bool:
    """
    检测是否是Worker类

    Args:
        cls: 要检测的类

    Returns:
        bool: 是否是Worker类
    """
    try:
        # 先检查是否是类
        if not inspect.isclass(cls):
            return False

        class_name = cls.__name__.lower()
        module_name = getattr(cls, '__module__', '').lower() if hasattr(cls, '__module__') else ''

        # 检查是否有KV缓存方法 - 最可靠的检测方式
        if _has_kv_methods(cls):
            return True

        # 检查是否有其他关键方法
        key_methods = ['get_finished']
        if any(hasattr(cls, method) for method in key_methods):
            return True

        # 检查类名模式 - 但排除明确的非Worker类
        if 'nonworker' not in class_name and 'regularclass' not in class_name:
            worker_patterns = ['worker', 'vllm', 'prefill', 'decode']
            if any(pattern in class_name for pattern in worker_patterns):
                return True

        # 检查模块名模式 - 但排除测试模块
        if not ('test' in module_name or 'mock' in module_name):
            module_patterns = ['worker', 'vllm', 'dynamo']
            if any(pattern in module_name for pattern in module_patterns):
                return True

        return False

    except Exception as e:
        # 发生异常时返回False，避免误判
        logger.debug(f"Error checking worker class {getattr(cls, '__name__', 'unknown')}: {e}")
        return False


def _has_kv_methods(cls: type) -> bool:
    """检查类是否有KV缓存相关方法"""
    try:
        kv_methods = ['recv_kv_caches', 'send_kv_caches']
        return any(hasattr(cls, method) for method in kv_methods)
    except Exception:
        return False


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
            if not sdk or not hasattr(sdk, 'cache_adapter') or not sdk.cache_adapter:
                return await original_method(self, *args, **kwargs)

            # 尝试从XConnector缓存检索
            cache_result = await sdk.cache_adapter.retrieve_kv_cache(*args, **kwargs)
            if cache_result is not None:
                logger.debug("KV cache hit from XConnector")
                return cache_result

            # 缓存未命中，调用原方法
            result = await original_method(self, *args, **kwargs)

            # 将结果存入缓存（异步）
            if result is not None:
                try:
                    await sdk.cache_adapter.store_kv_cache(result, *args, **kwargs)
                except Exception as cache_error:
                    logger.debug(f"Failed to cache result: {cache_error}")

            return result

        except Exception as e:
            logger.debug(f"Error in recv_kv_caches wrapper: {e}")
            # 出错时回退到原方法
            return await original_method(self, *args, **kwargs)

    return wrapped_recv_kv_caches


def _create_send_kv_wrapper(original_method: Callable, sdk) -> Callable:
    """
    创建send_kv_caches方法的包装器

    Args:
        original_method: 原始方法
        sdk: MinimalXConnectorSDK实例

    Returns:x`
        包装后的方法
    """

    @functools.wraps(original_method)
    async def wrapped_send_kv_caches(self, *args, **kwargs):
        """
        包装的KV发送方法

        先存储到XConnector缓存，然后调用原方法
        """
        try:
            # 检查SDK和KV处理器是否可用
            if sdk and hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                # 尝试存储到XConnector缓存
                try:
                    await sdk.cache_adapter.store_kv_cache(*args, **kwargs)
                    logger.debug("KV cache stored to XConnector")
                except Exception as cache_error:
                    logger.debug(f"Failed to store to cache: {cache_error}")

            # 调用原方法
            return await original_method(self, *args, **kwargs)

        except Exception as e:
            logger.debug(f"Error in send_kv_caches wrapper: {e}")
            # 出错时回退到原方法
            return await original_method(self, *args, **kwargs)

    return wrapped_send_kv_caches


def patch_worker_class(worker_class: type, sdk) -> bool:
    """
    对单个Worker类进行monkey patch

    Args:
        worker_class: 要patch的Worker类
        sdk: MinimalXConnectorSDK实例

    Returns:
        bool: patch是否成功
    """
    try:
        # 检查是否已经patch过
        if worker_class in _patched_classes:
            logger.debug(f"Class {worker_class.__name__} already patched")
            return True

        # 检查是否确实是Worker类
        if not _is_worker_class(worker_class):
            logger.debug(f"Class {worker_class.__name__} is not a Worker class")
            return False

        class_name = worker_class.__name__

        # 保存原始方法
        original_methods = {}

        # Patch recv_kv_caches方法
        if hasattr(worker_class, 'recv_kv_caches'):
            original_recv = getattr(worker_class, 'recv_kv_caches')
            wrapped_recv = _create_recv_kv_wrapper(original_recv, sdk)
            setattr(worker_class, 'recv_kv_caches', wrapped_recv)
            original_methods['recv_kv_caches'] = original_recv

        # Patch send_kv_caches方法
        if hasattr(worker_class, 'send_kv_caches'):
            original_send = getattr(worker_class, 'send_kv_caches')
            wrapped_send = _create_send_kv_wrapper(original_send, sdk)
            setattr(worker_class, 'send_kv_caches', wrapped_send)
            original_methods['send_kv_caches'] = original_send

        # 只有实际patch了方法才标记为已patch
        if original_methods:
            _patched_classes.add(worker_class)
            _original_methods[class_name] = original_methods
            logger.info(f"Successfully patched worker class: {class_name}")
            return True
        else:
            logger.debug(f"No methods to patch in class: {class_name}")
            return False

    except Exception as e:
        logger.error(f"Failed to patch worker class {getattr(worker_class, '__name__', 'unknown')}: {e}")
        return False


def patch_existing_workers(sdk) -> int:
    """
    Patch当前已导入的所有Worker类

    Args:
        sdk: MinimalXConnectorSDK实例

    Returns:
        int: patch成功的类数量
    """
    patched_count = 0

    try:
        # 创建sys.modules的快照，避免迭代过程中字典大小改变
        modules_snapshot = list(sys.modules.items())

        for module_name, module in modules_snapshot:
            if module is None:
                continue

            try:
                # 安全地获取模块属性
                if not hasattr(module, '__dict__'):
                    continue

                # 创建属性快照
                attr_items = list(module.__dict__.items())

                for attr_name, attr in attr_items:
                    try:
                        # 安全检查属性
                        if attr is None:
                            continue

                        # 检查是否是类并且是Worker类
                        if (inspect.isclass(attr) and
                                _is_worker_class(attr) and
                                attr not in _patched_classes):

                            if patch_worker_class(attr, sdk):
                                patched_count += 1
                                logger.debug(f"Patched worker class: {module_name}.{attr_name}")

                    except Exception as attr_error:
                        # 记录但不中断处理
                        logger.debug(f"Error processing attribute {attr_name}: {attr_error}")

            except Exception as module_error:
                # 记录但不中断处理
                logger.debug(f"Error processing module {module_name}: {module_error}")

    except Exception as e:
        logger.error(f"Error patching existing workers: {e}")

    return patched_count


def setup_import_hooks(sdk):
    """
    设置import钩子，自动patch未来导入的Worker类 (修复版)

    Args:
        sdk: MinimalXConnectorSDK实例
    """
    global _original_import

    try:
        # 修复：正确访问 builtins 中的 __import__
        if _original_import is None:
            _original_import = builtins.__import__

        def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
            """带hook的import函数"""
            try:
                # 调用原始import
                module = _original_import(name, globals, locals, fromlist, level)

                # 检查新导入的模块是否包含Worker类
                if module and hasattr(module, '__dict__'):
                    try:
                        # 安全地检查模块属性
                        attr_items = list(module.__dict__.items())
                        for attr_name, attr in attr_items:
                            try:
                                if (inspect.isclass(attr) and
                                        _is_worker_class(attr) and
                                        attr not in _patched_classes):
                                    logger.debug(f"Auto-patching newly imported worker: {attr.__name__}")
                                    patch_worker_class(attr, sdk)
                            except Exception as attr_error:
                                logger.debug(f"Error checking attribute {attr_name}: {attr_error}")
                    except Exception as module_error:
                        logger.debug(f"Error checking module {name}: {module_error}")

                return module

            except Exception as e:
                logger.debug(f"Error in import hook: {e}")
                # 出错时回退到原始import
                return _original_import(name, globals, locals, fromlist, level)

        # 安装hook - 修复：使用builtins而不是__builtins__
        builtins.__import__ = hooked_import
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

        # 检查是否曾经被patch过
        if class_name not in _original_methods:
            logger.debug(f"Class {class_name} was not patched")
            return False

        original_methods = _original_methods[class_name]

        for method_name, original_method in original_methods.items():
            setattr(worker_class, method_name, original_method)

        _patched_classes.discard(worker_class)
        del _original_methods[class_name]

        logger.debug(f"Unpatched worker class: {class_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to unpatch {getattr(worker_class, '__name__', 'unknown')}: {e}")
        return False


def cleanup_import_hooks():
    """清理import hooks"""
    global _original_import
    try:
        if _original_import is not None:
            builtins.__import__ = _original_import
            _original_import = None
            logger.debug("✓ Import hooks cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup import hooks: {e}")


def get_patch_status() -> Dict[str, Any]:
    """获取patch状态信息"""
    return {
        "patched_classes_count": len(_patched_classes),
        "patched_classes": [cls.__name__ for cls in _patched_classes],
        "original_methods_saved": len(_original_methods),
        "import_hooks_active": _original_import is not None
    }


# 导出
__all__ = [
    'patch_worker_class',
    'patch_existing_workers',
    'setup_import_hooks',
    'unpatch_worker_class',
    'cleanup_import_hooks',
    'get_patch_status'
]