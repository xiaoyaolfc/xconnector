# xconnector/integrations/dynamo/autopatch.py
"""
XConnector-Dynamo 自动patch入口

用户只需要在Dynamo启动脚本开头导入此模块：
    import xconnector.integrations.dynamo.autopatch

该模块会自动：
1. 检测是否在Dynamo环境中运行
2. 检测配置文件中的XConnector配置
3. 初始化最小化的XConnector SDK
4. Monkey patch相关的Worker类
5. 注册生命周期钩子

设计原则：
- 零侵入：出错时不影响Dynamo正常运行
- 智能检测：自动判断是否需要启用
- 延迟初始化：只在需要时才初始化组件
"""

import os
import sys
import logging
import threading
from typing import Dict, Any, Optional
import inspect


# 获取logger
def _get_logger():
    """获取专用logger"""
    logger = logging.getLogger('xconnector.dynamo.autopatch')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = _get_logger()

# 全局状态
_integration_initialized = False
_integration_lock = threading.Lock()
_minimal_sdk = None


def _detect_dynamo_environment() -> bool:
    """
    检测是否在Dynamo环境中运行

    检测方法：
    1. 检查环境变量
    2. 检查调用栈中是否有dynamo相关模块
    3. 检查当前目录是否有dynamo配置文件
    4. 检查命令行参数
    """
    try:
        # 方法1: 检查环境变量
        dynamo_env_vars = [
            'DYNAMO_WORKER', 'DYNAMO_CONFIG', 'DYNAMO_MODE',
            'VLLM_WORKER', 'PREFILL_WORKER'
        ]

        for env_var in dynamo_env_vars:
            if os.getenv(env_var):
                logger.debug(f"Detected Dynamo environment via {env_var}")
                return True

        # 方法2: 检查调用栈
        for frame_info in inspect.stack():
            filename = frame_info.filename.lower()
            if any(keyword in filename for keyword in ['dynamo', 'vllm_worker', 'worker']):
                logger.debug(f"Detected Dynamo environment via call stack: {filename}")
                return True

        # 方法3: 检查配置文件（使用config_detector的逻辑）
        from .config_detector import detect_config_files
        config_files = detect_config_files()
        if config_files:
            logger.debug(f"Detected Dynamo environment via config files: {[str(f) for f in config_files]}")
            return True

        # 方法4: 检查命令行参数
        if any('dynamo' in arg.lower() or 'worker' in arg.lower() for arg in sys.argv):
            logger.debug("Detected Dynamo environment via command line args")
            return True

        return False

    except Exception as e:
        logger.debug(f"Error detecting Dynamo environment: {e}")
        return False


def _initialize_minimal_sdk(config: Dict[str, Any]) -> bool:
    """
    初始化最小化的XConnector SDK

    Args:
        config: XConnector配置

    Returns:
        bool: 初始化是否成功
    """
    global _minimal_sdk

    try:
        # 延迟导入，避免循环依赖
        from .minimal_sdk import MinimalXConnectorSDK

        # 创建最小SDK实例
        _minimal_sdk = MinimalXConnectorSDK(config)

        # 异步初始化
        import asyncio

        async def async_init():
            success = await _minimal_sdk.initialize()
            if success:
                logger.info("✓ Minimal XConnector SDK initialized successfully")
            else:
                logger.warning("⚠ Minimal XConnector SDK initialization failed")
            return success

        # 如果已经在异步上下文中，直接初始化
        try:
            loop = asyncio.get_running_loop()
            # 创建任务但不等待，让它在后台运行
            loop.create_task(async_init())
            logger.info("✓ XConnector SDK initialization started (async)")
            return True
        except RuntimeError:
            # 没有运行的事件循环，使用同步方式
            try:
                result = asyncio.run(async_init())
                return result
            except Exception as e:
                logger.warning(f"⚠ Async initialization failed: {e}")
                # 回退到同步初始化
                _minimal_sdk.initialize_sync()
                logger.info("✓ XConnector SDK initialized (sync fallback)")
                return True

    except ImportError as e:
        logger.error(f"✗ Failed to import XConnector components: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to initialize minimal XConnector SDK: {e}")
        return False


def _patch_worker_classes():
    """
    Monkey patch相关的Worker类

    这个函数会：
    1. 检测已导入的Worker类
    2. 对这些类进行monkey patch
    3. 设置钩子以patch未来导入的类
    """
    try:
        from .worker_injector import patch_existing_workers, setup_import_hooks

        # Patch已经导入的Worker类
        patched_count = patch_existing_workers(_minimal_sdk)
        if patched_count > 0:
            logger.info(f"✓ Patched {patched_count} existing Worker classes")

        # 设置import钩子，patch未来导入的Worker类
        setup_import_hooks(_minimal_sdk)
        logger.info("✓ Import hooks installed for future Worker classes")

        return True

    except ImportError as e:
        logger.error(f"✗ Failed to import worker injector: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to patch Worker classes: {e}")
        return False


def _setup_lifecycle_hooks():
    """设置生命周期钩子（可选功能）"""
    try:
        # 尝试导入生命周期管理器（如果存在的话）
        from .lifecycle_manager import setup_hooks
        setup_hooks(_minimal_sdk)
        logger.debug("✓ Lifecycle hooks installed")
        return True
    except ImportError:
        logger.debug("Lifecycle manager not available, skipping hooks")
        return True
    except Exception as e:
        logger.debug(f"Failed to setup lifecycle hooks: {e}")
        return True  # 非关键功能，不影响主流程


def _auto_initialize():
    """
    自动初始化XConnector集成

    这是模块导入时执行的主要函数
    """
    global _integration_initialized

    # 使用锁确保只初始化一次
    with _integration_lock:
        if _integration_initialized:
            return

        try:
            logger.debug("Starting XConnector auto-initialization...")

            # 步骤1: 检测Dynamo环境
            if not _detect_dynamo_environment():
                logger.debug("Not in Dynamo environment, skipping XConnector initialization")
                return

            logger.info("✓ Detected Dynamo environment")

            # 步骤2: 检测XConnector配置（使用专门的config_detector）
            from .config_detector import detect_xconnector_config, validate_xconnector_config

            config = detect_xconnector_config()
            if not config:
                logger.info("No XConnector configuration found, skipping initialization")
                return

            # 验证配置
            is_valid, errors = validate_xconnector_config(config)
            if not is_valid:
                logger.warning(f"Invalid XConnector config: {errors}")
                if not config.get('ignore_validation_errors', False):
                    return

            if not config.get('enabled', False):
                logger.info("XConnector is disabled in configuration")
                return

            logger.info(f"✓ Found valid XConnector config: enabled={config.get('enabled')}")

            # 设置日志级别
            log_level = config.get('log_level', 'INFO').upper()
            if hasattr(logging, log_level):
                logger.setLevel(getattr(logging, log_level))

            # 步骤3: 初始化最小SDK
            if not _initialize_minimal_sdk(config):
                logger.warning("⚠ XConnector SDK initialization failed, continuing without XConnector")
                return

            # 步骤4: Patch Worker类
            if not _patch_worker_classes():
                logger.warning("⚠ Worker patching failed, XConnector features may not work")
                return

            # 步骤5: 设置生命周期钩子（可选）
            _setup_lifecycle_hooks()

            _integration_initialized = True
            logger.info("🎉 XConnector-Dynamo integration completed successfully!")

        except Exception as e:
            logger.error(f"✗ XConnector auto-initialization failed: {e}")
            logger.error("Dynamo will continue running without XConnector integration")

            # 确保不会因为XConnector的问题影响Dynamo启动
            if config and config.get('fail_on_error', False):
                raise

        finally:
            # 确保状态正确设置
            _integration_initialized = True


def get_minimal_sdk():
    """获取当前的最小SDK实例（供其他模块使用）"""
    return _minimal_sdk


def is_integration_enabled() -> bool:
    """检查集成是否已启用"""
    return _integration_initialized and _minimal_sdk is not None


def get_integration_status() -> Dict[str, Any]:
    """获取集成状态信息"""
    try:
        from .config_detector import detect_xconnector_config
        config_found = detect_xconnector_config() is not None
    except:
        config_found = False

    return {
        "initialized": _integration_initialized,
        "sdk_available": _minimal_sdk is not None,
        "sdk_ready": _minimal_sdk.is_ready() if _minimal_sdk else False,
        "dynamo_environment": _detect_dynamo_environment(),
        "config_found": config_found
    }


# 模块导入时自动执行初始化
try:
    _auto_initialize()
except Exception as e:
    # 确保即使初始化失败也不会影响模块导入
    logger.error(f"Critical error during XConnector auto-initialization: {e}")
    logger.error("XConnector integration disabled, Dynamo will run normally")

# 导出公共接口
__all__ = [
    'get_minimal_sdk',
    'is_integration_enabled',
    'get_integration_status'
]