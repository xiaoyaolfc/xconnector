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
import inspect
from typing import Dict, Any, Optional
from pathlib import Path


# ============================================================
# 依赖检测
# ============================================================

def check_optional_dependencies():
    """检测可选依赖，缺失时返回 None"""
    deps = {}

    # 检查 etcd3
    try:
        import etcd3
        deps['etcd3'] = etcd3
    except ImportError:
        deps['etcd3'] = None
        logging.debug("etcd3 not available, etcd functionality disabled")

    # 检查 nats
    try:
        import nats
        deps['nats'] = nats
    except ImportError:
        deps['nats'] = None
        logging.debug("nats not available, NATS functionality disabled")

    # 检查 aiohttp
    try:
        import aiohttp
        deps['aiohttp'] = aiohttp
    except ImportError:
        deps['aiohttp'] = None
        logging.debug("aiohttp not available, HTTP client disabled")

    # 检查 aiofiles
    try:
        import aiofiles
        deps['aiofiles'] = aiofiles
    except ImportError:
        deps['aiofiles'] = None
        logging.debug("aiofiles not available, async file I/O disabled")

    return deps


# 全局依赖状态
OPTIONAL_DEPS = check_optional_dependencies()


# ============================================================
# Logger配置
# ============================================================

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

# ============================================================
# 全局状态管理
# ============================================================

_integration_initialized = False
_integration_lock = threading.Lock()
_minimal_sdk = None


# ============================================================
# 环境检测
# ============================================================

def _detect_dynamo_environment() -> bool:
    """
    检测是否在Dynamo环境中运行（增强版）

    更宽松的检测策略，避免错过合法的Dynamo环境
    """
    try:
        # 方法1: 检查环境变量（扩展列表）
        dynamo_env_vars = [
            'DYNAMO_WORKER', 'DYNAMO_CONFIG', 'DYNAMO_MODE',
            'VLLM_WORKER', 'PREFILL_WORKER',
            'ENABLE_XCONNECTOR',  # 添加XConnector特定变量
            'XCONNECTOR_ENABLED',
            'XCONNECTOR_CONFIG_FILE'
        ]

        for env_var in dynamo_env_vars:
            if os.getenv(env_var):
                logger.debug(f"Detected Dynamo environment via {env_var}")
                return True

        # 方法2: 检查工作目录路径
        cwd = str(Path.cwd())
        dynamo_paths = [
            '/workspace/example/llm',  # Dynamo运行目录
            '/workspace',  # 工作空间目录
            'example/llm',  # 相对路径
            'dynamo',
            'vllm'
        ]

        for path in dynamo_paths:
            if path in cwd:
                logger.debug(f"Detected Dynamo environment via path: {cwd}")
                return True

        # 方法3: 检查调用栈
        for frame_info in inspect.stack():
            filename = frame_info.filename.lower()
            if any(keyword in filename for keyword in ['dynamo', 'vllm', 'worker', 'example/llm']):
                logger.debug(f"Detected Dynamo environment via call stack: {filename}")
                return True

        # 方法4: 检查配置文件存在
        try:
            from .config_detector import detect_config_files
            config_files = detect_config_files()
            if config_files:
                logger.debug(f"Detected Dynamo environment via config files: {[str(f) for f in config_files]}")
                return True
        except ImportError:
            # config_detector不可用时，直接检查目录
            config_dirs = [
                '/workspace/configs',
                '/workspace/example/llm/configs',
                '/workspace/xconnector/integrations/dynamo/configs'
            ]
            for config_dir in config_dirs:
                if Path(config_dir).exists():
                    logger.debug(f"Detected Dynamo environment via config dir: {config_dir}")
                    return True

        # 方法5: 检查命令行参数
        if any('dynamo' in arg.lower() or 'worker' in arg.lower() or 'llm' in arg.lower() for arg in sys.argv):
            logger.debug("Detected Dynamo environment via command line args")
            return True

        # 方法6: 如果XConnector被明确启用，也认为是Dynamo环境
        if os.getenv('ENABLE_XCONNECTOR', '').lower() in ['true', '1', 'yes']:
            logger.debug("Detected Dynamo environment via ENABLE_XCONNECTOR")
            return True

        return False

    except Exception as e:
        logger.debug(f"Error detecting Dynamo environment: {e}")
        # 如果检测出错，保守地返回True，让后续逻辑决定是否初始化
        return True


# ============================================================
# SDK初始化
# ============================================================

def _initialize_minimal_sdk(config: Dict[str, Any]) -> bool:
    """
    初始化最小化的XConnector SDK（修复版）

    改进的初始化逻辑，更好地处理同步和异步场景
    """
    global _minimal_sdk

    try:
        # 延迟导入，避免循环依赖
        from .minimal_sdk import MinimalXConnectorSDK

        # 创建最小SDK实例
        _minimal_sdk = MinimalXConnectorSDK(config)
        logger.info("✓ Minimal XConnector SDK instance created")

        # 尝试异步初始化
        import asyncio

        async def async_init():
            success = await _minimal_sdk.initialize()
            if success:
                logger.info("✓ Minimal XConnector SDK initialized successfully (async)")
            else:
                logger.warning("⚠ Minimal XConnector SDK initialization failed (async)")
            return success

        # 检查是否已经在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 在现有循环中创建任务（不阻塞）
            task = loop.create_task(async_init())
            logger.info("✓ XConnector SDK initialization task created")

            # 标记为成功（初始化将在后台完成）
            return True

        except RuntimeError:
            # 没有运行的事件循环，尝试同步初始化
            logger.debug("No running event loop, attempting sync initialization")

            # 首先尝试同步初始化方法
            if hasattr(_minimal_sdk, 'initialize_sync'):
                success = _minimal_sdk.initialize_sync()
                if success:
                    logger.info("✓ XConnector SDK initialized (sync)")
                    return True
                else:
                    logger.warning("⚠ Sync initialization failed")

            # 如果没有同步方法或失败，尝试创建新的事件循环
            try:
                success = asyncio.run(async_init())
                return success
            except Exception as e:
                logger.warning(f"⚠ Async initialization with new loop failed: {e}")

                # 最后的回退：只要SDK实例创建成功就认为成功
                if _minimal_sdk and hasattr(_minimal_sdk, 'cache_adapter'):
                    logger.info("✓ XConnector SDK instance ready (without full initialization)")
                    return True
                return False

    except ImportError as e:
        logger.error(f"✗ Failed to import XConnector components: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to initialize minimal XConnector SDK: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# ============================================================
# Worker Patching
# ============================================================

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
        logger.debug(f"Worker injector not available: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to patch Worker classes: {e}")
        return False


# ============================================================
# 生命周期管理
# ============================================================

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


# ============================================================
# 自动初始化
# ============================================================

def _auto_initialize():
    """
    自动初始化XConnector集成（修复版）

    更宽容的初始化策略，确保XConnector能够正确初始化
    """
    global _integration_initialized, _minimal_sdk

    # 使用锁确保只初始化一次
    with _integration_lock:
        if _integration_initialized:
            return

        config = None  # 确保config在finally块中可用

        try:
            logger.info("=" * 60)
            logger.info("Starting XConnector auto-initialization...")
            logger.info(f"Working directory: {Path.cwd()}")

            # 步骤1: 检测Dynamo环境（更宽松）
            if not _detect_dynamo_environment():
                logger.debug("Not in Dynamo environment, skipping XConnector initialization")
                _integration_initialized = True  # 标记为已处理
                return

            logger.info("✓ Detected Dynamo environment")

            # 步骤2: 检测XConnector配置
            from .config_detector import detect_xconnector_config, validate_xconnector_config

            config = detect_xconnector_config()
            if not config:
                # 如果环境变量明确启用了XConnector，创建默认配置
                if os.getenv('ENABLE_XCONNECTOR', '').lower() in ['true', '1', 'yes']:
                    logger.info("ENABLE_XCONNECTOR is set, creating default config")
                    config = {
                        'enabled': True,
                        'mode': 'embedded',
                        'adapters': [
                            {
                                'name': 'lmcache',
                                'type': 'cache',
                                'class_path': 'xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter',
                                'enabled': True,
                                'config': {
                                    'storage_backend': 'memory',
                                    'max_cache_size': 1024
                                }
                            }
                        ]
                    }
                else:
                    logger.info("No XConnector configuration found, skipping initialization")
                    _integration_initialized = True
                    return

            # 验证配置
            is_valid, errors = validate_xconnector_config(config)
            if not is_valid:
                logger.warning(f"Config validation warnings: {errors}")
                # 继续执行，除非明确要求失败
                if config.get('fail_on_validation_error', False):
                    _integration_initialized = True
                    return

            # 检查是否启用
            if not config.get('enabled', True):  # 默认启用
                logger.info("XConnector is disabled in configuration")
                _integration_initialized = True
                return

            logger.info(f"✓ Found valid XConnector config: enabled={config.get('enabled')}")
            if config.get('_config_source'):
                logger.info(f"  Config source: {config.get('_config_source')}")

            # 设置日志级别
            log_level = config.get('log_level', 'INFO').upper()
            if hasattr(logging, log_level):
                logger.setLevel(getattr(logging, log_level))

            # 步骤3: 初始化最小SDK
            if not _initialize_minimal_sdk(config):
                logger.warning("⚠ XConnector SDK initialization failed")
                # 不要失败，让Dynamo继续运行
                _integration_initialized = True
                return

            # 步骤4: Patch Worker类（可选）
            try:
                if not _patch_worker_classes():
                    logger.debug("Worker patching not available or failed")
                    # 继续，不要失败
            except Exception as e:
                logger.debug(f"Worker patching skipped: {e}")

            # 步骤5: 设置生命周期钩子（可选）
            try:
                _setup_lifecycle_hooks()
            except Exception as e:
                logger.debug(f"Lifecycle hooks skipped: {e}")

            logger.info("🎉 XConnector-Dynamo integration completed successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"✗ XConnector auto-initialization failed: {e}")
            logger.error("Dynamo will continue running without XConnector integration")
            import traceback
            logger.debug(traceback.format_exc())

            # 确保不会因为XConnector的问题影响Dynamo启动
            if config and config.get('fail_on_error', False):
                raise

        finally:
            # 确保状态正确设置
            _integration_initialized = True


# ============================================================
# 公共接口
# ============================================================

def get_minimal_sdk():
    """获取当前的最小SDK实例（供其他模块使用）"""
    return _minimal_sdk


def is_integration_enabled() -> bool:
    """检查集成是否已启用"""
    return _integration_initialized and _minimal_sdk is not None


def get_integration_status() -> Dict[str, Any]:
    """
    获取集成状态信息（增强版）

    提供更详细的状态信息用于调试
    """
    try:
        from .config_detector import detect_xconnector_config
        config = detect_xconnector_config()
        config_found = config is not None
        config_source = config.get('_config_source', 'unknown') if config else None
    except:
        config_found = False
        config_source = None

    status = {
        "initialized": _integration_initialized,
        "sdk_available": _minimal_sdk is not None,
        "sdk_ready": _minimal_sdk.is_ready() if _minimal_sdk else False,
        "dynamo_environment": _detect_dynamo_environment(),
        "config_found": config_found,
        "config_source": config_source,
        "working_directory": str(Path.cwd()),
        "environment_vars": {
            "ENABLE_XCONNECTOR": os.getenv('ENABLE_XCONNECTOR'),
            "XCONNECTOR_CONFIG_FILE": os.getenv('XCONNECTOR_CONFIG_FILE'),
            "XCONNECTOR_ENABLED": os.getenv('XCONNECTOR_ENABLED')
        }
    }

    # 添加SDK详细状态
    if _minimal_sdk:
        status["sdk_details"] = {
            "has_cache_adapter": hasattr(_minimal_sdk, 'cache_adapter'),
            "cache_adapter_not_none": _minimal_sdk.cache_adapter is not None if hasattr(_minimal_sdk,
                                                                                        'cache_adapter') else False,
            "initialized_flag": _minimal_sdk.initialized if hasattr(_minimal_sdk, 'initialized') else False
        }

    return status


# ============================================================
# 模块初始化
# ============================================================

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