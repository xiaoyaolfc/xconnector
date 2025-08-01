# integrations/dynamo/extension_loader.py
"""
XConnector Extension Loader for AI-Dynamo

支持嵌入式和远程模式的扩展加载器
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, Optional, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XConnectorExtension:
    """嵌入式XConnector扩展"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.connector = None

    def load(self) -> bool:
        """加载XConnector（嵌入式模式）"""
        if not self.enabled:
            logger.info("XConnector extension is disabled")
            return False

        try:
            # Add XConnector to path if needed
            xconnector_path = self.config.get("xconnector_path")
            if xconnector_path and xconnector_path not in sys.path:
                sys.path.insert(0, xconnector_path)

            # Import XConnector
            from xconnector.core.connector import XConnector, AdapterConfig, AdapterType

            # Initialize XConnector
            self.connector = XConnector()

            # Load configured adapters
            adapters_config = self.config.get("adapters", {})
            for name, adapter_cfg in adapters_config.items():
                if adapter_cfg.get("enabled", True):
                    self._load_adapter(name, adapter_cfg)

            # Start XConnector
            asyncio.create_task(self.connector.start())

            logger.info("XConnector extension loaded successfully (embedded mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to load XConnector: {e}")
            if self.config.get("fail_on_error", False):
                raise
            return False

    def _load_adapter(self, name: str, config: Dict[str, Any]):
        """加载单个适配器"""
        try:
            from xconnector.core.connector import AdapterConfig, AdapterType

            adapter_config = AdapterConfig(
                name=name,
                type=AdapterType(config.get("type", "inference")),
                class_path=config["class_path"],
                config=config.get("config", {})
            )

            asyncio.create_task(self.connector.load_adapter(adapter_config))

            logger.info(f"Loaded adapter: {name}")

        except Exception as e:
            logger.error(f"Failed to load adapter {name}: {e}")


class RemoteXConnectorExtension:
    """远程XConnector扩展（通过HTTP API调用）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.service_url = config.get("service_url")
        self.client = None
        self._session = None

    def load(self) -> bool:
        """初始化远程连接"""
        if not self.enabled:
            logger.info("Remote XConnector extension is disabled")
            return False

        if not self.service_url:
            logger.error("Remote mode requires service_url")
            return False

        try:
            # Test connection using requests (synchronous)
            import requests

            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to XConnector service at {self.service_url}")

                # Initialize async client for later use
                self._init_async_client()

                return True
            else:
                logger.error(f"XConnector service unhealthy: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to XConnector service: {e}")
            if self.config.get("fail_on_error", False):
                raise
            return False

    def _init_async_client(self):
        """初始化异步HTTP客户端"""
        try:
            import httpx
            self.client = httpx.AsyncClient(base_url=self.service_url)
        except ImportError:
            logger.warning("httpx not available, using requests for sync operations")

    async def route_message(self, source: str, target: str, method: str, **kwargs) -> Any:
        """通过远程XConnector服务路由消息"""
        if not self.client:
            # Fallback to synchronous requests
            return self._route_message_sync(source, target, method, **kwargs)

        try:
            response = await self.client.post(
                "/route",
                json={
                    "source": source,
                    "target": target,
                    "method": method,
                    "params": kwargs
                }
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                return result.get("result")
            else:
                raise Exception(result.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Remote routing failed: {e}")
            raise

    def _route_message_sync(self, source: str, target: str, method: str, **kwargs) -> Any:
        """同步版本的消息路由（fallback）"""
        try:
            import requests

            response = requests.post(
                f"{self.service_url}/route",
                json={
                    "source": source,
                    "target": target,
                    "method": method,
                    "params": kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                return result.get("result")
            else:
                raise Exception(result.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Sync remote routing failed: {e}")
            raise


class ExtensionLoader:
    """主扩展加载器"""

    _instance = None
    _extensions = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_extensions(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从配置加载扩展

        Args:
            config: Worker配置字典

        Returns:
            Dict of loaded extensions
        """
        loader = cls()
        extensions_config = config.get("extensions", {})

        # Check environment variable override
        if os.getenv("DISABLE_ALL_EXTENSIONS", "").lower() == "true":
            logger.info("All extensions disabled by environment variable")
            return {}

        # Load XConnector extension if configured
        xconnector_config = extensions_config.get("xconnector", {})

        # Environment variable override for XConnector
        if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
            xconnector_config["enabled"] = True
        elif os.getenv("DISABLE_XCONNECTOR", "").lower() == "true":
            xconnector_config["enabled"] = False

        if xconnector_config:
            loader._load_xconnector(xconnector_config)

        return cls._extensions

    def _load_xconnector(self, config: Dict[str, Any]):
        """加载XConnector扩展"""
        service_mode = config.get("service_mode", "embedded")

        try:
            if service_mode == "remote":
                # 使用远程XConnector服务
                extension = RemoteXConnectorExtension(config)
                logger.info("Loading XConnector in remote mode")
            else:
                # 嵌入式模式
                extension = XConnectorExtension(config)
                logger.info("Loading XConnector in embedded mode")

            if extension.load():
                self._extensions["xconnector"] = extension
                logger.info(f"XConnector extension registered in {service_mode} mode")
            else:
                logger.warning(f"Failed to load XConnector extension in {service_mode} mode")

        except Exception as e:
            logger.error(f"Failed to load XConnector extension: {e}")
            if config.get("fail_on_error", False):
                raise

    @classmethod
    def get_extension(cls, name: str) -> Optional[Any]:
        """获取已加载的扩展"""
        return cls._extensions.get(name)

    @classmethod
    def inject_into_worker(cls, worker_instance: Any) -> None:
        """
        将扩展注入到worker实例中

        Args:
            worker_instance: Dynamo worker实例
        """
        xconnector_ext = cls.get_extension("xconnector")
        if xconnector_ext:
            # Add XConnector to worker
            worker_instance.xconnector = xconnector_ext.connector if hasattr(xconnector_ext,
                                                                             'connector') else xconnector_ext
            worker_instance.xconnector_enabled = True
            worker_instance.xconnector_mode = "remote" if isinstance(xconnector_ext,
                                                                     RemoteXConnectorExtension) else "embedded"

            # Wrap methods if needed
            cls._wrap_worker_methods(worker_instance)

            logger.info(f"XConnector injected into worker (mode: {worker_instance.xconnector_mode})")
        else:
            logger.warning("XConnector extension not available for injection")

    @classmethod
    def _wrap_worker_methods(cls, worker: Any) -> None:
        """包装worker方法以集成XConnector"""
        # Store original methods
        if hasattr(worker, 'recv_kv_caches'):
            worker._original_recv_kv_caches = worker.recv_kv_caches
            worker.recv_kv_caches = cls._create_wrapped_recv_kv(worker)

        if hasattr(worker, 'send_kv_caches'):
            worker._original_send_kv_caches = worker.send_kv_caches
            worker.send_kv_caches = cls._create_wrapped_send_kv(worker)

    @staticmethod
    def _create_wrapped_recv_kv(worker: Any) -> Callable:
        """创建包装的recv_kv_caches方法"""

        async def wrapped_recv_kv_caches(*args, **kwargs):
            # Try XConnector first if enabled
            if getattr(worker, 'xconnector_enabled', False) and worker.xconnector:
                try:
                    # Determine how to call XConnector based on mode
                    if getattr(worker, 'xconnector_mode', 'embedded') == 'remote':
                        # Remote mode: call via HTTP API
                        result = await worker.xconnector.route_message(
                            source="vllm",
                            target="lmcache",
                            method="retrieve_kv",
                            model_input=args[1] if len(args) > 1 else kwargs.get('model_input'),
                            kv_caches=args[2] if len(args) > 2 else kwargs.get('kv_caches')
                        )
                    else:
                        # Embedded mode: direct call
                        result = await worker.xconnector.route_message(
                            source="vllm",
                            target="lmcache",
                            method="retrieve_kv",
                            model_input=args[1] if len(args) > 1 else kwargs.get('model_input'),
                            kv_caches=args[2] if len(args) > 2 else kwargs.get('kv_caches')
                        )

                    if result and result.get("found"):
                        return (
                            result.get("hidden_states"),
                            result.get("skip_forward", False),
                            result.get("updated_input", args[1] if len(args) > 1 else kwargs.get('model_input'))
                        )
                except Exception as e:
                    logger.debug(f"XConnector recv_kv failed, falling back: {e}")

            # Fall back to original method
            return await worker._original_recv_kv_caches(*args, **kwargs)

        return wrapped_recv_kv_caches

    @staticmethod
    def _create_wrapped_send_kv(worker: Any) -> Callable:
        """创建包装的send_kv_caches方法"""

        async def wrapped_send_kv_caches(*args, **kwargs):
            # Call original method first
            result = await worker._original_send_kv_caches(*args, **kwargs)

            # Then send to XConnector if enabled
            if getattr(worker, 'xconnector_enabled', False) and worker.xconnector:
                try:
                    # Determine how to call XConnector based on mode
                    if getattr(worker, 'xconnector_mode', 'embedded') == 'remote':
                        # Remote mode: call via HTTP API
                        await worker.xconnector.route_message(
                            source="vllm",
                            target="lmcache",
                            method="store_kv",
                            model_input=args[1] if len(args) > 1 else kwargs.get('model_input'),
                            kv_caches=args[2] if len(args) > 2 else kwargs.get('kv_caches'),
                            hidden_states=args[3] if len(args) > 3 else kwargs.get('hidden_or_intermediate_states')
                        )
                    else:
                        # Embedded mode: direct call
                        await worker.xconnector.route_message(
                            source="vllm",
                            target="lmcache",
                            method="store_kv",
                            model_input=args[1] if len(args) > 1 else kwargs.get('model_input'),
                            kv_caches=args[2] if len(args) > 2 else kwargs.get('kv_caches'),
                            hidden_states=args[3] if len(args) > 3 else kwargs.get('hidden_or_intermediate_states')
                        )
                except Exception as e:
                    logger.debug(f"XConnector send_kv failed: {e}")

            return result

        return wrapped_send_kv_caches


# 便捷函数用于简化集成
def load_extensions_for_worker(worker_class: type) -> type:
    """
    装饰器：自动为worker类加载扩展

    Usage:
        @load_extensions_for_worker
        class VllmWorker:
            ...
    """
    original_init = worker_class.__init__

    def wrapped_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)

        # Load extensions if config is available
        if hasattr(self, 'config') and isinstance(self.config, dict):
            ExtensionLoader.load_extensions(self.config)
            ExtensionLoader.inject_into_worker(self)

    worker_class.__init__ = wrapped_init
    return worker_class


# 简单钩子用于Dynamo配置处理
def process_worker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理worker配置以加载扩展

    可以从Dynamo的配置加载管道中调用
    """
    ExtensionLoader.load_extensions(config)
    return config


# 测试工具函数
def test_xconnector_connection(service_url: str) -> bool:
    """测试XConnector服务连接"""
    try:
        import requests
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # 测试脚本
    import argparse

    parser = argparse.ArgumentParser(description="Test XConnector Extension Loader")
    parser.add_argument("--service-url", default="http://localhost:8081",
                        help="XConnector service URL")
    parser.add_argument("--mode", choices=["embedded", "remote"], default="remote",
                        help="Extension mode to test")

    args = parser.parse_args()

    if args.mode == "remote":
        print(f"Testing connection to {args.service_url}...")
        if test_xconnector_connection(args.service_url):
            print("✓ Connection successful")
        else:
            print("✗ Connection failed")

    # Test extension loading
    test_config = {
        "extensions": {
            "xconnector": {
                "enabled": True,
                "service_mode": args.mode,
                "service_url": args.service_url if args.mode == "remote" else None,
                "fail_on_error": False
            }
        }
    }

    print(f"Testing {args.mode} mode extension loading...")
    extensions = ExtensionLoader.load_extensions(test_config)

    if "xconnector" in extensions:
        print("✓ XConnector extension loaded successfully")
    else:
        print("✗ XConnector extension failed to load")