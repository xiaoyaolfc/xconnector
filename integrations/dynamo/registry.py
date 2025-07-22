# integrations/dynamo/extension_loader.py
"""
XConnector Extension Loader for AI-Dynamo

A minimal, non-invasive extension loader that integrates XConnector
with Dynamo workers through configuration.
"""

import os
import sys
# import logging
from xconnector.utils.xconnector_logging import get_logger
import importlib
from typing import Any, Dict, Optional, Callable
from pathlib import Path

logger = get_logger(__name__)


class XConnectorExtension:
    """XConnector extension for Dynamo workers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.connector = None
        self.adapter = None

    def load(self) -> bool:
        """Load XConnector if enabled"""
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
            import asyncio
            asyncio.create_task(self.connector.start())

            logger.info("XConnector extension loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load XConnector: {e}")
            if self.config.get("fail_on_error", False):
                raise
            return False

    def _load_adapter(self, name: str, config: Dict[str, Any]):
        """Load individual adapter"""
        try:
            from xconnector.core.connector import AdapterConfig, AdapterType

            adapter_config = AdapterConfig(
                name=name,
                type=AdapterType(config.get("type", "inference")),
                class_path=config["class_path"],
                config=config.get("config", {})
            )

            import asyncio
            asyncio.create_task(self.connector.load_adapter(adapter_config))

            logger.info(f"Loaded adapter: {name}")

        except Exception as e:
            logger.error(f"Failed to load adapter {name}: {e}")


class ExtensionLoader:
    """Main extension loader for Dynamo integration"""

    _instance = None
    _extensions = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_extensions(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load extensions from configuration

        Args:
            config: Worker configuration dict

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
        """Load XConnector extension"""
        try:
            extension = XConnectorExtension(config)
            if extension.load():
                self._extensions["xconnector"] = extension
                logger.info("XConnector extension registered")
        except Exception as e:
            logger.error(f"Failed to load XConnector extension: {e}")

    @classmethod
    def get_extension(cls, name: str) -> Optional[Any]:
        """Get loaded extension by name"""
        return cls._extensions.get(name)

    @classmethod
    def inject_into_worker(cls, worker_instance: Any) -> None:
        """
        Inject extensions into worker instance

        Args:
            worker_instance: Dynamo worker instance
        """
        xconnector_ext = cls.get_extension("xconnector")
        if xconnector_ext and xconnector_ext.connector:
            # Add XConnector to worker
            worker_instance.xconnector = xconnector_ext.connector
            worker_instance.xconnector_enabled = True

            # Wrap methods if needed
            cls._wrap_worker_methods(worker_instance)

            logger.info("XConnector injected into worker")

    @classmethod
    def _wrap_worker_methods(cls, worker: Any) -> None:
        """Wrap worker methods to integrate XConnector"""
        # Store original methods
        if hasattr(worker, 'recv_kv_caches'):
            worker._original_recv_kv_caches = worker.recv_kv_caches
            worker.recv_kv_caches = cls._create_wrapped_recv_kv(worker)

        if hasattr(worker, 'send_kv_caches'):
            worker._original_send_kv_caches = worker.send_kv_caches
            worker.send_kv_caches = cls._create_wrapped_send_kv(worker)

    @staticmethod
    def _create_wrapped_recv_kv(worker: Any) -> Callable:
        """Create wrapped recv_kv_caches method"""

        async def wrapped_recv_kv_caches(*args, **kwargs):
            # Try XConnector first if enabled
            if getattr(worker, 'xconnector_enabled', False) and worker.xconnector:
                try:
                    # Use XConnector for cache retrieval
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
        """Create wrapped send_kv_caches method"""

        async def wrapped_send_kv_caches(*args, **kwargs):
            # Call original method first
            result = await worker._original_send_kv_caches(*args, **kwargs)

            # Then send to XConnector if enabled
            if getattr(worker, 'xconnector_enabled', False) and worker.xconnector:
                try:
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


# Convenience function for easy integration
def load_extensions_for_worker(worker_class: type) -> type:
    """
    Decorator to automatically load extensions for a worker class

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


# Simple hook for Dynamo config processing
def process_worker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process worker configuration to load extensions

    This can be called from Dynamo's config loading pipeline
    """
    ExtensionLoader.load_extensions(config)
    return config