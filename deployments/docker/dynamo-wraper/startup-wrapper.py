#!/usr/bin/env python3
"""
Startup wrapper for AI-Dynamo with XConnector extension
This script wraps the original dynamo startup to inject XConnector
"""

import sys
import os
import time
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_xconnector_service():
    """Wait for XConnector service to be ready"""
    import requests

    service_url = os.getenv("XCONNECTOR_SERVICE_URL", "http://xconnector-service:8081")
    max_retries = 30
    retry_delay = 2

    for i in range(max_retries):
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("XConnector service is ready")
                return True
        except Exception as e:
            logger.info(f"Waiting for XConnector service... ({i + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)

    logger.warning("XConnector service not ready, continuing anyway")
    return False


def inject_xconnector():
    """Inject XConnector extension loader into Python path and patch Worker"""

    # Wait for XConnector service
    if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
        logger.info("XConnector enabled, waiting for service...")
        wait_for_xconnector_service()

    # Add extension loader to Python path
    extension_loader_path = "/xconnector-integration/extension_loader.py"

    if os.path.exists(extension_loader_path):
        try:
            # Load the extension loader module
            spec = importlib.util.spec_from_file_location("xconnector_extension", extension_loader_path)
            extension_module = importlib.util.module_from_spec(spec)
            sys.modules["xconnector_extension"] = extension_module
            spec.loader.exec_module(extension_module)

            logger.info("[XConnector] Extension loader imported successfully")

            # Monkey patch the VllmWorker
            patch_vllm_worker()

        except Exception as e:
            logger.error(f"[XConnector] Failed to load extension: {e}")
            if os.getenv("XCONNECTOR_FAIL_ON_ERROR", "false").lower() == "true":
                raise
    else:
        logger.warning("[XConnector] Extension loader not found, running without XConnector")


def patch_vllm_worker():
    """Monkey patch VllmWorker to load XConnector extension"""
    try:
        # Wait for the Dynamo modules to be available
        max_attempts = 10
        worker_class = None

        for attempt in range(max_attempts):
            try:
                # Try different possible import paths for VllmWorker
                import_paths = [
                    "components.worker",
                    "vllm_v0.components.worker",
                    "examples.vllm_v0.components.worker",
                    "worker"
                ]

                for import_path in import_paths:
                    try:
                        module = importlib.import_module(import_path)
                        if hasattr(module, 'VllmWorker'):
                            worker_class = getattr(module, 'VllmWorker')
                            logger.info(f"[XConnector] Found VllmWorker in {import_path}")
                            break
                    except ImportError:
                        continue

                if worker_class:
                    break

                time.sleep(1)

            except ImportError:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.warning("[XConnector] Failed to import VllmWorker after multiple attempts")
                    return

        if not worker_class:
            logger.warning("[XConnector] VllmWorker not found in any expected location")
            return

        # Store original init
        original_init = worker_class.__init__

        # Create wrapped init
        def wrapped_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)

            # Load XConnector extension only if enabled
            if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
                logger.info("[XConnector] Loading extension for VllmWorker")
                try:
                    from xconnector_extension import ExtensionLoader

                    # Get config from self
                    config = getattr(self, 'config', {})
                    if hasattr(config, '__dict__'):
                        config = config.__dict__

                    # Add XConnector service configuration
                    if 'extensions' not in config:
                        config['extensions'] = {}

                    if 'xconnector' not in config['extensions']:
                        config['extensions']['xconnector'] = {}

                    # Configure remote mode
                    config['extensions']['xconnector'].update({
                        'enabled': True,
                        'service_mode': 'remote',
                        'service_url': os.getenv("XCONNECTOR_SERVICE_URL", "http://xconnector-service:8081"),
                        'fail_on_error': os.getenv("XCONNECTOR_FAIL_ON_ERROR", "false").lower() == "true"
                    })

                    ExtensionLoader.load_extensions(config)
                    ExtensionLoader.inject_into_worker(self)
                    logger.info("[XConnector] Extension loaded successfully")

                except Exception as e:
                    logger.error(f"[XConnector] Failed to load extension: {e}")
                    if os.getenv("XCONNECTOR_FAIL_ON_ERROR", "false").lower() == "true":
                        raise
            else:
                logger.info("[XConnector] XConnector disabled, skipping extension loading")

        # Replace init
        worker_class.__init__ = wrapped_init
        logger.info("[XConnector] VllmWorker patched successfully")

    except Exception as e:
        logger.error(f"[XConnector] Failed to patch VllmWorker: {e}")
        if os.getenv("XCONNECTOR_FAIL_ON_ERROR", "false").lower() == "true":
            raise


# Inject XConnector before starting Dynamo
logger.info("[XConnector] Starting injection process...")
inject_xconnector()
logger.info("[XConnector] Injection process completed")

# Continue with normal Dynamo execution
if __name__ == "__main__":
    logger.info("Starting Dynamo CLI...")

    # Try to import and run dynamo CLI
    try:
        # Try different possible import paths for dynamo CLI
        dynamo_main = None
        import_paths = [
            "dynamo.cli",
            "ai_dynamo.cli",
            "cli"
        ]

        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                if hasattr(module, 'main'):
                    dynamo_main = getattr(module, 'main')
                    logger.info(f"Found dynamo main in {import_path}")
                    break
            except ImportError:
                continue

        if dynamo_main:
            dynamo_main()
        else:
            logger.error("Could not find dynamo CLI main function")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to start Dynamo: {e}")
        sys.exit(1)