#!/usr/bin/env python3
"""
Startup wrapper for AI-Dynamo with XConnector extension
This script wraps the original dynamo startup to inject XConnector
"""

import sys
import os
import importlib.util


def inject_xconnector():
    """Inject XConnector extension loader into Python path and patch Worker"""

    # Add extension loader to Python path
    extension_loader_path = "/xconnector-integration/extension_loader.py"

    if os.path.exists(extension_loader_path):
        # Load the extension loader module
        spec = importlib.util.spec_from_file_location("xconnector_extension", extension_loader_path)
        extension_module = importlib.util.module_from_spec(spec)
        sys.modules["xconnector_extension"] = extension_module
        spec.loader.exec_module(extension_module)

        # Monkey patch the VllmWorker
        print("[XConnector] Patching VllmWorker...")
        patch_vllm_worker()
    else:
        print("[XConnector] Extension loader not found, running without XConnector")


def patch_vllm_worker():
    """Monkey patch VllmWorker to load XConnector extension"""
    try:
        # Wait for the module to be available
        import time
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                # Import the worker module
                from components.worker import VllmWorker

                # Store original init
                original_init = VllmWorker.__init__

                # Create wrapped init
                def wrapped_init(self, *args, **kwargs):
                    # Call original init
                    original_init(self, *args, **kwargs)

                    # Load XConnector extension
                    print("[XConnector] Loading extension for VllmWorker")
                    from xconnector_extension import ExtensionLoader

                    # Get config from self
                    config = getattr(self, 'config', {})
                    if isinstance(config, object) and hasattr(config, '__dict__'):
                        config = config.__dict__

                    ExtensionLoader.load_extensions(config)
                    ExtensionLoader.inject_into_worker(self)
                    print("[XConnector] Extension loaded successfully")

                # Replace init
                VllmWorker.__init__ = wrapped_init
                print("[XConnector] VllmWorker patched successfully")
                break

            except ImportError:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                else:
                    print("[XConnector] Failed to import VllmWorker after multiple attempts")

    except Exception as e:
        print(f"[XConnector] Failed to patch VllmWorker: {e}")


# Inject XConnector before starting Dynamo
inject_xconnector()

# Continue with normal Dynamo execution
if __name__ == "__main__":
    # Import and run dynamo CLI
    from dynamo.cli import main

    main()