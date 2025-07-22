# xconnector/integration/dynamo/xconnector_service.py
"""
XConnector Service for AI-Dynamo

Provides a centralized XConnector service that can be deployed
as a Dynamo component for managing adapters and routing.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo.sdk import service, endpoint, depends, async_on_start
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.runtime import EtcdKvCache

from xconnector.core.connector import XConnector, AdapterConfig, AdapterType
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


# === Request/Response Models ===

class AdapterRegistrationRequest(BaseModel):
    """Adapter registration request"""
    name: str
    type: str  # inference, cache, distributed
    class_path: str
    config: Dict[str, Any]
    enabled: bool = True


class RouteRequest(BaseModel):
    """Route request model"""
    source: str
    target: str
    method: str
    params: Dict[str, Any] = {}


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    details: Dict[str, Any]


# === XConnector Service ===

@service(
    dynamo={
        "namespace": "xconnector",
        "replicas": 1,
    },
    resources={"cpu": "2", "memory": "4Gi"},
    workers=1,
    app=FastAPI(title="XConnector Service", version="1.0.0")
)
class XConnectorService:
    """
    XConnector management service for Dynamo

    Provides centralized adapter management and routing coordination
    for distributed inference workloads.
    """

    def __init__(self):
        # Load configuration
        self.service_config = ServiceConfig.get_parsed_config("XConnectorService")

        # Initialize XConnector
        self.connector = XConnector(self.service_config)

        # Service metadata
        self.namespace = None
        self.component_name = None
        self.config_cache = None

        # Adapter registry
        self.adapter_registry = {}

        logger.info("XConnectorService initialized")

    @async_on_start
    async def initialize(self):
        """Initialize service using Dynamo lifecycle hook"""
        try:
            # Get Dynamo context
            from dynamo.sdk import dynamo_context

            runtime = dynamo_context["runtime"]
            self.namespace, self.component_name = self.__class__.dynamo_address()

            # Initialize configuration cache
            self.config_cache = await EtcdKvCache.create(
                runtime.etcd_client(),
                f"/{self.namespace}/config/",
                self._get_default_config()
            )

            # Load pre-configured adapters
            await self._load_default_adapters()

            # Start XConnector
            await self.connector.start()

            # Start configuration watcher
            asyncio.create_task(self._watch_config_changes())

            # Register service in etcd
            await self._register_service(runtime)

            logger.info(f"XConnectorService started at {self.namespace}.{self.component_name}")

        except Exception as e:
            logger.error(f"Failed to initialize XConnectorService: {e}")
            raise

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "adapters": {
                "vllm": {
                    "enabled": True,
                    "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
                    "config": {
                        "model_name": "",
                        "tensor_parallel_size": 1,
                        "enable_prefix_caching": True
                    }
                },
                "lmcache": {
                    "enabled": True,
                    "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                    "config": {
                        "storage_backend": "local",
                        "max_cache_size": 1024,
                        "enable_compression": True
                    }
                },
                "dynamo": {
                    "enabled": True,
                    "class_path": "xconnector.adapters.distributed.dynamo_adapter.DynamoAdapter",
                    "config": {
                        "namespace": "dynamo",
                        "routing_policy": {
                            "strategy": "least_loaded"
                        }
                    }
                }
            },
            "routing": {
                "inference_to_cache": {
                    "enabled": True,
                    "timeout": 30.0,
                    "retry_count": 2
                },
                "cache_to_inference": {
                    "enabled": True,
                    "timeout": 30.0
                }
            },
            "monitoring": {
                "metrics_enabled": True,
                "health_check_interval": 30
            }
        }

    async def _load_default_adapters(self):
        """Load default adapters from configuration"""
        config = await self.config_cache.get("adapters")

        for adapter_name, adapter_config in config.items():
            if adapter_config.get("enabled", True):
                try:
                    # Determine adapter type
                    adapter_type = self._infer_adapter_type(adapter_name, adapter_config["class_path"])

                    # Create adapter configuration
                    config = AdapterConfig(
                        name=adapter_name,
                        type=adapter_type,
                        class_path=adapter_config["class_path"],
                        config=adapter_config.get("config", {})
                    )

                    # Load adapter
                    await self.connector.load_adapter(config)

                    # Store in registry
                    self.adapter_registry[adapter_name] = {
                        "type": adapter_type.value,
                        "config": config,
                        "loaded_at": asyncio.get_event_loop().time()
                    }

                    logger.info(f"Loaded adapter: {adapter_name}")

                except Exception as e:
                    logger.error(f"Failed to load adapter {adapter_name}: {e}")

    def _infer_adapter_type(self, name: str, class_path: str) -> AdapterType:
        """Infer adapter type from name and class path"""
        name_lower = name.lower()
        path_lower = class_path.lower()

        if "inference" in path_lower or any(x in name_lower for x in ["vllm", "tgi", "llm"]):
            return AdapterType.INFERENCE
        elif "cache" in path_lower or any(x in name_lower for x in ["cache", "redis"]):
            return AdapterType.CACHE
        elif "distributed" in path_lower or any(x in name_lower for x in ["dynamo", "distributed"]):
            return AdapterType.DISTRIBUTED
        else:
            return AdapterType.INFERENCE  # Default

    async def _watch_config_changes(self):
        """Watch for configuration changes in etcd"""
        try:
            async for key, value in self.config_cache.watch_iter():
                logger.info(f"Configuration changed: {key}")

                # Handle adapter configuration changes
                if key.startswith("adapters/"):
                    adapter_name = key.split("/")[1]
                    await self._handle_adapter_config_change(adapter_name, value)

        except Exception as e:
            logger.error(f"Config watcher error: {e}")

    async def _handle_adapter_config_change(self, adapter_name: str, config: Dict[str, Any]):
        """Handle adapter configuration changes"""
        try:
            # Check if adapter exists
            if adapter_name in self.adapter_registry:
                # Update existing adapter
                adapter_type = AdapterType(self.adapter_registry[adapter_name]["type"])
                adapter = self.connector.get_adapter(adapter_name, adapter_type)

                if adapter and hasattr(adapter, "update_config"):
                    adapter.update_config(config.get("config", {}))
                    logger.info(f"Updated configuration for adapter: {adapter_name}")
            else:
                # Load new adapter
                if config.get("enabled", True):
                    adapter_type = self._infer_adapter_type(adapter_name, config["class_path"])

                    adapter_config = AdapterConfig(
                        name=adapter_name,
                        type=adapter_type,
                        class_path=config["class_path"],
                        config=config.get("config", {})
                    )

                    await self.connector.load_adapter(adapter_config)

                    self.adapter_registry[adapter_name] = {
                        "type": adapter_type.value,
                        "config": adapter_config,
                        "loaded_at": asyncio.get_event_loop().time()
                    }

                    logger.info(f"Loaded new adapter: {adapter_name}")

        except Exception as e:
            logger.error(f"Failed to handle adapter config change: {e}")

    async def _register_service(self, runtime):
        """Register service in etcd for discovery"""
        try:
            service_info = {
                "namespace": self.namespace,
                "component": self.component_name,
                "endpoints": [
                    "get_status",
                    "list_adapters",
                    "register_adapter",
                    "unregister_adapter",
                    "route_request"
                ],
                "version": "1.0.0"
            }

            key = f"/{self.namespace}/services/xconnector"
            await runtime.etcd_client().put(key, json.dumps(service_info))

        except Exception as e:
            logger.error(f"Failed to register service: {e}")

    # === Endpoints ===

    @endpoint()
    async def get_status(self) -> HealthCheckResponse:
        """Get XConnector service status"""
        try:
            health_status = await self.connector.get_health_status()

            return HealthCheckResponse(
                status="healthy" if health_status["connector"]["status"] == "healthy" else "unhealthy",
                timestamp=asyncio.get_event_loop().time(),
                details=health_status
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status="error",
                timestamp=asyncio.get_event_loop().time(),
                details={"error": str(e)}
            )

    @endpoint()
    async def list_adapters(self) -> Dict[str, Any]:
        """List all loaded adapters"""
        adapters = self.connector.list_adapters()

        # Add registry information
        detailed_info = {}
        for adapter_type, adapter_list in adapters.items():
            detailed_info[adapter_type] = []

            for adapter_name in adapter_list:
                info = {
                    "name": adapter_name,
                    "loaded": adapter_name in self.adapter_registry
                }

                if adapter_name in self.adapter_registry:
                    info.update(self.adapter_registry[adapter_name])

                detailed_info[adapter_type].append(info)

        return detailed_info

    @endpoint()
    async def register_adapter(self, request: AdapterRegistrationRequest) -> Dict[str, Any]:
        """Register a new adapter"""
        try:
            # Create adapter configuration
            adapter_config = AdapterConfig(
                name=request.name,
                type=AdapterType(request.type),
                class_path=request.class_path,
                config=request.config,
                enabled=request.enabled
            )

            # Load adapter
            adapter = await self.connector.load_adapter(adapter_config)

            # Update registry
            self.adapter_registry[request.name] = {
                "type": request.type,
                "config": adapter_config,
                "loaded_at": asyncio.get_event_loop().time()
            }

            # Persist to etcd
            await self.config_cache.set(
                f"adapters/{request.name}",
                {
                    "enabled": request.enabled,
                    "class_path": request.class_path,
                    "config": request.config
                }
            )

            return {
                "status": "success",
                "adapter": request.name,
                "message": f"Adapter {request.name} registered successfully"
            }

        except Exception as e:
            logger.error(f"Failed to register adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @endpoint()
    async def unregister_adapter(self, adapter_name: str, adapter_type: str) -> Dict[str, Any]:
        """Unregister an adapter"""
        try:
            # Unload adapter
            adapter = self.connector.unload_adapter(adapter_name, AdapterType(adapter_type))

            if adapter:
                # Remove from registry
                self.adapter_registry.pop(adapter_name, None)

                # Update etcd
                await self.config_cache.set(
                    f"adapters/{adapter_name}/enabled",
                    False
                )

                return {
                    "status": "success",
                    "message": f"Adapter {adapter_name} unregistered successfully"
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Adapter {adapter_name} not found"
                }

        except Exception as e:
            logger.error(f"Failed to unregister adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @endpoint()
    async def route_request(self, request: RouteRequest) -> Dict[str, Any]:
        """Route a request through XConnector"""
        try:
            result = await self.connector.route_message(
                source=request.source,
                target=request.target,
                method=request.method,
                **request.params
            )

            return {
                "status": "success",
                "result": result
            }

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    @endpoint()
    async def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific adapter"""
        # Check all adapter types
        for adapter_type in [AdapterType.INFERENCE, AdapterType.CACHE, AdapterType.DISTRIBUTED]:
            adapter = self.connector.get_adapter(adapter_name, adapter_type)
            if adapter:
                info = adapter.get_info()
                info["registry_info"] = self.adapter_registry.get(adapter_name, {})
                return info

        raise HTTPException(status_code=404, detail=f"Adapter {adapter_name} not found")

    @endpoint()
    async def update_routing_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Update routing policy for Dynamo adapter"""
        try:
            dynamo_adapter = self.connector.get_adapter("dynamo", AdapterType.DISTRIBUTED)

            if dynamo_adapter and hasattr(dynamo_adapter, "update_routing_policy"):
                success = dynamo_adapter.update_routing_policy(policy)

                if success:
                    # Persist to etcd
                    await self.config_cache.set(
                        "adapters/dynamo/config/routing_policy",
                        policy
                    )

                    return {
                        "status": "success",
                        "message": "Routing policy updated successfully"
                    }

            return {
                "status": "error",
                "message": "Dynamo adapter not found or does not support policy updates"
            }

        except Exception as e:
            logger.error(f"Failed to update routing policy: {e}")
            raise HTTPException(status_code=500, detail=str(e))