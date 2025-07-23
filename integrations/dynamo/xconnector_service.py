# integrations/dynamo/xconnector_service.py
"""
XConnector Service for AI-Dynamo

Fixed version with proper FastAPI integration and standalone service capability.
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from xconnector.core.connector import XConnector, AdapterConfig, AdapterType
    from xconnector.utils.config import ConnectorConfig
    from xconnector.utils.xconnector_logging import get_logger
except ImportError as e:
    logger.warning(f"XConnector import failed: {e}, running in mock mode")


    # Mock classes for testing without XConnector
    class XConnector:
        def __init__(self, *args, **kwargs): pass

        async def start(self): pass

        async def stop(self): pass

        async def get_health_status(self): return {"status": "mock"}

        def list_adapters(self): return {"inference": [], "cache": [], "distributed": []}


    class AdapterConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


    class AdapterType:
        INFERENCE = "inference"
        CACHE = "cache"
        DISTRIBUTED = "distributed"


    class ConnectorConfig:
        def __init__(self): pass


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

class XConnectorService:
    """
    Standalone XConnector service for AI-Dynamo integration

    Provides centralized adapter management and routing coordination
    without requiring Dynamo SDK dependencies.
    """

    def __init__(self):
        # Load configuration from environment or config file
        self.config = self._load_config()

        # Initialize XConnector
        try:
            self.connector = XConnector(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize XConnector: {e}")
            self.connector = None

        # Service metadata
        self.service_id = f"xconnector-service-{os.getpid()}"
        self.start_time = datetime.now()

        # Adapter registry
        self.adapter_registry = {}

        # Create FastAPI app
        self.app = FastAPI(title="XConnector Service", version="1.0.0")
        self._setup_routes()

        logger.info("XConnectorService initialized")

    def _load_config(self) -> ConnectorConfig:
        """Load configuration from file or environment"""
        try:
            config_file = os.getenv("XCONNECTOR_CONFIG", "/app/configs/xconnector_config.yaml")

            if os.path.exists(config_file):
                logger.info(f"Loading config from: {config_file}")
                # Load YAML config if available
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    return self._dict_to_config(config_data)
                except Exception as e:
                    logger.warning(f"Failed to load config file: {e}")

            # Use default config
            return ConnectorConfig()

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return ConnectorConfig()

    def _dict_to_config(self, config_data: Dict[str, Any]) -> ConnectorConfig:
        """Convert dictionary to ConnectorConfig"""
        config = ConnectorConfig()

        # Load adapter configurations
        if 'adapters' in config_data:
            adapters = []
            for name, adapter_data in config_data['adapters'].items():
                adapter_config = AdapterConfig(
                    name=name,
                    type=adapter_data.get('type', 'inference'),
                    class_path=adapter_data.get('class_path', ''),
                    config=adapter_data.get('config', {}),
                    enabled=adapter_data.get('enabled', True)
                )
                adapters.append(adapter_config)
            config.adapters = adapters

        return config

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return await self.get_status()

        @self.app.get("/status")
        async def get_status():
            """Get service status"""
            return await self.get_status()

        @self.app.get("/adapters")
        async def list_adapters():
            """List all adapters"""
            return await self.list_adapters()

        @self.app.post("/adapters")
        async def register_adapter(request: AdapterRegistrationRequest):
            """Register new adapter"""
            return await self.register_adapter(request)

        @self.app.delete("/adapters/{adapter_name}")
        async def unregister_adapter(adapter_name: str, adapter_type: str):
            """Unregister adapter"""
            return await self.unregister_adapter(adapter_name, adapter_type)

        @self.app.post("/route")
        async def route_request(request: RouteRequest):
            """Route request through XConnector"""
            return await self.route_request(request)

        @self.app.get("/adapters/{adapter_name}")
        async def get_adapter_info(adapter_name: str):
            """Get adapter information"""
            return await self.get_adapter_info(adapter_name)

    # === Service Methods ===

    async def start(self):
        """Start XConnector service"""
        try:
            if self.connector:
                # Load pre-configured adapters
                await self._load_default_adapters()

                # Start XConnector
                await self.connector.start()

            logger.info(f"XConnectorService started: {self.service_id}")

        except Exception as e:
            logger.error(f"Failed to start XConnectorService: {e}")
            raise

    async def stop(self):
        """Stop XConnector service"""
        try:
            if self.connector:
                await self.connector.stop()

            logger.info(f"XConnectorService stopped: {self.service_id}")

        except Exception as e:
            logger.error(f"Error stopping XConnectorService: {e}")

    async def _load_default_adapters(self):
        """Load default adapters from configuration"""
        if not self.connector:
            return

        for adapter_config in self.config.adapters:
            if adapter_config.enabled:
                try:
                    # Load adapter
                    await self.connector.load_adapter(adapter_config)

                    # Store in registry
                    self.adapter_registry[adapter_config.name] = {
                        "type": adapter_config.type,
                        "config": adapter_config,
                        "loaded_at": datetime.now().isoformat()
                    }

                    logger.info(f"Loaded adapter: {adapter_config.name}")

                except Exception as e:
                    logger.error(f"Failed to load adapter {adapter_config.name}: {e}")

    # === API Endpoints ===

    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            status = {
                "service": {
                    "id": self.service_id,
                    "status": "healthy",
                    "start_time": self.start_time.isoformat(),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
                },
                "xconnector": {
                    "available": self.connector is not None,
                    "status": "unknown"
                }
            }

            if self.connector:
                try:
                    health_status = await self.connector.get_health_status()
                    status["xconnector"]["status"] = health_status.get("connector", {}).get("status", "unknown")
                    status["xconnector"]["details"] = health_status
                except Exception as e:
                    status["xconnector"]["status"] = "error"
                    status["xconnector"]["error"] = str(e)

            return status

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                "service": {"status": "error", "error": str(e)},
                "xconnector": {"available": False, "status": "error"}
            }

    async def list_adapters(self) -> Dict[str, Any]:
        """List all loaded adapters"""
        try:
            if not self.connector:
                return {"adapters": {}, "registry": self.adapter_registry}

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

            return {
                "adapters": detailed_info,
                "registry": self.adapter_registry
            }

        except Exception as e:
            logger.error(f"List adapters failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def register_adapter(self, request: AdapterRegistrationRequest) -> Dict[str, Any]:
        """Register a new adapter"""
        try:
            if not self.connector:
                raise HTTPException(status_code=503, detail="XConnector not available")

            # Create adapter configuration
            adapter_config = AdapterConfig(
                name=request.name,
                type=request.type,
                class_path=request.class_path,
                config=request.config,
                enabled=request.enabled
            )

            # Load adapter
            await self.connector.load_adapter(adapter_config)

            # Update registry
            self.adapter_registry[request.name] = {
                "type": request.type,
                "config": adapter_config,
                "loaded_at": datetime.now().isoformat()
            }

            return {
                "status": "success",
                "adapter": request.name,
                "message": f"Adapter {request.name} registered successfully"
            }

        except Exception as e:
            logger.error(f"Failed to register adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def unregister_adapter(self, adapter_name: str, adapter_type: str) -> Dict[str, Any]:
        """Unregister an adapter"""
        try:
            if not self.connector:
                raise HTTPException(status_code=503, detail="XConnector not available")

            # Convert string to AdapterType
            type_mapping = {
                "inference": AdapterType.INFERENCE,
                "cache": AdapterType.CACHE,
                "distributed": AdapterType.DISTRIBUTED
            }

            adapter_type_enum = type_mapping.get(adapter_type)
            if not adapter_type_enum:
                raise HTTPException(status_code=400, detail=f"Invalid adapter type: {adapter_type}")

            # Unload adapter
            adapter = self.connector.unload_adapter(adapter_name, adapter_type_enum)

            if adapter:
                # Remove from registry
                self.adapter_registry.pop(adapter_name, None)

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

    async def route_request(self, request: RouteRequest) -> Dict[str, Any]:
        """Route a request through XConnector"""
        try:
            if not self.connector:
                raise HTTPException(status_code=503, detail="XConnector not available")

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

    async def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific adapter"""
        try:
            if not self.connector:
                raise HTTPException(status_code=503, detail="XConnector not available")

            # Check all adapter types
            for adapter_type in [AdapterType.INFERENCE, AdapterType.CACHE, AdapterType.DISTRIBUTED]:
                adapter = self.connector.get_adapter(adapter_name, adapter_type)
                if adapter:
                    info = adapter.get_info() if hasattr(adapter, 'get_info') else {"name": adapter_name}
                    info["registry_info"] = self.adapter_registry.get(adapter_name, {})
                    return info

            raise HTTPException(status_code=404, detail=f"Adapter {adapter_name} not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get adapter info failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# === Standalone Service Runner ===

async def create_service():
    """Create and start XConnector service"""
    service = XConnectorService()
    await service.start()
    return service


# Global service instance
_service_instance = None


async def get_service():
    """Get service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = await create_service()
    return _service_instance


# Create FastAPI app for standalone use
app = FastAPI(title="XConnector Service", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting XConnector service...")
    await get_service()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Stopping XConnector service...")
    global _service_instance
    if _service_instance:
        await _service_instance.stop()


# Mount service routes
@app.get("/health")
async def health_check():
    service = await get_service()
    return await service.get_status()


@app.get("/status")
async def get_status():
    service = await get_service()
    return await service.get_status()


@app.get("/adapters")
async def list_adapters():
    service = await get_service()
    return await service.list_adapters()


@app.post("/adapters")
async def register_adapter(request: AdapterRegistrationRequest):
    service = await get_service()
    return await service.register_adapter(request)


@app.delete("/adapters/{adapter_name}")
async def unregister_adapter(adapter_name: str, adapter_type: str):
    service = await get_service()
    return await service.unregister_adapter(adapter_name, adapter_type)


@app.post("/route")
async def route_request(request: RouteRequest):
    service = await get_service()
    return await service.route_request(request)


@app.get("/adapters/{adapter_name}")
async def get_adapter_info(adapter_name: str):
    service = await get_service()
    return await service.get_adapter_info(adapter_name)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("XCONNECTOR_PORT", "8081"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "integrations.dynamo.xconnector_service:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=False
    )