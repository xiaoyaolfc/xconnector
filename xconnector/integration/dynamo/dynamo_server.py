# xconnector/integration/dynamo/dynamo_server.py
from fastapi import FastAPI
from dynamo.sdk import service, endpoint


@service(
    dynamo={"namespace": "xconnector"},
    workers=1
)
class XConnectorServer:
    """XConnector management server for Dynamo"""

    def __init__(self):
        self.connector = XConnector()
        self.app = FastAPI(title="XConnector Server")

    @endpoint()
    async def get_status(self):
        """Get XConnector status"""
        return await self.connector.get_health_status()

    @endpoint()
    async def register_adapter(self, adapter_config: Dict[str, Any]):
        """Register new adapter"""
        config = AdapterConfig(**adapter_config)
        await self.connector.load_adapter(config)
        return {"status": "registered", "adapter": config.name}