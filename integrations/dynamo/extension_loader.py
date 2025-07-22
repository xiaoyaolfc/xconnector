# integrations/dynamo/extension.py
"""
XConnector Extension for AI-Dynamo

This extension provides a plugin mechanism to integrate XConnector
without modifying Dynamo source code.
"""
import httpx
import asyncio
from typing import Any, Dict, Optional


class RemoteXConnectorExtension:
    """Remote XConnector extension that communicates with XConnector service"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.service_url = config.get("service_url")
        self.client = None

    def load(self) -> bool:
        """Initialize remote connection"""
        if not self.enabled:
            logger.info("Remote XConnector extension is disabled")
            return False

        if not self.service_url:
            logger.error("Remote mode requires service_url")
            return False

        try:
            # Create HTTP client
            self.client = httpx.AsyncClient(base_url=self.service_url)

            # Test connection
            response = httpx.get(f"{self.service_url}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Connected to XConnector service at {self.service_url}")
                return True
            else:
                logger.error(f"XConnector service unhealthy: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to XConnector service: {e}")
            if self.config.get("fail_on_error", False):
                raise
            return False

    async def route_message(self, source: str, target: str, method: str, **kwargs) -> Any:
        """Route message through remote XConnector service"""
        if not self.client:
            raise RuntimeError("Remote XConnector not initialized")

        try:
            response = await self.client.post(
                "/route_request",
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


# 修改 ExtensionLoader._load_xconnector 方法
def _load_xconnector(self, config: Dict[str, Any]):
    """Load XConnector extension"""
    service_mode = config.get("service_mode", "embedded")

    try:
        if service_mode == "remote":
            # 使用远程 XConnector 服务
            extension = RemoteXConnectorExtension(config)
        else:
            # 嵌入式模式
            extension = XConnectorExtension(config)

        if extension.load():
            self._extensions["xconnector"] = extension
            logger.info(f"XConnector extension registered in {service_mode} mode")
    except Exception as e:
        logger.error(f"Failed to load XConnector extension: {e}")