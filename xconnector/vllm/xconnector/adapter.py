# xconnector/adapter.py
from .core import XConnectorCore
from .interfaces import VLLMInterface, LMCacheInterface

class BaseAdapter:
    def __init__(self, core: XConnectorCore):
        self.core = core
        self.register_endpoints()
        
    def register_endpoints(self):
        """由子类实现端点注册"""
        raise NotImplementedError
        
    async def call(self, endpoint: str, *args, **kwargs):
        """调用远端端点"""
        return await self.core.route(endpoint, *args, **kwargs)