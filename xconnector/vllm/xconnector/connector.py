# xconnector/connector.py
from xconnector.vllm.xconnector.core import XConnectorCore
from xconnector.vllm.xconnector.vllm_adapter import VLLMAdapter
from xconnector.vllm.xconnector.lmcache_adapter import LMCacheAdapter

# 纯粹当一个中间件，引用vllm和lmcache中的adapter，通过xconnector管理
class XConnector:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.core = XConnectorCore()
            cls._instance.vllm_adapter = VLLMAdapter(cls._instance.core)
            cls._instance.lmcache_adapter = LMCacheAdapter(cls._instance.core)
        return cls._instance
    
    @property
    def vllm(self):
        return self._instance.vllm_adapter
    
    @property
    def lmcache(self):
        return self._instance.lmcache_adapter