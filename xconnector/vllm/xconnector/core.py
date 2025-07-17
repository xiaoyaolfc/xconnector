# xconnector/core.py
import asyncio
import inspect
from typing import Any, Coroutine, Dict, Tuple
from .interfaces import VLLMInterface, LMCacheInterface

class XConnectorCore:
    def __init__(self):
        self.vllm_interface: VLLMInterface = None
        self.lmcache_interface: LMCacheInterface = None
        self.connection_table: Dict[str, asyncio.Queue] = {}
        self.task_table: Dict[str, asyncio.Task] = {}
        
    def register_vllm(self, adapter: VLLMInterface):
        self.vllm_interface = adapter
        
    def register_lmcache(self, adapter: LMCacheInterface):
        self.lmcache_interface = adapter
        
    async def route(self, endpoint: str, *args, **kwargs) -> Any:
        """路由消息到目标端点"""
        if endpoint.startswith('vllm/'):
            handler = getattr(self.vllm_interface, endpoint[5:], None)
        elif endpoint.startswith('lmcache/'):
            handler = getattr(self.lmcache_interface, endpoint[7:], None)
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}")
            
        if not handler:
            raise AttributeError(f"Handler not found for {endpoint}")
            
        if inspect.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        else:
            return handler(*args, **kwargs)
            
    def create_endpoint(self, endpoint: str, queue_size: int = 100):
        """创建端点的消息队列"""
        self.connection_table[endpoint] = asyncio.Queue(queue_size)
        
    async def send(self, endpoint: str, *args, **kwargs):
        """发送消息到指定端点"""
        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        await self.connection_table[endpoint].put((args, kwargs))
        
    async def receive(self, endpoint: str) -> Tuple[Tuple, Dict]:
        """从端点接收消息"""
        if endpoint not in self.connection_table:
            self.create_endpoint(endpoint)
        return await self.connection_table[endpoint].get()
        
    def start_task(self, name: str, coro: Coroutine):
        """启动后台任务"""
        self.task_table[name] = asyncio.create_task(coro)
        
    def stop_task(self, name: str):
        """停止后台任务"""
        if name in self.task_table:
            self.task_table[name].cancel()