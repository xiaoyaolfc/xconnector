# xconnector/vllm_adapter.py
import asyncio
from typing import Any, Tuple, Union,List, Optional
import torch
from .adapter import BaseAdapter
from .interfaces import VLLMInterface

class VLLMAdapter(BaseAdapter, VLLMInterface):
    def register_endpoints(self):
        self.core.register_vllm(self)
        
    async def recv_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: Any,
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        # 实际实现略，示例：调用LMCache
        result = await self.call(
            'lmcache/recv_kv_caches', 
            model_executable, 
            model_input, 
            kv_caches
        )
        return result
        
    async def send_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: Any,
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, Any]
    ) -> None:
        await self.call(
            'lmcache/send_kv_caches',
            model_executable,
            model_input,
            kv_caches,
            hidden_or_intermediate_states
        )
        
    async def get_finished(
        self,
        finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        return await self.call('lmcache/get_finished', finished_req_ids)