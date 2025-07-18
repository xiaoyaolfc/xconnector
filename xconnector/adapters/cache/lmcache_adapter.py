# xconnector/lmcache_adapter.py
import asyncio
from typing import Any, Optional, Tuple
import torch
from xconnector.vllm.xconnector.adapter import BaseAdapter
from xconnector.vllm.xconnector.interfaces import LMCacheInterface
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

class LMCacheAdapter(BaseAdapter, LMCacheInterface):
    def register_endpoints(self):
        self.core.register_lmcache(self)
        
    async def start_load_kv(self, context: ForwardContext, **kwargs) -> None:
        await self.call('vllm/start_load_kv', context, **kwargs)
        
    async def wait_for_layer_load(self, layer_name: str) -> None:
        await self.call('vllm/wait_for_layer_load', layer_name)
        
    async def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        await self.call(
            'vllm/save_kv_layer',
            layer_name,
            kv_layer,
            attn_metadata,
            **kwargs
        )
        
    async def wait_for_save(self) -> None:
        await self.call('vllm/wait_for_save')
        
    async def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        return await self.call(
            'vllm/get_num_new_matched_tokens',
            request,
            num_computed_tokens
        )
        
    async def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        await self.call(
            'vllm/update_state_after_alloc',
            request,
            blocks,
            num_external_tokens
        )