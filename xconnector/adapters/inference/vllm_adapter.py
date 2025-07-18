# xconnector/vllm_adapter.py
import asyncio
from typing import Any, Tuple, Union, List, Optional
import torch
from xconnector.vllm.xconnector.adapter import BaseAdapter
from xconnector.vllm.xconnector.interfaces import VLLMInterface

class VLLMAdapter(BaseAdapter, VLLMInterface):
    def __init__(self, core):
        super().__init__(core)
        self.kv_cache_store = {}  # 临时存储KV缓存
        self.finished_requests = set()
        
    def register_endpoints(self):
        self.core.register_vllm(self)
        
    async def recv_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: Any,
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """接收KV缓存，与LMCache交互"""
        try:
            # 1. 检查是否需要从LMCache检索
            should_retrieve = await self.call('lmcache/should_retrieve', model_input)
            
            if should_retrieve:
                # 2. 从LMCache获取KV缓存
                cached_kv = await self.call('lmcache/retrieve_kv', model_input)
                if cached_kv:
                    # 3. 合并缓存的KV和新的KV
                    merged_kv = self._merge_kv_caches(cached_kv, kv_caches)
                    
                    # 4. 执行模型推理
                    hidden_states = await self._execute_model(
                        model_executable, model_input, merged_kv
                    )
                    
                    return hidden_states, False, model_input
            
            # 5. 如果没有缓存，正常执行
            hidden_states = await self._execute_model(
                model_executable, model_input, kv_caches
            )
            
            return hidden_states, False, model_input
            
        except Exception as e:
            # 错误处理
            print(f"Error in recv_kv_caches: {e}")
            return None, True, model_input
    
    async def send_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: Any,
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, Any]
    ) -> None:
        """发送KV缓存到LMCache"""
        try:
            # 1. 检查是否需要存储到LMCache
            should_store = await self.call('lmcache/should_store', model_input)
            
            if should_store:
                # 2. 存储KV缓存到LMCache
                await self.call(
                    'lmcache/store_kv',
                    model_input,
                    kv_caches,
                    hidden_or_intermediate_states
                )
                
            # 3. 本地临时存储（可选）
            request_id = getattr(model_input, 'request_id', None)
            if request_id:
                self.kv_cache_store[request_id] = {
                    'kv_caches': kv_caches,
                    'hidden_states': hidden_or_intermediate_states
                }
                
        except Exception as e:
            print(f"Error in send_kv_caches: {e}")
    
    async def get_finished(
        self,
        finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        """获取已完成的请求ID"""
        # 更新本地完成状态
        self.finished_requests.update(finished_req_ids)
        
        # 清理已完成请求的缓存
        for req_id in finished_req_ids:
            if req_id in self.kv_cache_store:
                del self.kv_cache_store[req_id]
        
        # 通知LMCache清理相关缓存
        await self.call('lmcache/cleanup_finished', finished_req_ids)
        
        return finished_req_ids, None
    
    def _merge_kv_caches(self, cached_kv: List[torch.Tensor], 
                        new_kv: List[torch.Tensor]) -> List[torch.Tensor]:
        """合并缓存的KV和新的KV"""
        merged = []
        for i, (cached, new) in enumerate(zip(cached_kv, new_kv)):
            if cached is not None and new is not None:
                # 沿着序列长度维度拼接
                merged.append(torch.cat([cached, new], dim=-2))
            elif cached is not None:
                merged.append(cached)
            elif new is not None:
                merged.append(new)
            else:
                merged.append(None)
        return merged
    
    async def _execute_model(self, model_executable: torch.nn.Module,
                           model_input: Any, kv_caches: List[torch.Tensor]) -> Any:
        """执行模型推理"""
        # 这里应该调用实际的模型执行逻辑
        # 暂时返回模拟结果
        return torch.randn(1, 512, 4096)  # 模拟hidden states