# xconnector/interfaces.py
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


class VLLMInterface:
    async def recv_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, "IntermediateTensors"], bool, Any]:
        raise NotImplementedError

    async def send_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, "IntermediateTensors"]
    ) -> None:
        raise NotImplementedError

    async def get_finished(
            self,
            finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        raise NotImplementedError


class LMCacheInterface:
    async def start_load_kv(self, context: "ForwardContext", **kwargs) -> None:
        raise NotImplementedError

    async def wait_for_layer_load(self, layer_name: str) -> None:
        raise NotImplementedError

    async def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        raise NotImplementedError

    async def wait_for_save(self) -> None:
        raise NotImplementedError

    async def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        raise NotImplementedError

    async def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        raise NotImplementedError