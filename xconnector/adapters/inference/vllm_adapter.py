# xconnector/adapters/inference/vllm_adapter.py
"""
VLLM Adapter for XConnector

This adapter provides integration between vLLM inference engine and XConnector,
enabling KV cache management and distributed inference capabilities.
支持SDK嵌入模式和独立服务模式。
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch

from xconnector.adapters.base_adapter import BaseAdapter
from xconnector.interfaces.base_interface import (
    AdapterStatus,
    HealthStatus,
    HealthCheckResult,
    Capability,
)
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class VLLMRequestStatus(Enum):
    """VLLM request processing status"""
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class VLLMRequest:
    """VLLM request metadata"""
    request_id: str
    tokens: List[int]
    status: VLLMRequestStatus
    seq_len: int = 0
    kv_cached: bool = False


class VLLMAdapter(BaseAdapter):
    """
    VLLM Adapter for XConnector

    Handles integration between vLLM inference engine and cache systems,
    providing KV cache management and optimization.
    支持SDK嵌入模式，直接集成到Dynamo Worker中。
    """

    __version__ = "1.0.0"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["torch", "vllm"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # 检测运行模式
        self.sdk_mode = core_instance is not None and hasattr(core_instance, 'sdk_mode') and core_instance.sdk_mode

        # VLLM specific configuration
        self.model_name = config.get("model_name", "")
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.pipeline_parallel_size = config.get("pipeline_parallel_size", 1)
        self.max_batch_size = config.get("max_batch_size", 256)

        # KV cache management
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")
        self.enable_prefix_caching = config.get("enable_prefix_caching", True)
        self.enable_chunked_prefill = config.get("enable_chunked_prefill", False)

        # Request tracking
        self.active_requests: Dict[str, VLLMRequest] = {}
        self.finished_requests: set = set()

        # Cache coordination
        self.cache_hit_rate = 0.0
        self.total_cache_queries = 0
        self.cache_hits = 0

        # SDK模式下的缓存处理器引用
        self._kv_handler = None

        logger.info(f"VLLMAdapter initialized with model: {self.model_name} (SDK mode: {self.sdk_mode})")

    # === Required BaseInterface Methods ===

    async def _initialize_impl(self) -> bool:
        """Initialize VLLM adapter components"""
        try:
            # Validate vLLM availability
            try:
                import vllm
                logger.info(f"vLLM version: {vllm.__version__}")
            except ImportError:
                logger.error("vLLM not installed")
                return False

            # SDK模式下获取KV处理器引用
            if self.sdk_mode and self.core:
                try:
                    self._kv_handler = self.core.get_kv_handler() if hasattr(self.core, 'get_kv_handler') else None
                    if self._kv_handler:
                        logger.info("Connected to SDK KV handler")
                    else:
                        logger.warning("SDK KV handler not available")
                except Exception as e:
                    logger.warning(f"Failed to get SDK KV handler: {e}")

            # Initialize metrics
            self._reset_metrics()

            return True
        except Exception as e:
            self.log_error(e, {"operation": "initialize"})
            return False

    async def _start_impl(self) -> bool:
        """Start VLLM adapter services"""
        try:
            # Register with core if available
            if self.core and not self.sdk_mode:
                await self._register_routes()

            logger.info("VLLMAdapter started successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "start"})
            return False

    async def _stop_impl(self) -> bool:
        """Stop VLLM adapter services"""
        try:
            # Clear active requests
            self.active_requests.clear()
            self.finished_requests.clear()

            logger.info("VLLMAdapter stopped successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "stop"})
            return False

    def get_capabilities(self) -> Dict[str, Capability]:
        """Return VLLM adapter capabilities"""
        return {
            "kv_cache_management": Capability(
                name="kv_cache_management",
                description="KV cache receive and send operations",
                version="1.0.0",
                supported=True,
                parameters={
                    "max_batch_size": self.max_batch_size,
                    "tensor_parallel": self.tensor_parallel_size,
                    "prefix_caching": self.enable_prefix_caching,
                    "sdk_mode": self.sdk_mode
                }
            ),
            "chunked_prefill": Capability(
                name="chunked_prefill",
                description="Support for chunked prefill operations",
                version="1.0.0",
                supported=self.enable_chunked_prefill,
                parameters={}
            ),
            "distributed_inference": Capability(
                name="distributed_inference",
                description="Support for distributed inference",
                version="1.0.0",
                supported=self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1,
                parameters={
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "pipeline_parallel_size": self.pipeline_parallel_size
                }
            )
        }

    async def _health_check_impl(self) -> Optional[HealthCheckResult]:
        """VLLM specific health check"""
        try:
            # Check if we can perform basic tensor operations
            test_tensor = torch.randn(1, 10)
            _ = test_tensor.sum()

            # Calculate cache efficiency
            cache_efficiency = (self.cache_hits / max(self.total_cache_queries, 1)) * 100

            # SDK模式下检查KV处理器连接
            kv_handler_status = "connected" if self._kv_handler else "not_available"

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="VLLM adapter is healthy",
                timestamp=self.last_health_check,
                details={
                    "active_requests": len(self.active_requests),
                    "finished_requests": len(self.finished_requests),
                    "cache_hit_rate": f"{cache_efficiency:.2f}%",
                    "tensor_ops": "available",
                    "sdk_mode": self.sdk_mode,
                    "kv_handler": kv_handler_status
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=self.last_health_check
            )

    # === VLLM Core Methods ===

    async def recv_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """
        Receive and process KV caches from vLLM

        Args:
            model_executable: The model being executed
            model_input: Input data for the model
            kv_caches: List of KV cache tensors

        Returns:
            Tuple of (hidden_states, skip_model_forward, updated_model_input)
        """
        try:
            self.total_cache_queries += 1

            # Extract request information
            request_id = getattr(model_input, 'request_id', None)
            if not request_id:
                logger.warning("No request_id in model_input")
                return None, False, model_input

            # Check if we should retrieve from cache
            should_retrieve = await self._should_retrieve_cache(model_input)

            if should_retrieve:
                # SDK模式：直接使用KV处理器
                if self.sdk_mode and self._kv_handler:
                    try:
                        cache_result = await self._kv_handler.retrieve_kv(model_input, kv_caches)

                        if cache_result.get("found"):
                            self.cache_hits += 1
                            hidden_states = cache_result.get("hidden_states")
                            skip_forward = cache_result.get("skip_forward", False)
                            updated_input = cache_result.get("updated_input", model_input)

                            # 缓存命中的情况
                            if skip_forward or hidden_states is not None:
                                logger.debug("SDK KV cache hit")
                                return hidden_states, skip_forward, updated_input
                            elif cache_result.get("kv_caches"):
                                # 有缓存的 KV，但需要继续执行模型
                                cached_kv = cache_result.get("kv_caches")
                                merged_kv = self._merge_kv_caches(cached_kv, kv_caches)
                                # 执行模型的前向传播
                                hidden_states = await self._execute_model_forward(
                                    model_executable, updated_input, merged_kv
                                )
                                return hidden_states, False, updated_input
                    except Exception as e:
                        logger.debug(f"SDK KV cache retrieval failed: {e}")

                # 独立模式：通过路由器调用缓存适配器
                elif not self.sdk_mode and self.core:
                    try:
                        cache_result = await self.core.route_message(
                            source="vllm",
                            target="lmcache",
                            method="retrieve_kv",
                            model_input=model_input,
                            kv_caches=kv_caches
                        )

                        if cache_result and cache_result.get("found"):
                            self.cache_hits += 1
                            cached_kv = cache_result.get("kv_caches")
                            skip_forward = cache_result.get("skip_forward", False)
                            updated_input = cache_result.get("updated_input", model_input)
                            hidden_states = cache_result.get("hidden_states")

                            # 缓存命中的情况
                            if skip_forward or hidden_states is not None:
                                logger.debug("Route KV cache hit")
                                return hidden_states, skip_forward, updated_input
                            elif cached_kv:
                                # 有缓存的 KV，但需要继续执行模型
                                merged_kv = self._merge_kv_caches(cached_kv, kv_caches)
                                # 执行模型的前向传播
                                hidden_states = await self._execute_model_forward(
                                    model_executable, updated_input, merged_kv
                                )
                                return hidden_states, False, updated_input
                    except Exception as e:
                        logger.debug(f"Route KV cache retrieval failed: {e}")

            # 缓存未命中的情况：不执行模型前向传播，返回 None
            # 这表示需要由调用者来执行正常的推理流程
            logger.debug("KV cache miss, proceeding with normal inference")
            return None, False, model_input

        except Exception as e:
            self.log_error(e, {"operation": "recv_kv_caches"})
            return None, False, model_input  # 出错时继续正常流程

    async def send_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, Any]
    ) -> None:
        """
        Send KV caches to cache system

        Args:
            model_executable: The model that was executed
            model_input: Input data that was processed
            kv_caches: KV cache tensors to store
            hidden_or_intermediate_states: Model output states
        """
        try:
            # Extract request information
            request_id = getattr(model_input, 'request_id', None)
            if not request_id:
                return

            # Check if we should store to cache
            should_store = await self._should_store_cache(model_input)

            if should_store:
                # SDK模式：直接使用KV处理器
                if self.sdk_mode and self._kv_handler:
                    try:
                        success = await self._kv_handler.store_kv(
                            model_input,
                            kv_caches,
                            hidden_or_intermediate_states,
                            metadata={
                                'timestamp': asyncio.get_event_loop().time(),
                                'model_name': self.model_name,
                                'tensor_parallel_size': self.tensor_parallel_size
                            }
                        )

                        if success:
                            logger.debug("Successfully stored KV to SDK cache")
                            # Update request tracking
                            if request_id in self.active_requests:
                                self.active_requests[request_id].kv_cached = True
                        else:
                            logger.debug("Failed to store KV to SDK cache")
                    except Exception as e:
                        logger.warning(f"SDK KV storage failed: {e}")

                # 独立模式：通过路由器调用缓存适配器
                elif not self.sdk_mode and self.core:
                    try:
                        await self.core.route_message(
                            source="vllm",
                            target="lmcache",
                            method="store_kv",
                            model_input=model_input,
                            kv_caches=kv_caches,
                            hidden_states=hidden_or_intermediate_states
                        )

                        # Update request tracking
                        if request_id in self.active_requests:
                            self.active_requests[request_id].kv_cached = True

                        logger.debug("Successfully stored KV via route")
                    except Exception as e:
                        logger.warning(f"Route KV storage failed: {e}")

        except Exception as e:
            self.log_error(e, {"operation": "send_kv_caches"})

    async def get_finished(
            self,
            finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        """
        Handle finished requests

        Args:
            finished_req_ids: Set of finished request IDs

        Returns:
            Tuple of (finished_ids, cancelled_ids)
        """
        try:
            # Update finished requests
            self.finished_requests.update(finished_req_ids)

            # Clean up active requests
            for req_id in finished_req_ids:
                if req_id in self.active_requests:
                    del self.active_requests[req_id]

            # Notify cache system about finished requests
            if finished_req_ids:
                # SDK模式：直接使用KV处理器
                if self.sdk_mode and self._kv_handler:
                    try:
                        cleaned_count = await self._kv_handler.cleanup_finished(list(finished_req_ids))
                        logger.debug(f"SDK cleaned {cleaned_count} cache entries for {len(finished_req_ids)} requests")
                    except Exception as e:
                        logger.warning(f"SDK cache cleanup failed: {e}")

                # 独立模式：通过路由器
                elif not self.sdk_mode and self.core:
                    try:
                        await self.core.route_message(
                            source="vllm",
                            target="lmcache",
                            method="cleanup_finished",
                            request_ids=finished_req_ids
                        )
                        logger.debug(f"Route cleaned cache for {len(finished_req_ids)} requests")
                    except Exception as e:
                        logger.warning(f"Route cache cleanup failed: {e}")

            return finished_req_ids, None

        except Exception as e:
            self.log_error(e, {"operation": "get_finished"})
            return finished_req_ids, None

    # === Helper Methods ===

    async def _should_retrieve_cache(self, model_input: Any) -> bool:
        """Determine if cache retrieval should be attempted"""
        try:
            # 检查是否启用了前缀缓存
            if not self.enable_prefix_caching:
                return False

            # 检查是否是 prefill 请求
            is_prefill = getattr(model_input, 'is_prompt', False)
            if not is_prefill:
                return False

            # 检查序列长度
            seq_len = getattr(model_input, 'seq_len', 0)
            min_cache_seq_len = self.config.get('min_cache_seq_len', 32)
            if seq_len < min_cache_seq_len:
                return False

            return True

        except Exception as e:
            logger.debug(f"Error in _should_retrieve_cache: {e}")
            return False

    async def _should_store_cache(self, model_input: Any) -> bool:
        """Determine if cache storage should be attempted"""
        try:
            # 检查是否启用了缓存
            if not self.enable_prefix_caching:
                return False

            # 检查是否是完成的 prefill
            is_prefill = getattr(model_input, 'is_prompt', False)
            do_sample = getattr(model_input, 'do_sample', True)

            if is_prefill and do_sample:
                return True

            # 检查解码阶段的缓存
            if self.config.get('cache_decode_tokens', False):
                seq_len = getattr(model_input, 'seq_len', 0)
                chunk_size = self.config.get('chunk_size', 64)
                if seq_len > 0 and seq_len % chunk_size == 0:
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error in _should_store_cache: {e}")
            return False

    def _merge_kv_caches(
            self,
            cached_kv: List[torch.Tensor],
            new_kv: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Merge cached KV with new KV tensors"""
        merged = []

        for cached, new in zip(cached_kv, new_kv):
            if cached is None and new is None:
                merged.append(None)
            elif cached is None:
                merged.append(new)
            elif new is None:
                merged.append(cached)
            else:
                # Concatenate along sequence dimension
                try:
                    merged.append(torch.cat([cached, new], dim=-2))
                except Exception as e:
                    logger.warning(f"Failed to merge KV cache: {e}")
                    merged.append(new)

        return merged

    async def _execute_model_forward(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> torch.Tensor:
        """Execute model forward pass"""
        # This is a placeholder - actual implementation would call vLLM's forward
        # For now, return a dummy tensor
        batch_size = getattr(model_input, 'batch_size', 1)
        seq_len = getattr(model_input, 'seq_len', 512)
        hidden_size = self.config.get('hidden_size', 4096)

        return torch.randn(batch_size, seq_len, hidden_size)

    async def _register_routes(self):
        """Register routes with the core router (独立模式)"""
        if not self.core or self.sdk_mode:
            return

        # Register this adapter with the core
        self.core.router.register_adapter("vllm", self)

    def _reset_metrics(self):
        """Reset cache metrics"""
        self.cache_hit_rate = 0.0
        self.total_cache_queries = 0
        self.cache_hits = 0

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """Get VLLM specific metrics"""
        cache_hit_rate = (self.cache_hits / max(self.total_cache_queries, 1)) * 100

        return {
            "active_requests": len(self.active_requests),
            "finished_requests": len(self.finished_requests),
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "cache_queries": self.total_cache_queries,
            "cache_hits": self.cache_hits,
            "model_name": self.model_name,
            "tensor_parallel": self.tensor_parallel_size,
            "prefix_caching": self.enable_prefix_caching,
            "sdk_mode": self.sdk_mode,
            "kv_handler_available": self._kv_handler is not None
        }

    # === Public API Methods ===

    async def register_request(self, request_id: str, tokens: List[int]) -> None:
        """Register a new inference request"""
        self.active_requests[request_id] = VLLMRequest(
            request_id=request_id,
            tokens=tokens,
            status=VLLMRequestStatus.PREFILL,
            seq_len=len(tokens)
        )

    async def update_request_status(
            self,
            request_id: str,
            status: VLLMRequestStatus
    ) -> None:
        """Update request processing status"""
        if request_id in self.active_requests:
            self.active_requests[request_id].status = status

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "total_queries": self.total_cache_queries,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{(self.cache_hits / max(self.total_cache_queries, 1)) * 100:.2f}%",
            "active_cached_requests": sum(
                1 for req in self.active_requests.values() if req.kv_cached
            ),
            "mode": "SDK" if self.sdk_mode else "standalone",
            "kv_handler_connected": self._kv_handler is not None
        }

    # === SDK集成特定方法 ===

    def set_kv_handler(self, kv_handler) -> None:
        """设置KV处理器引用（SDK模式下使用）"""
        self._kv_handler = kv_handler
        logger.info("KV handler reference updated")

    def get_sdk_info(self) -> Dict[str, Any]:
        """获取SDK集成信息"""
        return {
            "sdk_mode": self.sdk_mode,
            "kv_handler_available": self._kv_handler is not None,
            "core_available": self.core is not None,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version
        }