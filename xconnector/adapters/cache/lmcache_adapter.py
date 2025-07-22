# xconnector/adapters/cache/lmcache_adapter.py
"""
LMCache Adapter for XConnector

This adapter provides KV cache management using LMCache system,
enabling efficient cache storage and retrieval for inference engines.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

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


class CacheStatus(Enum):
    """Cache operation status"""
    HIT = "hit"
    MISS = "miss"
    STORED = "stored"
    ERROR = "error"


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    tokens: List[int]
    seq_len: int
    layer_count: int
    stored_at: datetime
    access_count: int = 0


class LMCacheAdapter(BaseAdapter):
    """
    LMCache Adapter for XConnector

    Provides KV cache management using LMCache system for efficient
    storage and retrieval of transformer key-value caches.
    """

    __version__ = "1.0.0"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["torch", "lmcache"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # LMCache configuration
        self.cache_config = config.get("cache_config", {})
        self.storage_backend = config.get("storage_backend", "local")
        self.max_cache_size = config.get("max_cache_size", 1024)  # MB
        self.enable_compression = config.get("enable_compression", True)

        # Cache management
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.total_queries = 0

        # LMCache engine instance (will be initialized during setup)
        self.lmcache_engine = None
        self.lmcache_config = None

        logger.info(f"LMCacheAdapter initialized with backend: {self.storage_backend}")

    # === Required BaseInterface Methods ===

    async def _initialize_impl(self) -> bool:
        """Initialize LMCache adapter components"""
        try:
            # Import LMCache components
            try:
                from lmcache.experimental.cache_engine import LMCacheEngineBuilder
                from lmcache.integration.vllm.utils import ENGINE_NAME
                from lmcache.integration.vllm.vllm_adapter import (
                    RetrieveStatus, StoreStatus, init_lmcache_engine,
                    lmcache_retrieve_kv, lmcache_should_retrieve,
                    lmcache_should_store, lmcache_store_kv
                )

                # Store references for later use
                self.LMCacheEngineBuilder = LMCacheEngineBuilder
                self.ENGINE_NAME = ENGINE_NAME
                self.RetrieveStatus = RetrieveStatus
                self.StoreStatus = StoreStatus
                self.init_lmcache_engine = init_lmcache_engine
                self.lmcache_retrieve_kv = lmcache_retrieve_kv
                self.lmcache_store_kv = lmcache_store_kv
                self.lmcache_should_retrieve = lmcache_should_retrieve
                self.lmcache_should_store = lmcache_should_store

                logger.info("LMCache components imported successfully")
            except ImportError as e:
                logger.error(f"LMCache not available: {e}")
                return False

            # Initialize cache engine with dummy configs (will be updated when vLLM provides real configs)
            self._initialize_dummy_configs()

            return True
        except Exception as e:
            self.log_error(e, {"operation": "initialize"})
            return False

    async def _start_impl(self) -> bool:
        """Start LMCache adapter services"""
        try:
            # Register cache routes with core if available
            if self.core:
                await self._register_cache_routes()

            logger.info("LMCacheAdapter started successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "start"})
            return False

    async def _stop_impl(self) -> bool:
        """Stop LMCache adapter services"""
        try:
            # Cleanup LMCache engine
            if hasattr(self, 'LMCacheEngineBuilder') and hasattr(self, 'ENGINE_NAME'):
                try:
                    self.LMCacheEngineBuilder.destroy(self.ENGINE_NAME)
                except Exception as e:
                    logger.warning(f"Error destroying LMCache engine: {e}")

            # Clear cache entries
            self.cache_entries.clear()

            logger.info("LMCacheAdapter stopped successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "stop"})
            return False

    def get_capabilities(self) -> Dict[str, Capability]:
        """Return LMCache adapter capabilities"""
        return {
            "kv_cache_storage": Capability(
                name="kv_cache_storage",
                description="Store and retrieve KV caches efficiently",
                version="1.0.0",
                supported=True,
                parameters={
                    "max_cache_size": self.max_cache_size,
                    "compression": self.enable_compression,
                    "backend": self.storage_backend
                }
            ),
            "prefix_caching": Capability(
                name="prefix_caching",
                description="Support for prefix-based KV cache sharing",
                version="1.0.0",
                supported=True,
                parameters={}
            ),
            "cache_analytics": Capability(
                name="cache_analytics",
                description="Detailed cache performance analytics",
                version="1.0.0",
                supported=True,
                parameters={
                    "hit_rate_tracking": True,
                    "entry_statistics": True
                }
            )
        }

    async def _health_check_impl(self) -> Optional[HealthCheckResult]:
        """LMCache specific health check"""
        try:
            # Check cache engine availability
            if not hasattr(self, 'lmcache_engine') or self.lmcache_engine is None:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="LMCache engine not initialized",
                    timestamp=datetime.now()
                )

            # Calculate cache efficiency
            hit_rate = (self.cache_hit_count / max(self.total_queries, 1)) * 100

            # Check memory usage (simplified)
            cache_size_mb = len(self.cache_entries) * 0.1  # Estimated

            health_status = HealthStatus.HEALTHY
            if hit_rate < 10:  # Less than 10% hit rate might indicate issues
                health_status = HealthStatus.DEGRADED
            elif cache_size_mb > self.max_cache_size * 0.9:  # Over 90% capacity
                health_status = HealthStatus.DEGRADED

            return HealthCheckResult(
                status=health_status,
                message="LMCache adapter is operational",
                timestamp=datetime.now(),
                details={
                    "cache_entries": len(self.cache_entries),
                    "hit_rate": f"{hit_rate:.2f}%",
                    "estimated_size_mb": f"{cache_size_mb:.2f}",
                    "queries_processed": self.total_queries
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.now()
            )

    # === Cache Management Methods ===

    async def retrieve_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Retrieve KV caches from LMCache

        Args:
            model_input: Model input metadata
            kv_caches: Current KV cache tensors

        Returns:
            Dict containing cache retrieval results
        """
        try:
            self.total_queries += 1

            # Check if we should attempt retrieval
            if not hasattr(self, 'lmcache_should_retrieve'):
                return {"found": False, "status": CacheStatus.ERROR}

            retrieve_status = self.lmcache_should_retrieve(model_input)

            if retrieve_status == self.RetrieveStatus.MISS:
                self.cache_miss_count += 1
                return {"found": False, "status": CacheStatus.MISS}

            # Attempt to retrieve from cache
            updated_input, skip_forward, hidden_states = self.lmcache_retrieve_kv(
                None,  # model_executable will be provided by vLLM adapter
                model_input,
                self._get_dummy_cache_config(),
                kv_caches,
                retrieve_status
            )

            if skip_forward or hidden_states is not None:
                self.cache_hit_count += 1

                # Update cache entry access count
                cache_key = self._generate_cache_key(model_input)
                if cache_key in self.cache_entries:
                    self.cache_entries[cache_key].access_count += 1

                return {
                    "found": True,
                    "status": CacheStatus.HIT,
                    "kv_caches": kv_caches,
                    "hidden_states": hidden_states,
                    "skip_forward": skip_forward,
                    "updated_input": updated_input
                }
            else:
                self.cache_miss_count += 1
                return {"found": False, "status": CacheStatus.MISS}

        except Exception as e:
            self.log_error(e, {"operation": "retrieve_kv"})
            return {"found": False, "status": CacheStatus.ERROR, "error": str(e)}

    async def store_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any]
    ) -> Dict[str, Any]:
        """
        Store KV caches to LMCache

        Args:
            model_input: Model input metadata
            kv_caches: KV cache tensors to store
            hidden_states: Hidden states to store

        Returns:
            Dict containing storage results
        """
        try:
            # Check if we should store
            if not hasattr(self, 'lmcache_should_store'):
                return {"stored": False, "status": CacheStatus.ERROR}

            store_status = self.lmcache_should_store(model_input)

            if store_status == self.StoreStatus.SKIP:
                return {"stored": False, "status": CacheStatus.MISS}

            # Store to cache
            self.lmcache_store_kv(
                self._get_dummy_model_config(),
                self._get_dummy_parallel_config(),
                self._get_dummy_cache_config(),
                None,  # model_executable will be provided by vLLM adapter
                model_input,
                kv_caches,
                store_status
            )

            # Update cache entry metadata
            cache_key = self._generate_cache_key(model_input)
            tokens = getattr(model_input, 'input_tokens', [])
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()

            self.cache_entries[cache_key] = CacheEntry(
                key=cache_key,
                tokens=tokens,
                seq_len=getattr(model_input, 'seq_len', 0),
                layer_count=len(kv_caches),
                stored_at=datetime.now()
            )

            return {"stored": True, "status": CacheStatus.STORED}

        except Exception as e:
            self.log_error(e, {"operation": "store_kv"})
            return {"stored": False, "status": CacheStatus.ERROR, "error": str(e)}

    async def cleanup_finished(self, request_ids: set) -> None:
        """
        Cleanup cache entries for finished requests

        Args:
            request_ids: Set of finished request IDs
        """
        try:
            # Remove finished requests from cache entries
            keys_to_remove = []
            for key, entry in self.cache_entries.items():
                # Simple cleanup logic - in practice, you might want more sophisticated logic
                if any(str(req_id) in key for req_id in request_ids):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache_entries[key]

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")

        except Exception as e:
            self.log_error(e, {"operation": "cleanup_finished"})

    # === Helper Methods ===

    def _generate_cache_key(self, model_input: Any) -> str:
        """Generate cache key from model input"""
        # Simple key generation - in practice, you might want more sophisticated logic
        request_id = getattr(model_input, 'request_id', 'unknown')
        tokens = getattr(model_input, 'input_tokens', [])

        if hasattr(tokens, 'tolist'):
            token_hash = hash(tuple(tokens.tolist()))
        else:
            token_hash = hash(str(tokens))

        return f"{request_id}_{token_hash}"

    def _initialize_dummy_configs(self):
        """Initialize dummy configurations for LMCache"""
        # These will be replaced with real configs from vLLM
        self.dummy_model_config = type('DummyModelConfig', (), {
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'num_key_value_heads': 32,
        })()

        self.dummy_parallel_config = type('DummyParallelConfig', (), {
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
        })()

        self.dummy_cache_config = type('DummyCacheConfig', (), {
            'block_size': 16,
            'num_gpu_blocks': 1000,
            'num_cpu_blocks': 1000,
        })()

    def _get_dummy_model_config(self):
        """Get dummy model config"""
        return getattr(self, 'dummy_model_config', None)

    def _get_dummy_parallel_config(self):
        """Get dummy parallel config"""
        return getattr(self, 'dummy_parallel_config', None)

    def _get_dummy_cache_config(self):
        """Get dummy cache config"""
        return getattr(self, 'dummy_cache_config', None)

    async def _register_cache_routes(self):
        """Register cache routes with the core router"""
        if not self.core or not hasattr(self.core, 'router'):
            return

        # Register this adapter with the core
        self.core.router.register_adapter("lmcache", self)

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """Get LMCache specific metrics"""
        hit_rate = (self.cache_hit_count / max(self.total_queries, 1)) * 100

        return {
            "cache_entries": len(self.cache_entries),
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "total_queries": self.total_queries,
            "hit_rate": f"{hit_rate:.2f}%",
            "storage_backend": self.storage_backend,
            "max_cache_size": self.max_cache_size,
            "compression_enabled": self.enable_compression
        }

    # === Public API Methods ===

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        hit_rate = (self.cache_hit_count / max(self.total_queries, 1)) * 100

        return {
            "total_entries": len(self.cache_entries),
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.cache_miss_count,
            "hit_rate": hit_rate,
            "miss_rate": 100 - hit_rate,
            "average_seq_len": (
                    sum(entry.seq_len for entry in self.cache_entries.values()) /
                    max(len(self.cache_entries), 1)
            ),
            "most_accessed_entries": sorted(
                [(entry.key, entry.access_count) for entry in self.cache_entries.values()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

    async def clear_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache_entries.clear()

            # Reset statistics
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self.total_queries = 0

            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "clear_cache"})
            return False

    async def get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Get specific cache entry"""
        return self.cache_entries.get(cache_key)

    def list_cache_entries(self) -> List[str]:
        """List all cache entry keys"""
        return list(self.cache_entries.keys())

    # === Configuration Update Support ===

    def update_cache_config(self, vllm_config: Dict[str, Any]) -> bool:
        """
        Update cache configuration from vLLM

        Args:
            vllm_config: Configuration from vLLM containing model, parallel, and cache configs

        Returns:
            bool: Whether update was successful
        """
        try:
            # Extract configs from vLLM
            model_config = vllm_config.get('model_config')
            parallel_config = vllm_config.get('parallel_config')
            cache_config = vllm_config.get('cache_config')

            if model_config:
                self.dummy_model_config = model_config
            if parallel_config:
                self.dummy_parallel_config = parallel_config
            if cache_config:
                self.dummy_cache_config = cache_config

            # Re-initialize LMCache engine with new configs
            if all([model_config, parallel_config, cache_config]):
                self.lmcache_engine = self.init_lmcache_engine(
                    model_config, parallel_config, cache_config
                )

            logger.info("Cache configuration updated successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "update_cache_config"})
            return False