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
from xconnector.interfaces.cache_manager import CacheManagerInterface, CacheResult, CacheStatus, CacheStats
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    tokens: List[int]
    seq_len: int
    layer_count: int
    stored_at: datetime
    access_count: int = 0


class LMCacheAdapter(BaseAdapter, CacheManagerInterface):
    """
    LMCache Adapter for XConnector

    Provides KV cache management using LMCache system for efficient
    storage and retrieval of transformer key-value caches.
    """

    __version__ = "1.0.0"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["torch", "lmcache"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        # 调用两个父类的初始化
        BaseAdapter.__init__(self, core_instance, config)

        # LMCache configuration
        self.cache_config = config.get("cache_config", {})
        self.storage_backend = config.get("storage_backend", "local")
        self.max_cache_size = config.get("max_cache_size", 1024)
        self.enable_compression = config.get("enable_compression", True)

        # 统一的统计信息
        self.total_queries = 0
        self.hit_count = 0
        self.miss_count = 0

        # Cache management
        self.cache_entries: Dict[str, CacheEntry] = {}

        # LMCache engine instance (will be initialized during setup)
        self.lmcache_engine = None
        self.lmcache_config = None

        logger.info(f"LMCacheAdapter initialized with backend: {self.storage_backend}")

    # === CacheManagerInterface 基础方法 ===

    async def initialize(self) -> bool:
        """统一的初始化方法"""
        return await self._initialize_impl()

    async def start(self) -> bool:
        """统一的启动方法"""
        return await self._start_impl()

    async def stop(self) -> bool:
        """统一的停止方法"""
        return await self._stop_impl()

    # === 通用缓存接口 ===

    async def get(self, key: str, default: Any = None) -> CacheResult:
        """通用缓存获取方法"""
        self.total_queries += 1

        if key in self.cache_entries:
            self.hit_count += 1
            entry = self.cache_entries[key]
            return CacheResult(
                status=CacheStatus.HIT,
                found=True,
                data=entry,
                metadata={
                    "access_count": entry.access_count,
                    "stored_at": entry.stored_at.isoformat()
                }
            )
        else:
            self.miss_count += 1
            return CacheResult(
                status=CacheStatus.MISS,
                found=False,
                data=default
            )

    async def set(
            self,
            key: str,
            value: Any,
            ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """通用缓存设置方法"""
        try:
            # 根据 value 类型决定如何存储
            if isinstance(value, dict) and "tokens" in value:
                # 创建 CacheEntry 对象
                entry = CacheEntry(
                    key=key,
                    tokens=value.get("tokens", []),
                    seq_len=value.get("seq_len", 0),
                    layer_count=value.get("layer_count", 0),
                    stored_at=datetime.now()
                )
                self.cache_entries[key] = entry
            else:
                # 直接存储原始值
                entry = CacheEntry(
                    key=key,
                    tokens=[],
                    seq_len=0,
                    layer_count=0,
                    stored_at=datetime.now()
                )
                # 将 value 存储在 entry 的额外字段中
                entry.raw_data = value
                self.cache_entries[key] = entry

            return True
        except Exception as e:
            self.log_error(e, {"operation": "set", "key": key})
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        try:
            if key in self.cache_entries:
                del self.cache_entries[key]
                return True
            return False
        except Exception as e:
            self.log_error(e, {"operation": "delete", "key": key})
            return False

    async def exists(self, key: str) -> bool:
        """检查缓存项是否存在"""
        return key in self.cache_entries

    async def clear(self) -> bool:
        """清空所有缓存"""
        try:
            self.cache_entries.clear()
            # 重置统计信息
            self.total_queries = 0
            self.hit_count = 0
            self.miss_count = 0
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            self.log_error(e, {"operation": "clear"})
            return False

    async def get_stats(self) -> CacheStats:
        """获取统一格式的缓存统计信息"""
        hit_rate = (self.hit_count / max(self.total_queries, 1)) * 100

        return CacheStats(
            total_queries=self.total_queries,
            hit_count=self.hit_count,
            miss_count=self.miss_count,
            hit_rate=hit_rate,
            total_size=len(self.cache_entries) * 100,  # 估算大小
            entry_count=len(self.cache_entries)
        )

    # === KV 缓存专用接口 ===

    async def retrieve_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> CacheResult:
        """
        检索 KV 缓存 - 返回统一的 CacheResult
        """
        try:
            self.total_queries += 1

            # 检查是否应该尝试检索
            if not hasattr(self, 'lmcache_should_retrieve'):
                return CacheResult(status=CacheStatus.ERROR, found=False, error="LMCache not initialized")

            retrieve_status = self.lmcache_should_retrieve(model_input)

            if retrieve_status == self.RetrieveStatus.MISS:
                self.miss_count += 1
                return CacheResult(status=CacheStatus.MISS, found=False)

            # 尝试从缓存中检索
            updated_input, skip_forward, hidden_states = self.lmcache_retrieve_kv(
                None,
                model_input,
                self._get_dummy_cache_config(),
                kv_caches,
                retrieve_status
            )

            if skip_forward or hidden_states is not None:
                self.hit_count += 1

                # 更新缓存条目访问计数
                cache_key = self._generate_cache_key(model_input)
                if cache_key in self.cache_entries:
                    self.cache_entries[cache_key].access_count += 1

                return CacheResult(
                    status=CacheStatus.HIT,
                    found=True,
                    data={
                        "kv_caches": kv_caches,
                        "hidden_states": hidden_states,
                        "skip_forward": skip_forward,
                        "updated_input": updated_input
                    }
                )
            else:
                self.miss_count += 1
                return CacheResult(status=CacheStatus.MISS, found=False)

        except Exception as e:
            self.log_error(e, {"operation": "retrieve_kv"})
            return CacheResult(
                status=CacheStatus.ERROR,
                found=False,
                error=str(e)
            )

    async def store_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any],
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        存储 KV 缓存 - 返回布尔值
        """
        try:
            # 检查是否应该存储
            if not hasattr(self, 'lmcache_should_store'):
                return False

            store_status = self.lmcache_should_store(model_input)

            if store_status == self.StoreStatus.SKIP:
                return False

            # 存储到缓存
            self.lmcache_store_kv(
                self._get_dummy_model_config(),
                self._get_dummy_parallel_config(),
                self._get_dummy_cache_config(),
                None,
                model_input,
                kv_caches,
                store_status
            )

            # 更新缓存条目元数据
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

            return True

        except Exception as e:
            self.log_error(e, {"operation": "store_kv"})
            return False

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """清理完成请求的缓存"""
        count = 0
        try:
            # Remove finished requests from cache entries
            keys_to_remove = []
            for key, entry in self.cache_entries.items():
                # Simple cleanup logic
                if any(str(req_id) in key for req_id in request_ids):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache_entries[key]
                count += 1

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")

            return count

        except Exception as e:
            self.log_error(e, {"operation": "cleanup_finished"})
            return count

    # === BaseAdapter Required Methods ===

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

            # Initialize cache engine with dummy configs
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
            hit_rate = (self.hit_count / max(self.total_queries, 1)) * 100

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

    # === Helper Methods ===

    def _generate_cache_key(self, model_input: Any) -> str:
        """Generate cache key from model input"""
        request_id = getattr(model_input, 'request_id', 'unknown')
        tokens = getattr(model_input, 'input_tokens', [])

        if hasattr(tokens, 'tolist'):
            token_hash = hash(tuple(tokens.tolist()))
        else:
            token_hash = hash(str(tokens))

        return f"{request_id}_{token_hash}"

    def _initialize_dummy_configs(self):
        """Initialize dummy configurations for LMCache"""
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
        hit_rate = (self.hit_count / max(self.total_queries, 1)) * 100

        return {
            "cache_entries": len(self.cache_entries),
            "cache_hit_count": self.hit_count,
            "cache_miss_count": self.miss_count,
            "total_queries": self.total_queries,
            "hit_rate": f"{hit_rate:.2f}%",
            "storage_backend": self.storage_backend,
            "max_cache_size": self.max_cache_size,
            "compression_enabled": self.enable_compression
        }

    # === 向后兼容的公共 API 方法 ===

    def get_cache_statistics(self) -> Dict[str, Any]:
        """保持向后兼容的统计方法"""
        hit_rate = (self.hit_count / max(self.total_queries, 1)) * 100

        return {
            "total_entries": len(self.cache_entries),
            "total_queries": self.total_queries,
            "cache_hits": self.hit_count,
            "cache_misses": self.miss_count,
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
        """向后兼容的清空缓存方法"""
        return await self.clear()

    async def get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Get specific cache entry"""
        return self.cache_entries.get(cache_key)

    def list_cache_entries(self) -> List[str]:
        """List all cache entry keys"""
        return list(self.cache_entries.keys())

    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """列出缓存键"""
        if prefix is None:
            return list(self.cache_entries.keys())
        else:
            return [key for key in self.cache_entries.keys() if key.startswith(prefix)]

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