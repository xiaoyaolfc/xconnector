# integrations/dynamo/minimal_sdk.py
"""
Minimal XConnector SDK for Dynamo Integration

轻量级的XConnector SDK，专门为Dynamo集成设计
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Get logger
logger = logging.getLogger('xconnector.dynamo.minimal_sdk')


@dataclass
class AdapterInfo:
    """适配器信息"""
    name: str
    type: str
    enabled: bool
    config: Dict[str, Any]


class SimpleKVHandler:
    """简化的KV处理器"""

    def __init__(self, cache_adapter):
        self.cache_adapter = cache_adapter
        self.total_requests = 0
        self.cache_hits = 0

    async def retrieve_kv(self, model_input: Any, kv_caches: List) -> Dict[str, Any]:
        """检索KV缓存"""
        if not self.cache_adapter:
            return {"found": False}

        try:
            result = await self.cache_adapter.retrieve_kv(model_input, kv_caches)
            self.total_requests += 1
            if result.get("found"):
                self.cache_hits += 1
            return result
        except Exception as e:
            logger.debug(f"KV retrieve error: {e}")
            return {"found": False}

    async def store_kv(self, model_input: Any, kv_caches: List,
                       hidden_states: Any) -> bool:
        """存储KV缓存"""
        if not self.cache_adapter:
            return False

        try:
            return await self.cache_adapter.store_kv(
                model_input, kv_caches, hidden_states
            )
        except Exception as e:
            logger.debug(f"KV store error: {e}")
            return False


class MinimalXConnectorSDK:
    """
    Minimal XConnector SDK for Dynamo Integration

    提供最小化的XConnector功能，专注于KV缓存集成
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Minimal XConnector SDK

        Args:
            config: SDK配置
        """
        self.config = config or {}
        self.cache_adapter = None
        self.kv_handler = None
        self.initialized = False

        # 初始化缓存适配器
        self._init_cache_adapter()

        logger.info("MinimalXConnectorSDK initialized")

    def _init_cache_adapter(self):
        """Initialize cache adapter"""
        try:
            from xconnector.adapters.cache.lmcache_adapter import LMCacheAdapter
            self.cache_adapter = LMCacheAdapter()

            # Create KV handler
            self.kv_handler = SimpleKVHandler(self.cache_adapter)

            logger.info("Cache adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache adapter: {e}")
            self.cache_adapter = None
            self.kv_handler = None

    async def initialize(self) -> bool:
        """
        Initialize SDK (async)

        Returns:
            bool: 是否初始化成功
        """
        try:
            if self.cache_adapter:
                await self.cache_adapter.initialize()
                await self.cache_adapter.start()
                logger.info("Cache adapter started successfully")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"SDK initialization failed: {e}")
            return False

    def initialize_sync(self) -> bool:
        """
        Initialize SDK (sync version)

        Returns:
            bool: 是否初始化成功
        """
        try:
            # Basic sync initialization
            self.initialized = True
            logger.info("SDK initialized synchronously")
            return True
        except Exception as e:
            logger.error(f"Sync initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if SDK is ready"""
        return self.initialized and self.cache_adapter is not None

    async def retrieve_kv(self, model_input: Any, kv_caches: List) -> Dict[str, Any]:
        """
        Retrieve KV cache

        Args:
            model_input: 模型输入
            kv_caches: KV缓存列表

        Returns:
            Dict: 检索结果
        """
        if not self.kv_handler:
            return {"found": False}

        return await self.kv_handler.retrieve_kv(model_input, kv_caches)

    async def store_kv(self, model_input: Any, kv_caches: List,
                       hidden_states: Any) -> bool:
        """
        Store KV cache

        Args:
            model_input: 模型输入
            kv_caches: KV缓存列表
            hidden_states: 隐藏状态

        Returns:
            bool: 是否存储成功
        """
        if not self.kv_handler:
            return False

        return await self.kv_handler.store_kv(model_input, kv_caches, hidden_states)

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """
        Clean up finished requests

        Args:
            request_ids: 请求ID列表

        Returns:
            int: 清理的数量
        """
        try:
            if hasattr(self.cache_adapter, 'cleanup_finished'):
                result = await self.cache_adapter.cleanup_finished(request_ids)
                return int(result) if result else len(request_ids)
            else:
                return len(request_ids)
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get simple statistics"""
        if self.kv_handler:
            hit_rate = (self.kv_handler.cache_hits / max(self.kv_handler.total_requests, 1)) * 100
            return {
                "total_requests": self.kv_handler.total_requests,
                "cache_hits": self.kv_handler.cache_hits,
                "hit_rate": f"{hit_rate:.1f}%",
                "adapter_available": self.cache_adapter is not None
            }
        else:
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "hit_rate": "0.0%",
                "adapter_available": False
            }


# Convenience function
def create_minimal_sdk(config: Dict[str, Any]) -> MinimalXConnectorSDK:
    """
    Create Minimal SDK instance

    Args:
        config: 配置字典

    Returns:
        MinimalXConnectorSDK: SDK实例
    """
    return MinimalXConnectorSDK(config)


# Export
__all__ = [
    'MinimalXConnectorSDK',
    'SimpleKVHandler',
    'AdapterInfo',
    'create_minimal_sdk'
]