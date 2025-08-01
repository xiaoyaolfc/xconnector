# xconnector/sdk/handlers/kv_cache.py
"""
KV缓存处理器

提供统一的KV缓存操作接口，封装不同缓存后端的具体实现，
为上层应用提供简单易用的缓存服务。
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import torch

from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class CacheOperation(Enum):
    """缓存操作类型"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class CacheRequest:
    """缓存请求"""
    operation: CacheOperation
    key: Optional[str] = None
    value: Any = None
    metadata: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None


@dataclass
class CacheResponse:
    """缓存响应"""
    success: bool
    found: bool = False
    data: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class KVCacheHandler:
    """
    KV缓存处理器

    提供统一的KV缓存操作接口，支持多种缓存后端
    """

    def __init__(self, sdk_instance):
        """
        初始化KV缓存处理器

        Args:
            sdk_instance: XConnectorSDK实例
        """
        self.sdk = sdk_instance
        self.cache_adapters: Dict[str, Any] = {}
        self.default_adapter: Optional[str] = None

        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

        # 配置
        self.config = sdk_instance.config.integration.get('kv_cache', {})
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.enable_batching = self.config.get('enable_batching', True)

        logger.info("KVCacheHandler initialized")

    async def initialize(self) -> bool:
        """
        初始化处理器

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 发现可用的缓存适配器
            await self._discover_cache_adapters()

            # 设置默认适配器
            self._setup_default_adapter()

            logger.info(f"KVCacheHandler initialized with adapters: {list(self.cache_adapters.keys())}")
            return True

        except Exception as e:
            logger.error(f"KVCacheHandler initialization failed: {e}")
            return False

    async def start(self) -> bool:
        """启动处理器"""
        try:
            # 启动所有缓存适配器
            for name, adapter in self.cache_adapters.items():
                if hasattr(adapter, 'start'):
                    await adapter.start()

            logger.info("KVCacheHandler started")
            return True

        except Exception as e:
            logger.error(f"KVCacheHandler start failed: {e}")
            return False

    async def stop(self) -> bool:
        """停止处理器"""
        try:
            # 停止所有缓存适配器
            for name, adapter in self.cache_adapters.items():
                if hasattr(adapter, 'stop'):
                    await adapter.stop()

            logger.info("KVCacheHandler stopped")
            return True

        except Exception as e:
            logger.error(f"KVCacheHandler stop failed: {e}")
            return False

    # === KV缓存核心接口 ===

    async def retrieve_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            adapter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        检索KV缓存

        Args:
            model_input: 模型输入
            kv_caches: KV缓存张量列表
            adapter_name: 指定使用的适配器名称

        Returns:
            Dict: 缓存检索结果
        """
        self.total_requests += 1

        try:
            # 选择适配器
            adapter = self._select_adapter(adapter_name)
            if not adapter:
                return self._create_miss_response("No adapter available")

            # 调用适配器的检索方法
            if hasattr(adapter, 'retrieve_kv'):
                result = await adapter.retrieve_kv(model_input, kv_caches)
            else:
                # 回退到通用缓存接口
                cache_key = self._generate_cache_key(model_input)
                result = await adapter.get(cache_key)

            # 处理结果
            if self._is_cache_hit(result):
                self.cache_hits += 1
                return self._process_cache_hit(result)
            else:
                self.cache_misses += 1
                return self._create_miss_response("Cache miss")

        except Exception as e:
            self.errors += 1
            logger.error(f"KV retrieval failed: {e}")

            if self.enable_fallback:
                return self._create_miss_response(f"Error: {e}")
            else:
                raise

    async def store_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any],
            adapter_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        存储KV缓存

        Args:
            model_input: 模型输入
            kv_caches: KV缓存张量列表
            hidden_states: 隐藏状态
            adapter_name: 指定使用的适配器名称
            metadata: 额外元数据

        Returns:
            bool: 存储是否成功
        """
        try:
            # 选择适配器
            adapter = self._select_adapter(adapter_name)
            if not adapter:
                logger.warning("No adapter available for KV storage")
                return False

            # 调用适配器的存储方法
            if hasattr(adapter, 'store_kv'):
                success = await adapter.store_kv(
                    model_input, kv_caches, hidden_states, metadata
                )
            else:
                # 回退到通用缓存接口
                cache_key = self._generate_cache_key(model_input)
                cache_data = {
                    'kv_caches': kv_caches,
                    'hidden_states': hidden_states,
                    'metadata': metadata
                }
                success = await adapter.set(cache_key, cache_data)

            if not success:
                logger.warning("KV storage failed")

            return success

        except Exception as e:
            self.errors += 1
            logger.error(f"KV storage failed: {e}")

            if self.enable_fallback:
                return False
            else:
                raise

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """
        清理完成请求的缓存

        Args:
            request_ids: 完成的请求ID列表

        Returns:
            int: 清理的缓存条目数量
        """
        total_cleaned = 0

        try:
            for adapter_name, adapter in self.cache_adapters.items():
                if hasattr(adapter, 'cleanup_finished'):
                    cleaned = await adapter.cleanup_finished(request_ids)
                    total_cleaned += cleaned
                    logger.debug(f"Adapter {adapter_name} cleaned {cleaned} entries")

            logger.info(f"Cleaned up {total_cleaned} cache entries for {len(request_ids)} requests")
            return total_cleaned

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return total_cleaned

    async def clear_cache(self, adapter_name: Optional[str] = None) -> bool:
        """
        清空缓存

        Args:
            adapter_name: 指定清空的适配器，None表示清空所有

        Returns:
            bool: 是否成功
        """
        try:
            if adapter_name:
                # 清空指定适配器
                adapter = self.cache_adapters.get(adapter_name)
                if adapter and hasattr(adapter, 'clear'):
                    await adapter.clear()
                    logger.info(f"Cleared cache for adapter: {adapter_name}")
                    return True
                else:
                    logger.warning(f"Adapter {adapter_name} not found or doesn't support clear")
                    return False
            else:
                # 清空所有适配器
                for name, adapter in self.cache_adapters.items():
                    if hasattr(adapter, 'clear'):
                        await adapter.clear()
                        logger.debug(f"Cleared cache for adapter: {name}")

                logger.info("Cleared all caches")
                return True

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    # === 统计和监控接口 ===

    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100

        stats = {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'errors': self.errors,
            'adapters': list(self.cache_adapters.keys()),
            'default_adapter': self.default_adapter
        }

        # 获取各适配器的统计信息
        adapter_stats = {}
        for name, adapter in self.cache_adapters.items():
            if hasattr(adapter, 'get_stats'):
                try:
                    adapter_stats[name] = adapter.get_stats()
                except Exception as e:
                    adapter_stats[name] = {'error': str(e)}

        if adapter_stats:
            stats['adapter_stats'] = adapter_stats

        return stats

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        logger.info("Cache statistics reset")

    # === 内部辅助方法 ===

    async def _discover_cache_adapters(self) -> None:
        """发现可用的缓存适配器"""
        if not self.sdk.connector:
            return

        # 从连接器中获取缓存适配器
        for name, adapter in self.sdk.connector.cache_adapters.items():
            self.cache_adapters[name] = adapter
            logger.debug(f"Discovered cache adapter: {name}")

    def _setup_default_adapter(self) -> None:
        """设置默认适配器"""
        if not self.cache_adapters:
            logger.warning("No cache adapters available")
            return

        # 优先使用配置中指定的默认适配器
        default_name = self.config.get('default_adapter')
        if default_name and default_name in self.cache_adapters:
            self.default_adapter = default_name
        else:
            # 使用第一个可用的适配器
            self.default_adapter = next(iter(self.cache_adapters.keys()))

        logger.info(f"Default cache adapter: {self.default_adapter}")

    def _select_adapter(self, adapter_name: Optional[str] = None):
        """选择要使用的适配器"""
        if adapter_name:
            return self.cache_adapters.get(adapter_name)
        elif self.default_adapter:
            return self.cache_adapters.get(self.default_adapter)
        else:
            return None

    def _generate_cache_key(self, model_input: Any) -> str:
        """生成缓存键"""
        try:
            # 简化的键生成逻辑
            request_id = getattr(model_input, 'request_id', 'unknown')

            # 尝试获取输入tokens
            tokens = getattr(model_input, 'input_tokens', None)
            if tokens is None:
                tokens = getattr(model_input, 'tokens', [])

            if hasattr(tokens, 'tolist'):
                token_hash = hash(tuple(tokens.tolist()))
            else:
                token_hash = hash(str(tokens))

            return f"kv_cache_{request_id}_{token_hash}"

        except Exception as e:
            logger.warning(f"Cache key generation failed: {e}")
            return f"kv_cache_fallback_{id(model_input)}"

    def _is_cache_hit(self, result: Any) -> bool:
        """判断是否为缓存命中"""
        if isinstance(result, dict):
            return result.get('found', False) or result.get('status') == 'HIT'
        else:
            return result is not None

    def _process_cache_hit(self, result: Any) -> Dict[str, Any]:
        """处理缓存命中结果"""
        if isinstance(result, dict):
            # 如果结果已经是字典格式，直接返回
            if 'found' in result:
                return result
            else:
                # 转换为标准格式
                return {
                    'found': True,
                    'data': result.get('data'),
                    'kv_caches': result.get('kv_caches'),
                    'hidden_states': result.get('hidden_states'),
                    'skip_forward': result.get('skip_forward', False),
                    'updated_input': result.get('updated_input')
                }
        else:
            # 简单的命中结果
            return {
                'found': True,
                'data': result,
                'skip_forward': False
            }

    def _create_miss_response(self, reason: str = "Cache miss") -> Dict[str, Any]:
        """创建缓存未命中响应"""
        return {
            'found': False,
            'data': None,
            'reason': reason,
            'skip_forward': False
        }


# === 工具函数 ===

def create_kv_handler(sdk_instance) -> KVCacheHandler:
    """创建KV缓存处理器的便捷函数"""
    return KVCacheHandler(sdk_instance)


# === 导出 ===

__all__ = [
    'KVCacheHandler',
    'CacheOperation',
    'CacheRequest',
    'CacheResponse',
    'create_kv_handler'
]