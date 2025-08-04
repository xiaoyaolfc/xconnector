# xconnector/integrations/dynamo/minimal_sdk.py
"""
最小化的XConnector SDK

只包含Dynamo集成所需的核心功能：
1. KV缓存处理
2. 基础适配器管理
3. 简化的路由功能

设计原则：
- 极简设计：只实现必要功能
- 快速初始化：启动时间最短
- 容错性强：出错时优雅降级
- 内存占用小：最小资源消耗
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# 获取logger
logger = logging.getLogger('xconnector.dynamo.minimal_sdk')


@dataclass
class AdapterInfo:
    """适配器信息"""
    name: str
    type: str
    class_path: str
    config: Dict[str, Any]
    enabled: bool = True


class MinimalXConnectorSDK:
    """
    最小化的XConnector SDK

    只包含Dynamo需要的核心功能，去除所有非必要组件
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化最小SDK

        Args:
            config: XConnector配置字典
        """
        self.config = config
        self.initialized = False
        self.started = False

        # 核心组件
        self.kv_handler = None
        self.adapters: Dict[str, Any] = {}
        self.adapter_configs: List[AdapterInfo] = []

        # 解析适配器配置
        self._parse_adapter_configs()

        logger.debug(f"MinimalXConnectorSDK created with {len(self.adapter_configs)} adapters")

    def _parse_adapter_configs(self):
        """解析适配器配置"""
        adapters_config = self.config.get('adapters', [])

        for adapter_config in adapters_config:
            if isinstance(adapter_config, dict):
                adapter_info = AdapterInfo(
                    name=adapter_config.get('name', ''),
                    type=adapter_config.get('type', ''),
                    class_path=adapter_config.get('class_path', ''),
                    config=adapter_config.get('config', {}),
                    enabled=adapter_config.get('enabled', True)
                )
                self.adapter_configs.append(adapter_info)

    async def initialize(self) -> bool:
        """
        异步初始化SDK

        Returns:
            bool: 初始化是否成功
        """
        if self.initialized:
            return True

        try:
            logger.debug("Initializing MinimalXConnectorSDK...")

            # 初始化KV缓存适配器（最重要的功能）
            kv_cache_adapter = self._find_cache_adapter()
            if kv_cache_adapter:
                success = await self._init_kv_cache_adapter(kv_cache_adapter)
                if success:
                    logger.info("✓ KV cache adapter initialized")
                else:
                    logger.warning("⚠ KV cache adapter initialization failed")

            self.initialized = True
            logger.debug("MinimalXConnectorSDK initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MinimalXConnectorSDK: {e}")
            return False

    def initialize_sync(self):
        """
        同步初始化（回退方案）
        """
        try:
            logger.debug("Initializing MinimalXConnectorSDK (sync)...")

            # 简化的同步初始化
            kv_cache_adapter = self._find_cache_adapter()
            if kv_cache_adapter:
                self._init_kv_cache_adapter_sync(kv_cache_adapter)
                logger.info("✓ KV cache adapter initialized (sync)")

            self.initialized = True
            logger.debug("MinimalXConnectorSDK initialized successfully (sync)")

        except Exception as e:
            logger.error(f"Failed to sync initialize MinimalXConnectorSDK: {e}")

    def _find_cache_adapter(self) -> Optional[AdapterInfo]:
        """查找缓存适配器配置"""
        for adapter_info in self.adapter_configs:
            if adapter_info.type == 'cache' and adapter_info.enabled:
                return adapter_info
        return None

    async def _init_kv_cache_adapter(self, adapter_info: AdapterInfo) -> bool:
        """
        初始化KV缓存适配器

        Args:
            adapter_info: 适配器信息

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 动态导入适配器类
            adapter_class = self._import_adapter_class(adapter_info.class_path)
            if not adapter_class:
                return False

            # 创建适配器实例（传入self作为core_instance，启用SDK模式）
            adapter_instance = adapter_class(self, adapter_info.config)

            # 简化的初始化流程
            if hasattr(adapter_instance, 'initialize'):
                success = await adapter_instance.initialize()
                if not success:
                    logger.warning(f"Adapter {adapter_info.name} initialization returned False")
                    return False

            # 保存适配器实例
            self.adapters[adapter_info.name] = adapter_instance

            # 创建KV处理器
            self.kv_handler = SimpleKVHandler(adapter_instance)

            return True

        except Exception as e:
            logger.error(f"Failed to initialize adapter {adapter_info.name}: {e}")
            return False

    def _init_kv_cache_adapter_sync(self, adapter_info: AdapterInfo) -> bool:
        """
        同步初始化KV缓存适配器（简化版）

        Args:
            adapter_info: 适配器信息

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 动态导入适配器类
            adapter_class = self._import_adapter_class(adapter_info.class_path)
            if not adapter_class:
                return False

            # 创建适配器实例
            adapter_instance = adapter_class(self, adapter_info.config)

            # 保存适配器实例
            self.adapters[adapter_info.name] = adapter_instance

            # 创建KV处理器
            self.kv_handler = SimpleKVHandler(adapter_instance)

            return True

        except Exception as e:
            logger.error(f"Failed to sync initialize adapter {adapter_info.name}: {e}")
            return False

    def _import_adapter_class(self, class_path: str):
        """
        动态导入适配器类

        Args:
            class_path: 类路径，如 "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter"

        Returns:
            适配器类或None
        """
        try:
            # 分离模块路径和类名
            module_path, class_name = class_path.rsplit('.', 1)

            # 导入模块
            import importlib
            module = importlib.import_module(module_path)

            # 获取类
            adapter_class = getattr(module, class_name)

            logger.debug(f"Successfully imported {class_path}")
            return adapter_class

        except Exception as e:
            logger.error(f"Failed to import adapter class {class_path}: {e}")
            return None

    def get_kv_handler(self):
        """获取KV处理器"""
        return self.kv_handler

    def is_ready(self) -> bool:
        """检查SDK是否就绪"""
        return self.initialized and self.kv_handler is not None

    def get_status(self) -> Dict[str, Any]:
        """获取SDK状态"""
        return {
            "initialized": self.initialized,
            "started": self.started,
            "kv_handler_available": self.kv_handler is not None,
            "adapters_count": len(self.adapters),
            "adapter_names": list(self.adapters.keys())
        }


class SimpleKVHandler:
    """
    简化的KV处理器

    只包装适配器的KV缓存功能，不添加额外复杂性
    """

    def __init__(self, cache_adapter):
        """
        初始化处理器

        Args:
            cache_adapter: 缓存适配器实例
        """
        self.cache_adapter = cache_adapter
        self.total_requests = 0
        self.cache_hits = 0

        logger.debug("SimpleKVHandler created")

    async def retrieve_kv(self, model_input: Any, kv_caches: List) -> Dict[str, Any]:
        """
        检索KV缓存

        Args:
            model_input: 模型输入
            kv_caches: KV缓存张量列表

        Returns:
            Dict: 缓存结果
        """
        self.total_requests += 1

        try:
            # 检查适配器是否有retrieve_kv方法
            if hasattr(self.cache_adapter, 'retrieve_kv'):
                result = await self.cache_adapter.retrieve_kv(model_input, kv_caches)

                # 统一返回格式
                if isinstance(result, dict):
                    if result.get("found"):
                        self.cache_hits += 1
                    return result
                else:
                    # 如果返回的不是字典，包装一下
                    return {"found": False, "data": result}
            else:
                logger.debug("Cache adapter doesn't have retrieve_kv method")
                return {"found": False, "reason": "method_not_available"}

        except Exception as e:
            logger.error(f"KV retrieve failed: {e}")
            return {"found": False, "error": str(e)}

    async def store_kv(self, model_input: Any, kv_caches: List,
                       hidden_states: Any, metadata: Optional[Dict] = None) -> bool:
        """
        存储KV缓存

        Args:
            model_input: 模型输入
            kv_caches: KV缓存张量列表
            hidden_states: 隐藏状态
            metadata: 元数据

        Returns:
            bool: 存储是否成功
        """
        try:
            # 检查适配器是否有store_kv方法
            if hasattr(self.cache_adapter, 'store_kv'):
                result = await self.cache_adapter.store_kv(
                    model_input, kv_caches, hidden_states, metadata
                )
                return bool(result)
            else:
                logger.debug("Cache adapter doesn't have store_kv method")
                return False

        except Exception as e:
            logger.error(f"KV store failed: {e}")
            return False

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """
        清理完成的请求

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
        """获取简单统计信息"""
        hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{hit_rate:.1f}%",
            "adapter_available": self.cache_adapter is not None
        }


# 便捷函数
def create_minimal_sdk(config: Dict[str, Any]) -> MinimalXConnectorSDK:
    """
    创建最小SDK的便捷函数

    Args:
        config: 配置字典

    Returns:
        MinimalXConnectorSDK: SDK实例
    """
    return MinimalXConnectorSDK(config)


# 导出
__all__ = [
    'MinimalXConnectorSDK',
    'SimpleKVHandler',
    'AdapterInfo',
    'create_minimal_sdk'
]