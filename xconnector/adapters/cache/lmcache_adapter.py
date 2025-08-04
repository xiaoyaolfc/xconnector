# xconnector/adapters/cache/lmcache_adapter.py
"""
LMCache Adapter for XConnector - 极简版

直接包装现有的LMCache connector，提供统一接口
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from xconnector.adapters.base_adapter import BaseAdapter
from xconnector.interfaces.base_interface import (
    HealthStatus, HealthCheckResult, Capability
)
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class LMCacheAdapter(BaseAdapter):
    """
    LMCache适配器 - 极简版

    直接包装LMCache的现有connector实现
    """

    __version__ = "1.0.0"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["lmcache"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # 检测SDK模式
        self.sdk_mode = core_instance is not None and hasattr(core_instance, 'sdk_mode')

        # LMCache connector实例
        self.lmcache_connector = None

        # 简单统计
        self.total_requests = 0
        self.cache_hits = 0

        logger.info(f"LMCacheAdapter initialized (SDK mode: {self.sdk_mode})")

    # === BaseAdapter必需方法 ===

    async def _initialize_impl(self) -> bool:
        """初始化LMCache connector"""
        try:
            # 尝试导入并创建LMCache connector
            from lmcache_connector import LMCacheConnector

            # 这里需要vllm config，在实际使用时会通过update_cache_config设置
            # 暂时用None，稍后更新
            self.lmcache_connector = None

            logger.info("LMCache connector ready for initialization")
            return True

        except ImportError:
            logger.warning("LMCache not available, using mock implementation")
            # 创建一个简单的mock实现
            self.lmcache_connector = MockLMCacheConnector()
            return True
        except Exception as e:
            logger.error(f"LMCache initialization failed: {e}")
            return False

    async def _start_impl(self) -> bool:
        """启动适配器"""
        # SDK模式下注册到VLLM适配器
        if self.sdk_mode and self.core:
            vllm_adapter = self._get_vllm_adapter()
            if vllm_adapter and hasattr(vllm_adapter, 'set_kv_handler'):
                vllm_adapter.set_kv_handler(self)
                logger.info("Connected to VLLM adapter in SDK mode")

        return True

    async def _stop_impl(self) -> bool:
        """停止适配器"""
        if self.lmcache_connector and hasattr(self.lmcache_connector, 'close'):
            self.lmcache_connector.close()
        return True

    def get_capabilities(self) -> Dict[str, Capability]:
        """返回适配器能力"""
        return {
            "kv_cache": Capability(
                name="kv_cache",
                description="KV cache via LMCache",
                version="1.0.0",
                supported=True,
                parameters={"sdk_mode": self.sdk_mode}
            )
        }

    async def _health_check_impl(self) -> Optional[HealthCheckResult]:
        """健康检查"""
        status = HealthStatus.HEALTHY if self.lmcache_connector else HealthStatus.DEGRADED

        return HealthCheckResult(
            status=status,
            message="LMCache adapter operational",
            timestamp=datetime.now(),
            details={
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "connector_available": self.lmcache_connector is not None
            }
        )

    # === KV缓存接口（给VLLM适配器调用） ===

    async def retrieve_kv(self, model_input: Any, kv_caches: List) -> Dict[str, Any]:
        """检索KV缓存"""
        self.total_requests += 1

        if not self.lmcache_connector:
            return {"found": False}

        try:
            # 调用LMCache connector的方法
            result = self.lmcache_connector.recv_kv_caches_and_hidden_states(
                None, model_input, kv_caches
            )

            hidden_states, skip_forward, updated_input = result

            if skip_forward or hidden_states is not None:
                self.cache_hits += 1
                return {
                    "found": True,
                    "hidden_states": hidden_states,
                    "skip_forward": skip_forward,
                    "updated_input": updated_input
                }

            return {"found": False}

        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
            return {"found": False}

    async def store_kv(self, model_input: Any, kv_caches: List,
                       hidden_states: Any, metadata: Optional[Dict] = None) -> bool:
        """存储KV缓存"""
        if not self.lmcache_connector:
            return False

        try:
            # 调用LMCache connector的方法
            self.lmcache_connector.send_kv_caches_and_hidden_states(
                None, model_input, kv_caches, hidden_states
            )
            return True

        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")
            return False

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """清理完成的请求"""
        # LMCache connector通常在内部处理这个
        return len(request_ids)

    # === 配置更新（重要！） ===

    def update_cache_config(self, vllm_config: Dict[str, Any]) -> bool:
        """
        从vLLM配置更新并创建LMCache connector
        这是关键方法！
        """
        try:
            if not vllm_config:
                return False

            # 如果已经有connector，先关闭
            if self.lmcache_connector and hasattr(self.lmcache_connector, 'close'):
                self.lmcache_connector.close()

            # 创建新的LMCache connector
            from lmcache_connector import LMCacheConnector

            # 模拟vLLM config结构（实际使用时会传入真实的VllmConfig对象）
            class MockVllmConfig:
                def __init__(self, config_dict):
                    self.model_config = config_dict.get('model_config')
                    self.parallel_config = config_dict.get('parallel_config')
                    self.cache_config = config_dict.get('cache_config')
                    self.kv_transfer_config = config_dict.get('kv_transfer_config', {})

            mock_config = MockVllmConfig(vllm_config)

            # 创建LMCache connector实例
            self.lmcache_connector = LMCacheConnector(
                rank=0,  # 这些参数在实际使用时需要正确设置
                local_rank=0,
                config=mock_config
            )

            logger.info("LMCache connector created with vLLM config")
            return True

        except Exception as e:
            logger.error(f"Failed to update cache config: {e}")
            # 创建mock实现作为fallback
            self.lmcache_connector = MockLMCacheConnector()
            return False

    def _get_vllm_adapter(self):
        """获取VLLM适配器引用"""
        if not self.core:
            return None

        # 尝试获取VLLM适配器
        for name, adapter in getattr(self.core, 'inference_adapters', {}).items():
            if 'vllm' in name.lower():
                return adapter

        return None

    # === 统计方法 ===

    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{hit_rate:.2f}%",
            "connector_type": type(self.lmcache_connector).__name__ if self.lmcache_connector else "None"
        }


class MockLMCacheConnector:
    """Mock LMCache connector for fallback"""

    def recv_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches):
        # 总是返回cache miss
        return None, False, model_input

    def send_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches, hidden_states):
        # Mock存储，什么都不做
        pass

    def close(self):
        pass


# === 使用示例 ===

if __name__ == "__main__":
    async def test_adapter():
        # 创建适配器
        adapter = LMCacheAdapter(None, {"storage_backend": "memory"})

        # 初始化
        await adapter.initialize()
        await adapter.start()

        # 模拟vLLM配置更新
        vllm_config = {
            "model_config": {"model": "test-model"},
            "cache_config": {"block_size": 16},
            "parallel_config": {"tensor_parallel_size": 1}
        }

        success = adapter.update_cache_config(vllm_config)
        print(f"Config update: {'Success' if success else 'Failed'}")

        # 获取统计
        stats = adapter.get_cache_statistics()
        print(f"Statistics: {stats}")

        await adapter.stop()


    import asyncio

    asyncio.run(test_adapter())