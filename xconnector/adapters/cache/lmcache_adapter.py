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
    LMCache适配器 - 修复版

    支持vLLM内置的LMCache连接器
    """

    __version__ = "1.0.1"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["lmcache", "vllm"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # 检测SDK模式
        self.sdk_mode = core_instance is not None and hasattr(core_instance, 'sdk_mode')

        # LMCache connector实例
        self.lmcache_connector = None
        self.connector_class = None

        # 简单统计
        self.total_requests = 0
        self.cache_hits = 0

        logger.info(f"LMCacheAdapter (Fixed) initialized (SDK mode: {self.sdk_mode})")

    def _find_lmcache_connector(self):
        """智能查找LMCache连接器类"""
        try:
            # 基于之前的测试结果，我们知道正确的导入路径
            try:
                logger.debug("尝试导入vLLM的LMCacheConnector...")
                from vllm.distributed.kv_transfer.kv_connector.lmcache_connector import LMCacheConnector
                logger.info("✅ 成功找到vLLM的LMCacheConnector")
                return LMCacheConnector

            except ImportError as e:
                logger.debug(f"vLLM LMCacheConnector导入失败: {e}")

            # 其他备用路径
            backup_attempts = [
                ("lmcache_connector", "LMCacheConnector"),
                ("lmcache.integration.vllm.lmcache_connector", "LMCacheConnector"),
            ]

            for module_path, class_name in backup_attempts:
                try:
                    logger.debug(f"尝试备用路径: {module_path}.{class_name}")
                    import importlib
                    module = importlib.import_module(module_path)

                    if hasattr(module, class_name):
                        connector_class = getattr(module, class_name)
                        logger.info(f"✅ 找到备用LMCache连接器: {module_path}.{class_name}")
                        return connector_class

                except ImportError:
                    continue
                except Exception as e:
                    logger.debug(f"备用路径错误 {module_path}: {e}")
                    continue

            logger.warning("未找到任何LMCache连接器实现")
            return None

        except Exception as e:
            logger.error(f"查找LMCache连接器时出错: {e}")
            return None

    async def _initialize_impl(self) -> bool:
        """初始化LMCache connector"""
        try:
            # 查找LMCache连接器类
            self.connector_class = self._find_lmcache_connector()

            if self.connector_class:
                logger.info("✅ LMCache connector class found, ready for initialization")
                return True
            else:
                logger.warning("❌ No LMCache connector found, using mock implementation")
                self.lmcache_connector = MockLMCacheConnector()
                return True

        except Exception as e:
            logger.error(f"LMCache initialization failed: {e}")
            # 回退到Mock实现
            self.lmcache_connector = MockLMCacheConnector()
            return True

    async def _start_impl(self) -> bool:
        """启动适配器"""
        if self.connector_class and not self.lmcache_connector:
            logger.info("LMCache connector class available, waiting for vLLM config")

        # SDK模式下注册到VLLM适配器
        if self.sdk_mode and self.core:
            vllm_adapter = self._get_vllm_adapter()
            if vllm_adapter and hasattr(vllm_adapter, 'register_cache_adapter'):
                vllm_adapter.register_cache_adapter(self)
                logger.info("Registered with vLLM adapter")

        return True

    async def _stop_impl(self) -> bool:
        """停止适配器"""
        if self.lmcache_connector and hasattr(self.lmcache_connector, 'close'):
            try:
                self.lmcache_connector.close()
                logger.info("LMCache connector closed")
            except Exception as e:
                logger.warning(f"Error closing LMCache connector: {e}")
        return True

    # === KV缓存操作 ===

    async def retrieve_kv(self, model_input: Any, kv_caches: List) -> Dict[str, Any]:
        """检索KV缓存"""
        self.total_requests += 1

        if not self.lmcache_connector:
            return {"found": False, "reason": "No connector"}

        try:
            # 调用LMCache connector的方法
            result = self.lmcache_connector.recv_kv_caches_and_hidden_states(
                None, model_input, kv_caches
            )

            if result and len(result) >= 2:
                hidden_states, cache_hit = result[0], result[1]
                updated_input = result[2] if len(result) > 2 else model_input

                if cache_hit:
                    self.cache_hits += 1
                    logger.debug("Cache hit!")
                    return {
                        "found": True,
                        "hidden_states": hidden_states,
                        "updated_input": updated_input
                    }
                else:
                    logger.debug("Cache miss")

            return {"found": False}

        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
            return {"found": False, "error": str(e)}

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
            logger.debug("Cache stored successfully")
            return True

        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")
            return False

    async def cleanup_finished(self, request_ids: List[str]) -> int:
        """清理完成的请求"""
        return len(request_ids)

    # === 配置更新（重要！） ===

    def update_cache_config(self, vllm_config: Dict[str, Any]) -> bool:
        """从vLLM配置更新并创建LMCache connector"""
        try:
            if not vllm_config:
                logger.warning("No vLLM config provided")
                return False

            if not self.connector_class:
                logger.warning("No LMCache connector class available")
                return False

            # 如果已经有connector，先关闭
            if self.lmcache_connector and hasattr(self.lmcache_connector, 'close'):
                self.lmcache_connector.close()

            # 模拟vLLM config结构
            class MockVllmConfig:
                def __init__(self, config_dict):
                    self.model_config = config_dict.get('model_config')
                    self.parallel_config = config_dict.get('parallel_config')
                    self.cache_config = config_dict.get('cache_config')
                    self.kv_transfer_config = config_dict.get('kv_transfer_config', {})

            mock_config = MockVllmConfig(vllm_config)

            # 创建LMCache connector实例
            try:
                # 尝试获取world_group，如果没有就创建默认的
                import torch.distributed as dist
                if dist.is_initialized():
                    world_group = dist.group.WORLD
                else:
                    world_group = None
            except:
                world_group = None

            self.lmcache_connector = self.connector_class(
                rank=0,
                local_rank=0,
                config=mock_config,
                world_group=world_group  # 添加缺失的参数
            )

            logger.info(f"🎉 真实的LMCache connector创建成功: {type(self.lmcache_connector).__name__}")
            return True

        except Exception as e:
            logger.error(f"Failed to create LMCache connector: {e}")
            logger.error(f"详细错误: {e.__class__.__name__}: {e}")
            # 回退到mock实现
            self.lmcache_connector = MockLMCacheConnector()
            logger.warning("回退到Mock实现")
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

        is_real_connector = (
                self.lmcache_connector is not None and
                not isinstance(self.lmcache_connector, MockLMCacheConnector)
        )

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{hit_rate:.2f}%",
            "connector_type": type(self.lmcache_connector).__name__ if self.lmcache_connector else "None",
            "connector_class_available": self.connector_class is not None,
            "real_connector": is_real_connector,
            "connector_module": self.lmcache_connector.__class__.__module__ if self.lmcache_connector else "None"
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