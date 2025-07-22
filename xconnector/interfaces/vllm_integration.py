# xconnector/adapters/inference/vllm_integration.py
"""
XConnector 与 vLLM 的集成适配器

这个模块提供了XConnector作为vLLM KV连接器的集成方案，
而不是重新实现vLLM的factory模式。
"""

from typing import TYPE_CHECKING, Union, Any, Dict, List
import torch

from xconnector.core.connector import XConnector, AdapterType
from xconnector.utils.xconnector_logging import get_logger

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.config import VllmConfig

logger = get_logger(__name__)


class XConnectorVLLMBridge:
    """
    XConnector 与 vLLM 的桥接器

    这个类实现了vLLM期望的KV连接器接口，
    但实际上将调用委托给XConnector的适配器系统。
    """

    def __init__(self, rank: int, local_rank: int, config: "VllmConfig"):
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = config

        # 初始化XConnector
        self.xconnector = XConnector()

        # 确保适配器已加载
        self._ensure_adapters_loaded()

        logger.info("XConnectorVLLMBridge initialized")

    def _ensure_adapters_loaded(self):
        """确保必要的适配器已加载"""
        # 检查vLLM适配器
        vllm_adapter = self.xconnector.get_adapter("vllm", AdapterType.INFERENCE)
        if not vllm_adapter:
            logger.warning("vLLM adapter not found, some features may not work")

        # 检查缓存适配器
        cache_adapter = self.xconnector.get_adapter("lmcache", AdapterType.CACHE)
        if not cache_adapter:
            logger.warning("Cache adapter not found, caching disabled")

        # 更新适配器配置
        if cache_adapter and hasattr(cache_adapter, 'update_cache_config'):
            vllm_config_dict = {
                'model_config': getattr(self.vllm_config, 'model_config', None),
                'parallel_config': getattr(self.vllm_config, 'parallel_config', None),
                'cache_config': getattr(self.vllm_config, 'cache_config', None),
            }
            cache_adapter.update_cache_config(vllm_config_dict)

    async def send_kv_caches_and_hidden_states(
            self,
            model_executable: torch.nn.Module,
            model_input: "ModelInputForGPUWithSamplingMetadata",
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, Any],
    ) -> None:
        """
        发送KV缓存和隐藏状态

        这个方法符合vLLM的接口规范，但实际调用XConnector的适配器
        """
        try:
            # 通过XConnector路由到缓存适配器
            await self.xconnector.route_message(
                source="vllm",
                target="lmcache",
                method="store_kv",
                model_input=model_input,
                kv_caches=kv_caches,
                hidden_states=hidden_or_intermediate_states
            )
        except Exception as e:
            logger.error(f"Failed to send KV caches: {e}")
            # 不抛出异常，避免影响vLLM主流程

    async def recv_kv_caches_and_hidden_states(
            self,
            model_executable: torch.nn.Module,
            model_input: "ModelInputForGPUWithSamplingMetadata",
            kv_caches: List[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, Any], bool, "ModelInputForGPUWithSamplingMetadata"]:
        """
        接收KV缓存和隐藏状态

        返回: (hidden_states, bypass_model_exec, updated_model_input)
        """
        try:
            # 通过XConnector路由到缓存适配器
            cache_result = await self.xconnector.route_message(
                source="vllm",
                target="lmcache",
                method="retrieve_kv",
                model_input=model_input,
                kv_caches=kv_caches
            )

            if cache_result and cache_result.get("found"):
                return (
                    cache_result.get("hidden_states"),
                    cache_result.get("skip_forward", False),
                    cache_result.get("updated_input", model_input)
                )
            else:
                # 缓存未命中，继续正常流程
                return None, False, model_input

        except Exception as e:
            logger.error(f"Failed to receive KV caches: {e}")
            # 出错时继续正常流程
            return None, False, model_input

    def close(self) -> None:
        """关闭连接器"""
        # XConnector的清理由其自身管理
        pass


# # 配置文件示例
# XCONNECTOR_VLLM_CONFIG = {
#     "adapters": [
#         {
#             "name": "vllm",
#             "type": "inference",
#             "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
#             "config": {
#                 "model_name": "your_model",
#                 "tensor_parallel_size": 1,
#                 "enable_prefix_caching": True
#             },
#             "enabled": True
#         },
#         {
#             "name": "lmcache",
#             "type": "cache",
#             "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
#             "config": {
#                 "storage_backend": "local",
#                 "max_cache_size": 1024,
#                 "enable_compression": True
#             },
#             "enabled": True
#         }
#     ]
# }
#
#
# # 在vLLM配置中注册XConnector连接器
# def register_xconnector_with_vllm():
#     """
#     将XConnector注册为vLLM的KV连接器
#
#     使用方法：
#     1. 在vLLM启动前调用这个函数
#     2. 在vLLM配置中设置 kv_connector="XConnector"
#     """
#     try:
#         from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
#
#         # 注册XConnector桥接器
#         KVConnectorFactory.register_connector(
#             "XConnector",
#             "xconnector.adapters.inference.vllm_integration",
#             "XConnectorVLLMBridge"
#         )
#
#         logger.info("XConnector registered with vLLM successfully")
#         return True
#
#     except ImportError:
#         logger.error("vLLM not available, cannot register XConnector")
#         return False
#     except Exception as e:
#         logger.error(f"Failed to register XConnector with vLLM: {e}")
#         return False
#
#
# # 使用示例
# async def example_usage():
#     """XConnector与vLLM集成的使用示例"""
#
#     # 1. 注册XConnector到vLLM
#     register_xconnector_with_vllm()
#
#     # 2. 配置vLLM使用XConnector
#     from vllm.config import VllmConfig
#     from vllm.distributed.kv_transfer.kv_transfer_config import KVTransferConfig
#
#     kv_transfer_config = KVTransferConfig(
#         kv_connector="XConnector",
#         is_kv_transfer_instance=True
#     )
#
#     # 3. vLLM会自动使用XConnector进行KV缓存管理
#     # 无需额外代码，vLLM的推理过程会自动调用XConnector的缓存功能
#
#
# if __name__ == "__main__":
#     # 测试注册
#     success = register_xconnector_with_vllm()
#     if success:
#         print("✅ XConnector successfully integrated with vLLM")
#     else:
#         print("❌ Failed to integrate XConnector with vLLM")