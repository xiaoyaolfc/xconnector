# xconnector/interfaces.py
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch
from abc import ABC, abstractmethod

from xconnector.utils.xconnector_logging import get_logger

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = get_logger(__name__)


class InferenceEngineInterface(ABC):
    """推理引擎通用接口"""

    @abstractmethod
    async def recv_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """接收并处理 KV 缓存"""
        pass

    @abstractmethod
    async def send_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, Any]
    ) -> None:
        """发送 KV 缓存到存储系统"""
        pass

    @abstractmethod
    async def get_finished(
            self,
            finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        """处理完成的请求"""
        pass


class CacheManagerInterface(ABC):
    """缓存管理通用接口"""

    @abstractmethod
    async def retrieve_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """检索 KV 缓存"""
        pass

    @abstractmethod
    async def store_kv(
            self,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any]
    ) -> Dict[str, Any]:
        """存储 KV 缓存"""
        pass

    @abstractmethod
    async def cleanup_finished(self, request_ids: set) -> None:
        """清理完成请求的缓存"""
        pass


# === vLLM 特定接口实现 ===

class VLLMInterface(InferenceEngineInterface):
    """vLLM 推理引擎接口"""

    # 继承基类的抽象方法，具体实现在 vllm_adapter.py 中
    pass


class LMCacheInterface(CacheManagerInterface):
    """LMCache 缓存管理接口（扩展版）"""

    async def start_load_kv(self, context: "ForwardContext", **kwargs) -> None:
        """开始加载 KV（LMCache 特有）"""
        pass

    async def wait_for_layer_load(self, layer_name: str) -> None:
        """等待层加载完成（LMCache 特有）"""
        pass

    async def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        """保存 KV 层（LMCache 特有）"""
        pass

    async def wait_for_save(self) -> None:
        """等待保存完成（LMCache 特有）"""
        pass

    async def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """获取新匹配的令牌数量（LMCache 特有）"""
        pass

    async def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """分配后更新状态（LMCache 特有）"""
        pass


# === vLLM 桥接器（简化版）===

class XConnectorVLLMBridge:
    """
    XConnector 与 vLLM 的桥接器（简化版）

    实现 vLLM 期望的 KV 连接器接口，委托给 XConnector
    """

    def __init__(self, rank: int, local_rank: int, config: "VllmConfig"):
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = config
        self.xconnector = None

        # 延迟初始化 XConnector
        self._init_xconnector()

        logger.info(f"XConnectorVLLMBridge initialized (rank={rank})")

    def _init_xconnector(self):
        """初始化 XConnector"""
        try:
            from xconnector.core.connector import get_connector
            self.xconnector = get_connector()

            # 更新缓存适配器配置
            cache_adapter = self.xconnector.get_adapter("lmcache",
                                                        self.xconnector.AdapterType.CACHE)
            if cache_adapter and hasattr(cache_adapter, 'update_cache_config'):
                vllm_config_dict = {
                    'model_config': getattr(self.vllm_config, 'model_config', None),
                    'parallel_config': getattr(self.vllm_config, 'parallel_config', None),
                    'cache_config': getattr(self.vllm_config, 'cache_config', None),
                }
                cache_adapter.update_cache_config(vllm_config_dict)

        except Exception as e:
            logger.error(f"Failed to initialize XConnector: {e}")

    async def send_kv_caches_and_hidden_states(
            self,
            model_executable: torch.nn.Module,
            model_input: "ModelInputForGPUWithSamplingMetadata",
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, Any],
    ) -> None:
        """发送 KV 缓存（vLLM 接口）"""
        if not self.xconnector:
            return

        try:
            await self.xconnector.route_message(
                source="vllm",
                target="lmcache",
                method="store_kv",
                model_input=model_input,
                kv_caches=kv_caches,
                hidden_states=hidden_or_intermediate_states
            )
        except Exception as e:
            logger.debug(f"KV cache store failed: {e}")  # 降级为 debug，避免影响主流程

    async def recv_kv_caches_and_hidden_states(
            self,
            model_executable: torch.nn.Module,
            model_input: "ModelInputForGPUWithSamplingMetadata",
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, "ModelInputForGPUWithSamplingMetadata"]:
        """接收 KV 缓存（vLLM 接口）"""
        if not self.xconnector:
            return None, False, model_input

        try:
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
        except Exception as e:
            logger.debug(f"KV cache retrieve failed: {e}")

        return None, False, model_input

    def close(self) -> None:
        """关闭连接器"""
        pass  # XConnector 生命周期由其自身管理


# === 注册函数（简化版）===

def register_xconnector_with_vllm() -> bool:
    """
    将 XConnector 注册为 vLLM 的 KV 连接器

    Returns:
        bool: 注册是否成功
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        KVConnectorFactory.register_connector(
            "XConnector",
            "xconnector.interfaces.inference_interface",
            "XConnectorVLLMBridge"
        )

        logger.info("XConnector registered with vLLM successfully")
        return True

    except ImportError:
        logger.warning("vLLM not available, cannot register XConnector")
        return False
    except Exception as e:
        logger.error(f"Failed to register XConnector with vLLM: {e}")
        return False


# === 工厂函数 ===

def create_inference_interface(engine_type: str) -> InferenceEngineInterface:
    """创建推理引擎接口"""
    if engine_type.lower() == "vllm":
        return VLLMInterface()
    else:
        raise ValueError(f"Unsupported inference engine: {engine_type}")


def create_cache_interface(cache_type: str) -> CacheManagerInterface:
    """创建缓存管理接口"""
    if cache_type.lower() == "lmcache":
        return LMCacheInterface()
    else:
        raise ValueError(f"Unsupported cache manager: {cache_type}")