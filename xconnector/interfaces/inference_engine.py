# xconnector/interfaces/inference_engine.py
"""
推理引擎通用接口

统一的推理引擎适配器接口，支持 vLLM、TGI、LightLLM 等多种推理框架

暂时不需要，后续接入mooncake和sglang的时候会用到
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import torch

from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class InferenceStatus(Enum):
    """推理状态"""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    prompt: Union[str, List[int]]  # 文本或token IDs
    params: Dict[str, Any]  # 采样参数
    priority: int = 0
    arrival_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    outputs: List[str]  # 生成的文本
    token_ids: List[List[int]]  # 生成的token IDs
    finished: bool
    finish_reason: Optional[str] = None
    usage_stats: Optional[Dict[str, Any]] = None


class InferenceEngineInterface(ABC):
    """推理引擎通用接口"""

    # === 基础生命周期管理 ===

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化推理引擎"""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """启动推理引擎"""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """停止推理引擎"""
        pass

    # === 请求管理 ===

    @abstractmethod
    async def add_request(self, request: InferenceRequest) -> bool:
        """添加推理请求"""
        pass

    @abstractmethod
    async def abort_request(self, request_id: str) -> bool:
        """取消推理请求"""
        pass

    @abstractmethod
    def get_request_status(self, request_id: str) -> Optional[InferenceStatus]:
        """获取请求状态"""
        pass

    # === 推理执行 ===

    @abstractmethod
    async def step(self) -> List[InferenceResponse]:
        """执行一步推理，返回已完成的响应"""
        pass

    # === 状态查询 ===

    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        """是否有未完成的请求"""
        pass

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """获取未完成请求数量"""
        pass

    # === 模型管理 ===

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        pass

    # === KV 缓存接口（可选） ===

    async def recv_kv_caches(
            self,
            model_executable: Optional[torch.nn.Module],
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """
        接收并处理 KV 缓存（可选实现）

        Returns:
            Tuple of (hidden_states, skip_forward, updated_input)
        """
        # 默认不处理 KV 缓存
        return None, False, model_input

    async def send_kv_caches(
            self,
            model_executable: Optional[torch.nn.Module],
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any]
    ) -> None:
        """发送 KV 缓存到存储系统（可选实现）"""
        # 默认不处理 KV 缓存
        pass

    async def get_finished(
            self,
            finished_req_ids: set
    ) -> Tuple[Optional[set], Optional[set]]:
        """
        处理完成的请求（可选实现）

        Returns:
            Tuple of (finished_ids, cancelled_ids)
        """
        return finished_req_ids, None


# === vLLM 专用接口 ===

class VLLMEngineInterface(InferenceEngineInterface):
    """
    vLLM 推理引擎接口

    基于 vLLM 的 LLMEngine 实现
    """

    def __init__(self, engine_config: Dict[str, Any]):
        self.config = engine_config
        self.engine = None  # 将在 initialize 中创建
        self.request_counter = 0

    async def initialize(self) -> bool:
        """初始化 vLLM 引擎"""
        try:
            from vllm import LLMEngine, EngineArgs

            # 创建引擎参数
            engine_args = EngineArgs(**self.config)

            # 创建引擎实例
            self.engine = LLMEngine.from_engine_args(engine_args)

            logger.info("vLLM engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            return False

    async def start(self) -> bool:
        """启动 vLLM 引擎"""
        if self.engine is None:
            return await self.initialize()
        return True

    async def stop(self) -> bool:
        """停止 vLLM 引擎"""
        # vLLM 引擎会在析构时自动清理
        self.engine = None
        return True

    async def add_request(self, request: InferenceRequest) -> bool:
        """添加推理请求到 vLLM"""
        if self.engine is None:
            return False

        try:
            from vllm import SamplingParams

            # 转换采样参数
            sampling_params = SamplingParams(**request.params)

            # 添加请求
            self.engine.add_request(
                request_id=request.request_id,
                prompt=request.prompt,
                params=sampling_params,
                arrival_time=request.arrival_time,
                priority=request.priority
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add request {request.request_id}: {e}")
            return False

    async def abort_request(self, request_id: str) -> bool:
        """取消 vLLM 请求"""
        if self.engine is None:
            return False

        try:
            self.engine.abort_request(request_id)
            return True
        except Exception as e:
            logger.error(f"Failed to abort request {request_id}: {e}")
            return False

    def get_request_status(self, request_id: str) -> Optional[InferenceStatus]:
        """获取请求状态（vLLM 不直接支持，简化实现）"""
        if self.engine is None:
            return None

        # vLLM 不提供单独的状态查询，简化为检查是否有未完成请求
        if self.engine.has_unfinished_requests():
            return InferenceStatus.RUNNING
        else:
            return InferenceStatus.FINISHED

    async def step(self) -> List[InferenceResponse]:
        """执行一步 vLLM 推理"""
        if self.engine is None:
            return []

        try:
            # 执行一步推理
            request_outputs = self.engine.step()

            # 转换为统一响应格式
            responses = []
            for output in request_outputs:
                response = InferenceResponse(
                    request_id=output.request_id,
                    outputs=[o.text for o in output.outputs],
                    token_ids=[o.token_ids for o in output.outputs],
                    finished=output.finished,
                    finish_reason=output.outputs[0].finish_reason if output.outputs else None,
                    usage_stats={
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": sum(len(o.token_ids) for o in output.outputs),
                        "total_tokens": len(output.prompt_token_ids) + sum(len(o.token_ids) for o in output.outputs)
                    }
                )
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"vLLM step failed: {e}")
            return []

    def has_unfinished_requests(self) -> bool:
        """是否有未完成的请求"""
        if self.engine is None:
            return False
        return self.engine.has_unfinished_requests()

    def get_num_unfinished_requests(self) -> int:
        """获取未完成请求数量"""
        if self.engine is None:
            return 0
        return self.engine.get_num_unfinished_requests()

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        if self.engine is None:
            return {}

        model_config = self.engine.get_model_config()
        return {
            "model": model_config.model,
            "tokenizer": model_config.tokenizer,
            "max_model_len": model_config.max_model_len,
            "dtype": str(model_config.dtype),
        }

    # === vLLM KV 缓存特定实现 ===

    async def recv_kv_caches(
            self,
            model_executable: Optional[torch.nn.Module],
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """vLLM KV 缓存接收实现"""
        # 这里应该调用 XConnector 的缓存适配器
        # 简化实现，实际应该通过 route_message 调用
        return None, False, model_input

    async def send_kv_caches(
            self,
            model_executable: Optional[torch.nn.Module],
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any]
    ) -> None:
        """vLLM KV 缓存发送实现"""
        # 这里应该调用 XConnector 的缓存适配器
        # 简化实现，实际应该通过 route_message 调用
        pass


# === TGI 接口（示例） ===

class TGIEngineInterface(InferenceEngineInterface):
    """
    Text Generation Inference (TGI) 接口

    简化的 TGI 集成示例
    """

    def __init__(self, engine_config: Dict[str, Any]):
        self.config = engine_config
        self.client = None
        self.pending_requests: Dict[str, InferenceRequest] = {}

    async def initialize(self) -> bool:
        """初始化 TGI 客户端"""
        try:
            # 这里应该初始化 TGI 客户端
            # from text_generation import AsyncClient
            # self.client = AsyncClient(self.config.get("endpoint"))

            logger.info("TGI client initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TGI client: {e}")
            return False

    async def start(self) -> bool:
        return await self.initialize()

    async def stop(self) -> bool:
        self.client = None
        self.pending_requests.clear()
        return True

    async def add_request(self, request: InferenceRequest) -> bool:
        """添加请求到 TGI"""
        if self.client is None:
            return False

        # 存储待处理请求
        self.pending_requests[request.request_id] = request
        return True

    async def abort_request(self, request_id: str) -> bool:
        """取消 TGI 请求"""
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
        return True

    def get_request_status(self, request_id: str) -> Optional[InferenceStatus]:
        """获取请求状态"""
        if request_id in self.pending_requests:
            return InferenceStatus.WAITING
        return InferenceStatus.FINISHED

    async def step(self) -> List[InferenceResponse]:
        """执行一步 TGI 推理"""
        # 简化实现 - 实际应该调用 TGI 的生成接口
        responses = []

        for req_id, request in list(self.pending_requests.items()):
            try:
                # 模拟 TGI 调用
                # response = await self.client.generate(request.prompt, **request.params)

                # 创建响应（模拟）
                response = InferenceResponse(
                    request_id=req_id,
                    outputs=["Generated text"],  # 模拟输出
                    token_ids=[[1, 2, 3, 4]],  # 模拟token IDs
                    finished=True,
                    finish_reason="stop"
                )

                responses.append(response)
                del self.pending_requests[req_id]

            except Exception as e:
                logger.error(f"TGI generation failed for {req_id}: {e}")

        return responses

    def has_unfinished_requests(self) -> bool:
        return len(self.pending_requests) > 0

    def get_num_unfinished_requests(self) -> int:
        return len(self.pending_requests)

    def get_model_config(self) -> Dict[str, Any]:
        return self.config.copy()


# === 工厂函数 ===

def create_inference_engine(
        engine_type: str,
        config: Dict[str, Any]
) -> InferenceEngineInterface:
    """
    创建推理引擎接口

    Args:
        engine_type: 引擎类型 ("vllm", "tgi", "lightllm")
        config: 引擎配置

    Returns:
        InferenceEngineInterface: 推理引擎实例
    """
    engine_type = engine_type.lower()

    if engine_type == "vllm":
        return VLLMEngineInterface(config)
    elif engine_type == "tgi":
        return TGIEngineInterface(config)
    else:
        raise ValueError(f"Unsupported inference engine type: {engine_type}")


# === 辅助函数 ===

def convert_vllm_request_output(vllm_output) -> InferenceResponse:
    """将 vLLM RequestOutput 转换为统一格式"""
    return InferenceResponse(
        request_id=vllm_output.request_id,
        outputs=[output.text for output in vllm_output.outputs],
        token_ids=[output.token_ids for output in vllm_output.outputs],
        finished=vllm_output.finished,
        finish_reason=vllm_output.outputs[0].finish_reason if vllm_output.outputs else None,
        usage_stats={
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": sum(len(output.token_ids) for output in vllm_output.outputs)
        }
    )