'''
    该部分代码替换原ai-dynamo/examples/vllm_v0/worker.py
'''

# 在文件开头添加 XConnector imports
from xconnector.core.connector import XConnector, AdapterConfig, AdapterType
from xconnector.adapters.distributed.dynamo_adapter import WorkerStatus
# from xconnector.utils.xconnector_logging import get_logger
# 引用ai-dynamo库中的文件
import asyncio
import logging
import os
import signal
import uuid

from components.disagg_router import PyDisaggregatedRouter
from components.prefill_worker import PrefillWorker
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import PreprocessedRequest
from utils.vllm import RouterType, parse_vllm_args
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.llm import ModelType, WorkerMetricsPublisher, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

# 配置日志
logger = logging.getLogger(__name__)

# 在 VllmWorker 类中添加 XConnector 集成
@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker:
    prefill_worker = depends(PrefillWorker)

    def __init__(self):
        # ... 原有代码保持不变 ...
        self.client = None
        self.disaggregated_router: PyDisaggregatedRouter = None  # type: ignore
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self.namespace, _ = VllmWorker.dynamo_address()  # type: ignore
        self._prefill_queue_stream_name = f"{self.namespace}_prefill_queue"
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == RouterType.KV:
            if not self.engine_args.enable_prefix_caching:
                logger.info(
                    "When using KV router, prefix caching must be enabled, setting to True"
                )
                self.engine_args.enable_prefix_caching = True

            VLLM_WORKER_ID = dynamo_context["endpoints"][0].lease_id()
            os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
            os.environ["VLLM_KV_NAMESPACE"] = "dynamo"
            os.environ["VLLM_KV_COMPONENT"] = class_name

        self.metrics_publisher = WorkerMetricsPublisher()

        model_config = self.engine_args.create_model_config()
        self.default_sampling_params = model_config.get_diff_sampling_param()

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        # XConnector 集成
        self.xconnector = None
        self.xconnector_enabled = self.config.get("enable_xconnector", True)
        self.dynamo_adapter = None

        # 在初始化结束时启动 XConnector
        if self.xconnector_enabled:
            asyncio.create_task(self._init_xconnector())

    async def _init_xconnector(self):
        """Initialize XConnector integration"""
        try:
            logger.info("Initializing XConnector...")

            # 创建 XConnector 实例
            self.xconnector = XConnector()

            # 配置 Dynamo adapter
            dynamo_config = AdapterConfig(
                name="dynamo",
                type=AdapterType.DISTRIBUTED,
                class_path="xconnector.adapters.distributed.dynamo_adapter.DynamoAdapter",
                config={
                    "namespace": self.namespace,
                    "component_name": "vllm_worker",
                    "routing_policy": {
                        "strategy": "least_loaded",
                        "max_requests_per_worker": 100
                    }
                }
            )

            # 加载 adapter
            await self.xconnector.load_adapter(dynamo_config)

            # 启动 XConnector
            await self.xconnector.start()

            # 获取 Dynamo adapter
            self.dynamo_adapter = self.xconnector.get_adapter("dynamo", AdapterType.DISTRIBUTED)

            # 注册当前 worker
            await self._register_worker()

            logger.info("XConnector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize XConnector: {e}")
            self.xconnector_enabled = False

    async def _register_worker(self):
        """Register this worker with XConnector"""
        if not self.dynamo_adapter:
            return

        try:
            # 获取 worker 信息
            worker_id = str(dynamo_context["endpoints"][0].lease_id())

            # 获取 GPU 信息
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory

            # 构建 worker 信息
            worker_info = {
                "model": self.engine_args.model,
                "endpoint": f"{self.namespace}.vllm_worker.generate",
                "gpu_memory": gpu_memory,
                "metadata": {
                    "tensor_parallel_size": self.engine_args.tensor_parallel_size,
                    "pipeline_parallel_size": self.engine_args.pipeline_parallel_size,
                    "max_batch_size": self.max_batch_size,
                    "block_size": self.engine_args.block_size,
                    "enable_prefix_caching": self.engine_args.enable_prefix_caching,
                    "enable_chunked_prefill": self.engine_args.enable_chunked_prefill
                }
            }

            # 注册 worker
            await self.dynamo_adapter.register_worker(worker_id, worker_info)

            logger.info(f"Worker {worker_id} registered with XConnector")

        except Exception as e:
            logger.error(f"Failed to register worker: {e}")

    async def recv_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Any], bool, Any]:
        """
        Enhanced KV cache receiving with XConnector integration
        """
        # 如果 XConnector 启用且可用
        if self.xconnector_enabled and self.xconnector:
            try:
                # 尝试通过 XConnector 获取缓存
                result = await self._recv_kv_with_xconnector(
                    model_executable, model_input, kv_caches
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"XConnector KV receive failed: {e}")

        # 回退到原有逻辑
        return await self._recv_kv_native(model_executable, model_input, kv_caches)

    async def _recv_kv_with_xconnector(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor]
    ) -> Optional[Tuple[Union[torch.Tensor, Any], bool, Any]]:
        """Receive KV caches through XConnector"""
        try:
            # 路由到缓存 adapter
            cache_result = await self.xconnector.route_message(
                source="vllm",
                target="lmcache",
                method="retrieve_kv",
                model_input=model_input,
                kv_caches=kv_caches
            )

            if cache_result and cache_result.get("found"):
                # 更新 worker 状态
                if self.dynamo_adapter:
                    worker_id = str(dynamo_context["endpoints"][0].lease_id())
                    await self.dynamo_adapter.update_worker_status(
                        worker_id,
                        WorkerStatus.BUSY,
                        {"active_requests": len(self.active_requests)}
                    )

                # 返回缓存的结果
                return (
                    cache_result.get("hidden_states"),
                    cache_result.get("skip_forward", False),
                    cache_result.get("updated_input", model_input)
                )

        except Exception as e:
            logger.error(f"XConnector cache retrieval error: {e}")

        return None

    async def send_kv_caches(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_or_intermediate_states: Union[torch.Tensor, Any]
    ) -> None:
        """
        Enhanced KV cache sending with XConnector integration
        """
        # 如果 XConnector 启用
        if self.xconnector_enabled and self.xconnector:
            try:
                await self._send_kv_with_xconnector(
                    model_executable, model_input, kv_caches, hidden_or_intermediate_states
                )
            except Exception as e:
                logger.error(f"XConnector KV send failed: {e}")

        # 同时执行原有逻辑（如果需要）
        if self.engine_args.remote_prefill:
            await self._send_kv_native(
                model_executable, model_input, kv_caches, hidden_or_intermediate_states
            )

    async def _send_kv_with_xconnector(
            self,
            model_executable: torch.nn.Module,
            model_input: Any,
            kv_caches: List[torch.Tensor],
            hidden_states: Union[torch.Tensor, Any]
    ) -> None:
        """Send KV caches through XConnector"""
        try:
            # 路由到缓存 adapter
            await self.xconnector.route_message(
                source="vllm",
                target="lmcache",
                method="store_kv",
                model_input=model_input,
                kv_caches=kv_caches,
                hidden_states=hidden_states
            )

            # 通过 Dynamo adapter 协调缓存
            if self.dynamo_adapter:
                cache_key = self._generate_cache_key(model_input)
                worker_id = str(dynamo_context["endpoints"][0].lease_id())

                await self.dynamo_adapter.coordinate_cache_operation(
                    operation="put",
                    cache_key=cache_key,
                    worker_id=worker_id
                )

        except Exception as e:
            logger.error(f"XConnector cache storage error: {e}")

    async def create_metrics_publisher_endpoint(self):
        """Extended metrics publisher with XConnector metrics"""
        component = dynamo_context["component"]
        logger.info("Creating metrics publisher endpoint with primary lease")
        await self.metrics_publisher.create_endpoint(component)

        # Start XConnector metrics collection if enabled
        if self.xconnector_enabled:
            asyncio.create_task(self._publish_xconnector_metrics())

    async def _publish_xconnector_metrics(self):
        """Publish XConnector metrics to Dynamo"""
        while True:
            try:
                if self.xconnector:
                    # Get XConnector health status
                    health_status = await self.xconnector.get_health_status()

                    # Extract cache metrics
                    cache_metrics = health_status.get("adapters", {}).get("cache", {})
                    if "lmcache" in cache_metrics:
                        lmcache_stats = cache_metrics["lmcache"]

                        # Publish to Dynamo metrics system
                        if hasattr(lmcache_stats, "hit_rate"):
                            self.metrics_publisher.publish(
                                gpu_prefix_cache_hit_rate=float(
                                    lmcache_stats["hit_rate"].rstrip("%")
                                ) / 100.0
                            )

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error publishing XConnector metrics: {e}")
                await asyncio.sleep(60)

    @endpoint()
    async def generate(self, request: PreprocessedRequest):
        """Enhanced generate endpoint with XConnector routing"""
        request_id = str(uuid.uuid4())

        # Check if request should be routed through XConnector
        if self.xconnector_enabled and self.dynamo_adapter:
            # Let Dynamo adapter decide routing
            routing_decision = await self.dynamo_adapter.route_request({
                "model": self.engine_args.model,
                "request_id": request_id,
                "token_count": len(request.token_ids),
                "prefix_hit_blocks": request.estimated_prefix_hit_num_blocks
            })

            if routing_decision and routing_decision != dynamo_context["endpoints"][0].lease_id():
                # Request should be handled by different worker
                logger.info(f"Routing request {request_id} to worker {routing_decision}")
                # Note: Actual routing would be handled by Dynamo's frontend
                # This is just for logging/metrics

        # Continue with normal generation process...
        # [Rest of the generate method remains the same]

    async def graceful_shutdown(self, runtime):
        """Extended shutdown with XConnector cleanup"""
        logger.info("Received shutdown signal, shutting down DistributedRuntime")

        # Unregister from XConnector
        if self.xconnector_enabled and self.dynamo_adapter:
            try:
                worker_id = str(dynamo_context["endpoints"][0].lease_id())
                await self.dynamo_adapter.unregister_worker(worker_id)

                # Stop XConnector
                await self.xconnector.stop()

            except Exception as e:
                logger.error(f"Error during XConnector shutdown: {e}")

        # Continue with normal shutdown
        runtime.shutdown()
        logger.info("DistributedRuntime shutdown complete")

    def _generate_cache_key(self, model_input: Any) -> str:
        """Generate cache key from model input"""
        # Simple implementation - can be enhanced
        tokens = getattr(model_input, "input_tokens", [])
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        return f"cache_{hash(tuple(tokens[:32]))}"  # Use first 32 tokens for key