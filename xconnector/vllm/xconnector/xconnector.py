# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LMCache KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches.
"""

from typing import TYPE_CHECKING, Union, Optional, List
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)



# from dataclasses import dataclass
from typing import Optional, Union, Any, List
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# 导入必要的配置类（根据实际项目路径调整）
from vllm.config import ModelConfig, ParallelConfig, CacheConfig, KVTransferConfig  # 新增导入

# 模仿VllmConfig定义XXConfig
@dataclass(config=ConfigDict(arbitrary_types_allowed=True)) 
class XXConfig:
    """XXCache的配置类（参考VllmConfig结构）"""
    
    # 基础配置字段
    cache_config: Optional[CacheConfig] = None  # 明确类型
    parallel_config: Optional[ParallelConfig] = None  # 明确类型
    max_batch_size: int = 32
    enable_prefix_caching: bool = True
    
    # 添加XXCacheConnector需要的配置字段
    model_config: Optional[ModelConfig] = None  # 新增
    kv_transfer_config: Optional[KVTransferConfig] = None  # 新增
    
    # 哈希计算方法
    def compute_hash(self) -> str:
        """计算配置的哈希值，用于区分不同配置"""
        import hashlib
        factors: List[Any] = []
        
        factors.append(self.max_batch_size)
        factors.append(self.enable_prefix_caching)
        if self.cache_config:
            factors.append(self.cache_config.compute_hash())
        if self.parallel_config:
            factors.append(self.parallel_config.compute_hash())
        if self.model_config:
            factors.append(self.model_config.compute_hash())
        if self.kv_transfer_config:
            factors.append(self.kv_transfer_config.compute_hash())
        
        return hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
    
    # 初始化后验证逻辑
    def __post_init__(self):
        """验证配置合法性"""
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size必须为正数")


class XXCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: XXConfig,
    ):
        self.xx_config = config
        
        # 检查必要的配置字段
        if config.kv_transfer_config is None:
            logger.warning("XXConfig缺少kv_transfer_config，可能导致KV传输功能无法正常工作")
            
        self.transfer_config = config.kv_transfer_config
        
        # 导入LMCache相关模块
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.integration.vllm.vllm_adapter import (
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_retrieve, lmcache_should_store,
            lmcache_store_kv)
        
        logger.info("Initializing LMCacheConfig under kv_transfer_config %s",
                    self.transfer_config)

        # 检查关键配置
        if config.model_config is None:
            raise ValueError("XXConfig必须包含model_config")
        if config.parallel_config is None:
            raise ValueError("XXConfig必须包含parallel_config")
        if config.cache_config is None:
            raise ValueError("XXConfig必须包含cache_config")
            
        self.engine = init_lmcache_engine(
            config.model_config,
            config.parallel_config,
            config.cache_config
        )
        
        self.lmcache_engine_name = ENGINE_NAME
        self.lmcache_engine_builder = LMCacheEngineBuilder

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_retrieve = lmcache_should_retrieve
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        retrieve_status = self.lmcache_should_retrieve(model_input)
        model_input, bypass_model_exec, hidden_or_intermediate_states =\
            self.lmcache_retrieve_kv(
                model_executable, model_input, self.cache_config, kv_caches,
                retrieve_status)
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        store_status = self.lmcache_should_store(model_input)
        self.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
        self.lmcache_engine_builder.destroy(self.lmcache_engine_name)