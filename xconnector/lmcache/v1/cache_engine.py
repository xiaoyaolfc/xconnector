# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Dict, Generator, List, Optional, Union
import asyncio
import multiprocessing
import time

# Third Party
import torch

# First Party
from xconnector.lmcache.config import LMCacheEngineMetadata
from xconnector.lmcache.logging import init_logger
from xconnector.lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from xconnector.lmcache.usage_context import InitializeUsageContext
from xconnector.lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from xconnector.lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed_server import (
    DistributedServerInterface,
    NaiveDistributedServer,
)
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    # VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.lookup_server import LookupServerInterface, RedisLookupServer
from lmcache.v1.memory_management import AdHocMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
)
from lmcache.v1.storage_backend.storage_manager import (
    DistributedStorageManager,
    StorageManager,
)
from xconnector.lmcache.v1.token_database import ChunkedTokenDatabase, TokenDatabase

logger = init_logger(__name__)


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
        layerwise: bool = False,
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.memory_allocator = memory_allocator
        self.token_database = token_database
        self.gpu_connector = gpu_connector

        self.enable_p2p = config.enable_p2p

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        self.lookup_server: Optional[LookupServerInterface] = None
        if self.enable_p2p:
            self.lookup_server = RedisLookupServer(config)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.config.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.use_distributed_storage_manager = False
        if config.enable_nixl:
            self.use_distributed_storage_manager = True
            self.storage_manager = DistributedStorageManager(
                config, metadata, self.memory_allocator
            )
        else:
            self.storage_manager = StorageManager(
                config,
                metadata,
                self.memory_allocator,
                self.lmcache_worker,
                self.lookup_server,
                layerwise,
            )  # type: ignore[assignment]

        # if self.enable_p2p:
        #     self.distributed_loop = asyncio.get_event_loop()
        #     assert self.lookup_server is not None
        #     assert isinstance(self.storage_manager, StorageManager)
        #     self.distributed_server: DistributedServerInterface = (
        #         NaiveDistributedServer(
        #             self.storage_manager,
        #             self.lookup_server,
        #             self.distributed_loop,
        #             config,
        #         )
        #     )

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_distributed(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens and mask into the cache engine.

        This function is only for distributed storage manager.

        This function will be refactored in the future.
        """
        st = time.perf_counter()
        if mask is not None:
            num_store_tokens = torch.sum(mask)
        else:
            num_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_store_tokens)

        # Register the put request
        keys = []
        metadatas = []
        steds = []
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memobj_meta = self.storage_manager.dry_allocate(kv_shape, kv_dtype)
            assert memobj_meta is not None
            keys.append(key)
            metadatas.append(memobj_meta)
            steds.append((start, end))

        self.storage_manager.prepare_put(keys, metadatas)

        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        # Offload the KV cache and write to remote
        for key, memobj_meta, (start, end) in zip(keys, metadatas, steds, strict=False):
            assert memobj_meta.dtype is not None
            kv_shape = memobj_meta.shape
            kv_dtype = memobj_meta.dtype

            # Allocate for a zero-copy buffer, trigger send if needed
            t = time.perf_counter()
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            put_time += time.perf_counter() - t
            if memory_obj is None:
                logger.warning(
                    "Failed to allocate memory for the KV cache.\n"
                    "The KV cache will not be stored."
                )
                break

            # Copy the KV cache to the zero-copy buffer
            t = time.perf_counter()
            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)
            offload_time += time.perf_counter() - t

            tot_kv_size += memory_obj.get_size()

        # Flush
        t = time.perf_counter()
        self.storage_manager.commit_put()
        put_time += time.perf_counter() - t
        ed = time.perf_counter()

        logger.info(
            "Store %d tokens takes: %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            num_store_tokens,
            (ed - st) * 1000,
            tot_kv_size / (ed - st) / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # FIXME(ApostaC): A HACK for distributed storage manager
        if self.use_distributed_storage_manager:
            self.store_distributed(tokens, mask, **kwargs)
            return

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)

        for start, end, key in self.token_database.process_tokens(tokens, mask):
            assert isinstance(key, CacheEngineKey)
            if self.storage_manager.contains(key):
                continue
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            if memory_obj is None:
                logger.warning(
                    "Failed to allocate memory for the KV cache.\n"
                    "The KV cache will not be stored."
                )
                break

            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)
            self.storage_manager.put(key, memory_obj)

            # Update lookup server
            if self.lookup_server is not None:
                self.lookup_server.insert(key)

        self.stats_monitor.on_store_finished(monitor_req_id)

        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            assert isinstance(key, CacheEngineKey)

            # Get the memory object from the storage backend
            memory_obj = self.storage_manager.get(key)

            if memory_obj is None:
                if self.enable_p2p:
                    future_memory_obj = asyncio.run_coroutine_threadsafe(
                        self.distributed_server.issue_get(key),
                        self.distributed_loop,
                    )
                    memory_obj = future_memory_obj.result()
                if memory_obj is None:
                    break

            ret_mask[start:end] = True

            # NOTE(Jiayi): memory_obj doesn't have to be a pinned
            # cpu tensor for the sake of performance.
            # For example, disk->gpu is faster than disk->cpu->gpu.
            # RDMA is another example.
            self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)
            memory_obj.ref_count_down()

            # NOTE (ApostaC): This is only for the current implementation:
            # When the object is retrieved back to vLLM, the storage backend
            # will immediately remove the object from itself
            if isinstance(self.storage_manager, DistributedStorageManager):
                self.storage_manager.remove(key)

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.debug(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )
        return ret_mask

    def prefetch(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the prefetching process in the storage manager to load the
        KV to the local CPU memory
        """
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            assert isinstance(key, CacheEngineKey)
            self.storage_manager.prefetch(key)

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: If True, pin the KV cache in the storage.

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        search_local = True  # we always lookup local storage_manager first
        # secondary lookup on p2p (via lookup_server) if enabled
        search_p2p = self.enable_p2p and (search_range is None or "p2p" in search_range)

        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            if search_local:
                if self.storage_manager.contains(key, search_range, pin):
                    # found in storage manager, no need to search p2p
                    continue
                else:
                    # key not found in storage_manager
                    # search only p2p from now on
                    search_local = False
            if search_p2p:
                assert self.lookup_server is not None
                if self.lookup_server.lookup(key):
                    # found in p2p
                    # continue loop to ensure a maximal prefix match
                    continue
            # not found in both storage_manager and p2p,
            # return start, which equals last iteration's end
            return start

        # all tokens where found, return the maximal end
        return end

    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
    ) -> int:
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        if self.enable_p2p:
            self.distributed_server.close()

        if self.lmcache_worker is not None:
            self.lmcache_worker.close()

        self.storage_manager.close()
        logger.info("LMCacheEngine closed.")

#
# # TODO(Jiayi): Using a separate class here.
# # Should use the same class once the code is stable.
# class LayerwiseLMCacheEngine(LMCacheEngine):
#     """A specialized LMCacheEngine for layerwise cache engine.
#
#     This class is used to store the layerwise cache engine. It is a
#     subclass of LMCacheEngine and inherits all the methods and attributes
#     from it. However, it retrieves the KV cache in a layerwise manner
#     instead of chunkwise manner.
#     """
#
#     def __init__(
#         self,
#         config: LMCacheEngineConfig,
#         metadata: LMCacheEngineMetadata,
#         memory_allocator: MemoryAllocatorInterface,
#         token_database: TokenDatabase,
#         layerwise_gpu_connector: GPUConnectorInterface,
#         layerwise: bool = True,
#     ):
#         super().__init__(
#             config,
#             metadata,
#             memory_allocator,
#             token_database,
#             layerwise_gpu_connector,
#             layerwise,
#         )
#         assert isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector)
#
#         self.num_layers = metadata.kv_shape[0]
#
#     @_lmcache_nvtx_annotate
#     @torch.inference_mode()
#     def store_layer(
#         self,
#         tokens: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Generator[None, None, None]:
#         """
#         Store the KV cache in a layerwise manner.
#
#         :param torch.Tensor tokens: The tokens of the corresponding KV caches.
#
#         :param Optional[torch.Tensor] mask: The mask for the tokens. Should
#             have the same length as tokens. And the mask should ALWAYS be like
#             FFFFFTTTTTTT, where True means the tokens needs to be matched.
#
#         :param **kwargs: The additional arguments for the storage backend which
#             will be passed into the gpu_connector.
#
#         return: A generator that yields None. In the first iteration, the
#             generator allocates the memory objects for all layers and moves
#             the KV cache of the first layer from GPU to CPU. In the next
#             iterations, it moves the KV cache of layer i from GPU to the memory
#             objects (on CPU) and puts the memory objects of layer i-1 to the
#             storage backends. In the last iteration, it puts the memory objects
#             of the last layer to the storage backends.
#         """
#
#         if mask is not None:
#             num_stored_tokens = torch.sum(mask).item()
#         else:
#             num_stored_tokens = len(tokens)
#         monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)
#
#         starts = []
#         ends = []
#         keys = []
#         memory_objs = []
#         kv_dtype = self.metadata.kv_dtype
#         for start, end, key in self.token_database.process_tokens(tokens, mask):
#             assert isinstance(key, CacheEngineKey)
#
#             keys_multi_layer = key.split_layers(self.num_layers)
#
#             # Only check the first layer
#             if self.storage_manager.contains(keys_multi_layer[0]):
#                 continue
#
#             # Allocate the memory object
#             num_tokens = end - start
#             kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)
#
#             # TODO(Jiayi): Optimize with batched allocation
#             memory_objs_multi_layer: List[MemoryObj] = []
#             no_space_left = False
#             for layer_id in range(self.num_layers):
#                 mem_obj_single_layer = self.storage_manager.allocate(
#                     kv_shape_single_layer, kv_dtype, fmt=MemoryFormat.KV_T2D
#                 )
#
#                 if mem_obj_single_layer is None:
#                     logger.warning(
#                         "Failed to allocate memory for the KV cache.\n"
#                         "The KV cache will not be stored."
#                     )
#                     no_space_left = True
#                     for mem_obj_prev_layer in memory_objs_multi_layer:
#                         mem_obj_prev_layer.ref_count_down()
#                     break
#
#                 memory_objs_multi_layer.append(mem_obj_single_layer)
#
#             if no_space_left:
#                 break
#
#             starts.append(start)
#             ends.append(end)
#             keys.append(keys_multi_layer)
#             memory_objs.append(memory_objs_multi_layer)
#
#             # Update lookup server
#             if self.lookup_server is not None:
#                 self.lookup_server.batched_insert(keys_multi_layer)
#
#         if keys:
#             # Transpose the keys and memory objects into layer major format
#             memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
#             keys = [list(row) for row in zip(*keys, strict=False)]
#
#             assert isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector)
#             mem_obj_generator = self.gpu_connector.batched_from_gpu(
#                 memory_objs, starts, ends, **kwargs
#             )
#
#             next(mem_obj_generator)
#
#             for layer_id in range(self.num_layers):
#                 yield
#                 next(mem_obj_generator)
#                 self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
#         else:
#             # If no cache are found, we still need to yield to avoid
#             # `StopIteration`
#             for layer_id in range(self.num_layers):
#                 yield
#
#         self.stats_monitor.on_store_finished(monitor_req_id)
#         logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")
#         yield
#
#     @_lmcache_nvtx_annotate
#     @torch.inference_mode()
#     def retrieve_layer(
#         self,
#         tokens: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Generator[Optional[torch.Tensor], None, None]:
#         """
#         Retrieve the KV cache in a layerwise manner.
#
#         :param torch.Tensor tokens: The tokens of the corresponding KV caches.
#
#         :param Optional[torch.Tensor] mask: The mask for the tokens. Should
#             have the same length as tokens. And the mask should ALWAYS be like
#             FFFFFTTTTTTT, where True means the tokens needs to be matched.
#
#         :param **kwargs: The additional arguments for the storage backend which
#             will be passed into the gpu_connector.
#
#         return: A generator that yields Optional[torch.Tensor]. The tensor will
#             be the boolean mask indicating which tokens are retrieved and will
#             only be returned in the last iteration. In the first iteration,
#             the generator retrieve the memory objects of the first layer from
#             the storage backends. In the next iterations, it moves the KV cache
#             of layer i from the memory objects (on CPU) to GPU and retrieves
#             the memory objects of layer i+1 from the storage backends. In the
#             last iteration, it moves the memory objects of the last layer to
#             the GPU.
#         """
#
#         if mask is not None:
#             num_required_tokens = torch.sum(mask).item()
#         else:
#             num_required_tokens = len(tokens)
#         monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)
#
#         ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
#
#         starts = []
#         ends = []
#         keys = []
#         for start, end, key in self.token_database.process_tokens(tokens, mask):
#             assert isinstance(key, CacheEngineKey)
#
#             keys_multi_layer = key.split_layers(self.num_layers)
#
#             # NOTE: Only check the first layer
#             if not self.storage_manager.contains(keys_multi_layer[0]):
#                 break
#
#             starts.append(start)
#             ends.append(end)
#             keys.append(keys_multi_layer)
#
#             ret_mask[start:end] = True
#
#         if keys:
#             # Transpose the keys into layer major format
#             keys_layer_major = [list(row) for row in zip(*keys, strict=False)]
#
#             get_generator = self.storage_manager.layerwise_batched_get(keys_layer_major)
#
#             assert isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector)
#             mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
#             next(mem_obj_consumer)
#
#             to_count_down = []
#             for layer_id in range(self.num_layers):
#                 tasks = next(get_generator)
#
#                 assert None not in tasks
#
#                 yield None
#
#                 mem_objs_layer = [task.result() for task in tasks]
#                 mem_obj_consumer.send(mem_objs_layer)
#                 to_count_down.extend(mem_objs_layer)
#
#             # TODO(Jiayi): Need to be done in a modular way
#             for keys_layer in keys_layer_major:
#                 self.storage_manager.batched_unpin(keys_layer)
#
#             for mem_obj in to_count_down:
#                 mem_obj.ref_count_down()
#         else:
#             # If no cache are found, we still need to yield to avoid
#             # `StopIteration`
#             for layer_id in range(self.num_layers):
#                 yield None
#
#         yield None
#
#         # synchronize the last layer
#         next(mem_obj_consumer)
#
#         retrieved_tokens = torch.sum(ret_mask)
#         self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
#         logger.debug(
#             f"Retrieved {retrieved_tokens} "
#             f"out of {num_required_tokens} "
#             f"out of total {len(tokens)} tokens"
#         )
#
#         yield ret_mask
#
#     def lookup(
#         self,
#         tokens: Union[torch.Tensor, List[int]],
#         search_range: Optional[List[str]] = None,
#         pin: bool = False,
#     ) -> int:
#         """
#         Checks the existence of KV cache of the tokens from the cache engine.
#
#         :param tokens: the input tokens, with shape [seq_len]
#
#         :param Optional[List[str]] search_range: The range of storage backends
#         to search in. Should be a subset of
#         ["LocalCPUBackend", "LocalDiskBackend"] for now.
#         If None, search in all backends.
#
#         :param bool pin: If True, pin the KV cache in the storage.
#
#         :return: An int indicating how many prefix tokens are cached.
#         """
#         end = 0
#         for start, end, key in self.token_database.process_tokens(tokens):
#             assert isinstance(key, CacheEngineKey)
#
#             # TODO(Jiayi): Optimize by checking only the existence of the key
#             # of one layer
#             key_all_layers = key.split_layers(self.num_layers)
#             for key_single_layer in key_all_layers:
#                 if not self.storage_manager.contains(
#                     key_single_layer, search_range, pin
#                 ):
#                     return start
#         return end


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> MemoryAllocatorInterface:
        if config.enable_nixl:
            assert config.nixl_buffer_device is not None
            return AdHocMemoryAllocator(config.nixl_buffer_device)

        max_local_cpu_size = config.max_local_cpu_size
        return MixedMemoryAllocator(int(max_local_cpu_size * 1024**3))

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: GPUConnectorInterface,
        use_layerwise_engine: bool = False,
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            memory_allocator = cls._Create_memory_allocator(config, metadata)
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            # HACK(Jiayi): Merge two types of engine into one in the future
            engine: Union[LayerwiseLMCacheEngine, LMCacheEngine]
            if use_layerwise_engine:
                engine = LayerwiseLMCacheEngine(
                    config,
                    metadata,
                    memory_allocator,
                    token_database,
                    gpu_connector,
                )
            else:
                engine = LMCacheEngine(
                    config,
                    metadata,
                    memory_allocator,
                    token_database,
                    gpu_connector,
                )
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
