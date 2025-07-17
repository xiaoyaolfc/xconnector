import asyncio
import torch
from xconnector.connector import XConnector
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.forward_context import ForwardContext
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks


async def run_xconnector():
    # 创建 XConnector 实例
    connector = XConnector()

    # 模拟一些必要的输入数据
    model_executable = torch.nn.Linear(10, 10)  # 简单的线性模型作为示例
    model_input = ModelInputForGPUWithSamplingMetadata()
    kv_caches = [torch.randn(10, 10) for _ in range(2)]
    hidden_or_intermediate_states = torch.randn(10, 10)
    finished_req_ids = {1, 2, 3}
    context = ForwardContext()
    request = Request(id=1, prompt="Test prompt")
    blocks = KVCacheBlocks()
    num_external_tokens = 5

    # 启动 LMCache 的 KV 加载
    await connector.lmcache.start_load_kv(context)
    print("LMCache started loading KV.")

    # VLLM 接收 KV 缓存
    result = await connector.vllm.recv_kv_caches(model_executable, model_input, kv_caches)
    print("VLLM received KV caches:", result)

    # VLLM 发送 KV 缓存
    await connector.vllm.send_kv_caches(model_executable, model_input, kv_caches, hidden_or_intermediate_states)
    print("VLLM sent KV caches.")

    # 获取完成的请求 ID
    finished_result = await connector.vllm.get_finished(finished_req_ids)
    print("VLLM got finished requests:", finished_result)

    # LMCache 保存 KV 层
    from vllm.attention.backends.abstract import AttentionMetadata
    attn_metadata = AttentionMetadata()
    kv_layer = torch.randn(10, 10)
    await connector.lmcache.save_kv_layer("layer1", kv_layer, attn_metadata)
    print("LMCache saved KV layer.")


if __name__ == "__main__":
    asyncio.run(run_xconnector())