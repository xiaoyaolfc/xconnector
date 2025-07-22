# 未来可拓展至mooncake和sglang，当前暂时不使用
调用该代码示例：
```python
# 创建 vLLM 引擎
config = {
    "model": "facebook/opt-125m",
    "tensor_parallel_size": 1
}
engine = create_inference_engine("vllm", config)

# 初始化和启动
await engine.initialize()
await engine.start()

# 添加请求
request = InferenceRequest(
    request_id="req_1",
    prompt="What is AI?",
    params={"temperature": 0.8, "max_tokens": 100}
)
await engine.add_request(request)

# 执行推理
while engine.has_unfinished_requests():
    responses = await engine.step()
    for response in responses:
        if response.finished:
            print(f"Response: {response.outputs[0]}")
```

拓展示例：
```python
class MooncakeEngineInterface(InferenceEngineInterface):
    # 实现 Mooncake 特定的推理逻辑
    # 统一的接口，方便切换

class SGLangEngineInterface(InferenceEngineInterface):  
    # 实现 SGLang 特定的推理逻辑
    # 保持接口一致性
```
在使用该代码示例时，只需要根据具体的推理引擎创建相应的接口实现，然后使用统一的接口进行推理操作。这样，当需要切换到其他推理引擎时，只需要修改创建接口的部分，而不需要修改调用代码的部分。
```python
# 到时候用户可以轻松切换引擎
engine = create_inference_engine("sglang", config)
# 或者
engine = create_inference_engine("mooncake", config)
```