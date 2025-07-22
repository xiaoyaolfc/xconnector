'''
    该部分代码替换原ai-dynamo/examples/vllm_v0/worker.py

# 在 VllmWorker 的 __init__ 方法最后添加
from integrations.dynamo.extension_loader import ExtensionLoader


class VllmWorker:
    def __init__(self):
        # ... 原有代码 ...

        # 加载扩展（一行代码）
        ExtensionLoader.load_extensions(self.config)
        ExtensionLoader.inject_into_worker(self)

'''