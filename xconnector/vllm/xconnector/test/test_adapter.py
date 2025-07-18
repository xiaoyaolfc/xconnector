from xconnector.core.core import XConnectorCore
from xconnector.adapters.inference.vllm_adapter import VLLMAdapter


class MockCore(XConnectorCore):
    def register_endpoints(self):
        pass


def test_vllm_adapter():  # 重命名测试用例
    core = MockCore()
    adapter = VLLMAdapter(core)  # 实例化子类（子类应实现register_endpoints）
    assert adapter is not None  # 验证初始化成功