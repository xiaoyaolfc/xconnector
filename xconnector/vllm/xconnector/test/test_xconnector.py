import pytest
from xconnector.vllm.xconnector.xconnector import XXCacheConnector
from vllm.config import VllmConfig  # 假设VllmConfig在vllm.config模块中


@pytest.mark.skip(reason="需要模拟依赖的lmcache模块，暂不测试")
def test_xxcache_connector():
    config = VllmConfig()
    connector = XXCacheConnector(0, 0, config)
    # 这里可以添加更多测试代码来测试XXCacheConnector的方法
