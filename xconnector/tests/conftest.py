# xconnector/tests/conftest.py
import pytest
import logging

def pytest_configure(config):
    """配置 pytest，设置日志级别为 DEBUG 以便查看详细日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """设置日志捕获"""
    caplog.set_level(logging.DEBUG)
    yield