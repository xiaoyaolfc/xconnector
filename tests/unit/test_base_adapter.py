import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from xconnector.adapters.base_adapter import BaseAdapter
from xconnector.interfaces.base_interface import AdapterStatus, HealthStatus


class TestableAdapter(BaseAdapter):
    """一个可测试的Adapter，避免使用抽象基类直接实例化"""
    def __init__(self):
        super().__init__(core_instance=None, config={})
        self.adapter_name = "TestAdapter"
        self.adapter_version = "1.0.0"


@pytest.fixture
def adapter():
    return TestableAdapter()


@pytest.mark.asyncio
async def test_initialize_success(adapter):
    with patch.object(adapter, 'validate_config', return_value=(True, None)), \
         patch.object(adapter, 'emit_event', new_callable=AsyncMock), \
         patch.object(adapter, '_initialize_impl', new_callable=AsyncMock, return_value=True):

        result = await adapter.initialize()

        assert result is True
        assert adapter.status == AdapterStatus.READY


@pytest.mark.asyncio
async def test_initialize_failure_due_to_invalid_config(adapter):
    with patch.object(adapter, 'validate_config', return_value=(False, "Missing field")), \
         patch.object(adapter, 'emit_event', new_callable=AsyncMock):

        result = await adapter.initialize()

        assert result is False
        assert adapter.status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_start_success(adapter):
    adapter.status = AdapterStatus.READY
    with patch.object(adapter, 'before_start', new_callable=AsyncMock), \
         patch.object(adapter, '_start_impl', new_callable=AsyncMock, return_value=True), \
         patch.object(adapter, 'after_start', new_callable=AsyncMock), \
         patch.object(adapter, 'emit_event', new_callable=AsyncMock):

        result = await adapter.start()

        assert result is True
        assert adapter.status == AdapterStatus.RUNNING


@pytest.mark.asyncio
async def test_stop_success(adapter):
    adapter.status = AdapterStatus.RUNNING
    with patch.object(adapter, 'before_stop', new_callable=AsyncMock), \
         patch.object(adapter, '_stop_impl', new_callable=AsyncMock, return_value=True), \
         patch.object(adapter, 'after_stop', new_callable=AsyncMock), \
         patch.object(adapter, 'emit_event', new_callable=AsyncMock):

        result = await adapter.stop()

        assert result is True
        assert adapter.status == AdapterStatus.STOPPED


@pytest.mark.asyncio
async def test_health_check_default_healthy(adapter):
    adapter.status = AdapterStatus.READY
    with patch.object(adapter, '_health_check_impl', return_value=None):
        result = await adapter.health_check()
        assert result.status == HealthStatus.HEALTHY
        assert "uptime" in result.details
        assert "request_count" in result.details


@pytest.mark.asyncio
async def test_execute_with_metrics_success(adapter):
    mock_func = MagicMock(return_value=42)
    result = await adapter._execute_with_metrics(mock_func)
    assert result == 42
    assert adapter._request_count == 1


@pytest.mark.asyncio
async def test_execute_with_metrics_success(adapter):
    # 创建一个真实的函数而不是 MagicMock
    async def mock_func():
        return 42

    result = await adapter._execute_with_metrics(mock_func)
    assert result == 42
    assert adapter._request_count == 1

def test_get_metrics_success(adapter):
    with patch.object(adapter._process, 'memory_info') as mem_mock, \
         patch.object(adapter._process, 'cpu_percent') as cpu_mock:
        mem_mock.return_value.rss = 50 * 1024 * 1024  # 50MB
        cpu_mock.return_value = 10.0
        metrics = adapter.get_metrics()

        assert metrics.memory_usage == 50.0
        assert metrics.cpu_usage == 10.0
        assert "uptime" in metrics.custom_metrics
