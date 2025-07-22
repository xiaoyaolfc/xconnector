import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from xconnector.adapters.distributed.dynamo_adapter import DynamoAdapter, WorkerStatus, WorkerInfo


@pytest.fixture
def dynamo_adapter():
    adapter = DynamoAdapter(config={
        "namespace": "test",
        "routing_policy": {
            "strategy": "least_loaded",
            "health_check_interval": 0.1,  # 更短的间隔便于测试
            "unhealthy_threshold": 3
        }
    })

    # 模拟ETCD客户端
    adapter.etcd_client = MagicMock()
    adapter.etcd_client.put = AsyncMock()
    adapter.etcd_client.get = AsyncMock(return_value=None)

    # 禁用实际的健康监控任务
    adapter.health_check_task = None

    return adapter


@pytest.mark.asyncio
async def test_worker_registration(dynamo_adapter):
    worker_id = "worker-001"
    worker_info = {
        "model": "test-model",
        "gpu_memory": 4096,
        "endpoint": "http://localhost:8000"
    }

    # 注册工作节点
    result = await dynamo_adapter.register_worker(worker_id, worker_info)

    assert result is True
    assert worker_id in dynamo_adapter.workers
    assert dynamo_adapter.workers[worker_id].status == WorkerStatus.READY
    assert dynamo_adapter.workers[worker_id].model_name == "test-model"

    # 验证ETCD注册被调用
    dynamo_adapter.etcd_client.put.assert_called()


@pytest.mark.asyncio
async def test_worker_routing(dynamo_adapter):
    now = datetime.now()

    # 添加两个工作节点
    worker1 = WorkerInfo(
        worker_id="worker-001",
        model_name="model-A",
        gpu_memory=4096,
        status=WorkerStatus.READY,
        endpoint="http://localhost:8001",
        registered_at=now,
        last_heartbeat=now,
        active_requests=5,
        total_requests=0,
        error_count=0,
        metadata={}
    )

    worker2 = WorkerInfo(
        worker_id="worker-002",
        model_name="model-A",
        gpu_memory=8192,
        status=WorkerStatus.READY,
        endpoint="http://localhost:8002",
        registered_at=now,
        last_heartbeat=now,
        active_requests=2,
        total_requests=0,
        error_count=0,
        metadata={}
    )

    dynamo_adapter.workers = {
        "worker-001": worker1,
        "worker-002": worker2
    }

    # 创建测试请求
    request = {"model": "model-A", "tokens": [1, 2, 3]}

    # 路由请求（最少连接策略）
    selected_worker = await dynamo_adapter.route_request(request)

    # 验证选择了负载较低的工作节点
    assert selected_worker == "worker-002"
    assert dynamo_adapter.workers["worker-002"].active_requests == 3


@pytest.mark.asyncio
async def test_health_monitoring(dynamo_adapter):
    now = datetime.now()

    # 添加工作节点
    worker = WorkerInfo(
        worker_id="worker-001",
        model_name="model-A",
        gpu_memory=4096,
        status=WorkerStatus.READY,
        endpoint="http://localhost:8001",
        registered_at=now,
        last_heartbeat=now - timedelta(seconds=100),
        active_requests=0,
        total_requests=0,
        error_count=0,
        metadata={}
    )

    dynamo_adapter.workers = {"worker-001": worker}

    # 直接调用一次健康检查逻辑，避免无限循环
    await dynamo_adapter._check_worker_health()

    # 验证节点状态变为不健康
    assert worker.status == WorkerStatus.UNHEALTHY
    assert worker.error_count == 1