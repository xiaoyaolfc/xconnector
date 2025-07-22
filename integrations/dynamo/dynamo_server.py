# xconnector/integration/dynamo/dynamo_service.py

from dynamo.sdk import service, endpoint, depends, async_on_start
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk import dynamo_context
from dynamo.planner.planner_sla import Planner
from xconnector.core.connector import XConnector
from typing import Dict, Any


@service(
    dynamo={
        "namespace": "xconnector",
        "replicas": 1,
    },
    resources={"cpu": "2", "memory": "4Gi"},
    workers=1,
)
class XConnectorService:
    """XConnector 作为 Dynamo 原生服务"""

    # 依赖注入 Dynamo 组件
    planner = depends(Planner)

    def __init__(self):
        # 使用 Dynamo 的配置系统
        self.config = ServiceConfig.get_parsed_config("XConnectorService")

        # 初始化 XConnector
        self.connector = XConnector(self.config)

        # 获取 Dynamo 运行时信息
        self.namespace = None
        self.component_name = None

    @async_on_start
    async def initialize(self):
        """使用 Dynamo 的生命周期钩子"""
        # 获取 Dynamo 上下文
        runtime = dynamo_context["runtime"]
        component = dynamo_context["component"]

        self.namespace, self.component_name = self.__class__.dynamo_address()

        # 启动 XConnector
        await self.connector.start()

        # 注册到 Dynamo 的服务发现
        await self._register_service(runtime)

    @endpoint()
    async def get_adapter_status(self) -> Dict[str, Any]:
        """暴露为 Dynamo endpoint"""
        return self.connector.list_adapters()

    @endpoint()
    async def route_cache_operation(self, request: Dict[str, Any]):
        """统一的缓存路由接口"""
        return await self.connector.route_message(
            source=request["source"],
            target=request["target"],
            method=request["method"],
            **request.get("params", {})
        )