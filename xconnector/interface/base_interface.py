class BaseInterface:
    def get_capabilities(self) -> Dict[str, Any]:
        """返回适配器支持的功能"""

    def health_check(self) -> bool:
        """健康检查"""

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""