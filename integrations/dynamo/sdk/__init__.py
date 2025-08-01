# integrations/dynamo/sdk/__init__.py
"""
Dynamo SDK集成模块

提供XConnector与AI-Dynamo的深度集成支持
"""

from integrations.dynamo.sdk.integration import (
    DynamoXConnectorIntegration,
    DynamoIntegrationConfig,
    setup_xconnector_integration,
    async_setup_xconnector_integration,
    get_integration_instance,
    detect_dynamo_environment
)

from integrations.dynamo.sdk.worker_wrapper import (
    wrap_kv_cache_methods,
    wrap_distributed_methods,
    unwrap_worker_methods,
    get_worker_monitoring_stats,
    xconnector_integration,
    check_worker_compatibility,
    create_compatibility_report,
    get_performance_monitor
)

__all__ = [
    # 集成管理
    'DynamoXConnectorIntegration',
    'DynamoIntegrationConfig',
    'setup_xconnector_integration',
    'async_setup_xconnector_integration',
    'get_integration_instance',
    'detect_dynamo_environment',

    # Worker包装
    'wrap_kv_cache_methods',
    'wrap_distributed_methods',
    'unwrap_worker_methods',
    'get_worker_monitoring_stats',
    'xconnector_integration',
    'check_worker_compatibility',
    'create_compatibility_report',
    'get_performance_monitor'
]