"""
vLLM Worker 启动时的 XConnector 注入钩子
"""
import os
import sys
import logging

# 添加 XConnector 路径
if '/workspace/xconnector' not in sys.path:
    sys.path.insert(0, '/workspace/xconnector')

# 设置环境变量
os.environ.update({
    'XCONNECTOR_CONFIG_FILE': '/workspace/configs/dynamo-xconnector-offline.yaml',
    'ENABLE_XCONNECTOR': 'true',
    'XCONNECTOR_LOG_DIR': '/workspace/xconnector/log',
    'XCONNECTOR_MODE': 'worker',
    'XCONNECTOR_WORKER_TYPE': 'vllm',
    'PYTHONPATH': f"{os.environ.get('PYTHONPATH', '')}:/workspace/xconnector",
})

print("🔗 vLLM Worker XConnector 启动钩子已激活")

# 导入并初始化 XConnector
try:
    from integrations.dynamo.autopatch import get_integration_status, get_minimal_sdk

    status = get_integration_status()

    if status.get('sdk_available'):
        sdk = get_minimal_sdk()
        print(f"XConnector SDK 就绪: {type(sdk)}")

        # 动态 patch 当前进程中的 Worker 类
        import inspect
        import vllm.worker.worker  # 确保导入 vLLM Worker 模块

        # 查找并 patch vLLM Worker 类
        for name, obj in inspect.getmembers(vllm.worker.worker):
            if inspect.isclass(obj) and 'Worker' in name:
                print(f"找到 Worker 类: {name}")

                # 直接注入 XConnector 功能
                if hasattr(obj, 'recv_kv_caches') or hasattr(obj, 'send_kv_caches'):
                    # 添加 XConnector 引用
                    obj._xconnector_sdk = sdk
                    print(f"已注入 XConnector 到 {name}")

except Exception as e:
    print(f" XConnector 启动钩子执行失败: {e}")

print("vLLM Worker 可以开始运行")