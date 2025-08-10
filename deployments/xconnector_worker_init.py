"""
VllmWorker进程专用的XConnector初始化（修复版）
"""
import os
import sys
import logging

# 添加XConnector到Python路径
if '/workspace/xconnector' not in sys.path:
    sys.path.insert(0, '/workspace/xconnector')


def init_xconnector_in_worker():
    """在VllmWorker进程中初始化XConnector"""

    # 设置环境变量
    os.environ['XCONNECTOR_CONFIG_FILE'] = '/workspace/configs/dynamo-xconnector-offline.yaml'
    os.environ['ENABLE_XCONNECTOR'] = 'true'
    
    # 添加日志目录环境变量
    os.environ['XCONNECTOR_LOG_DIR'] = '/workspace/xconnector/log'

    print("[INIT] Initializing XConnector in VllmWorker process...")

    try:
        # 导入并初始化XConnector
        from integrations.dynamo.autopatch import get_integration_status
        status = get_integration_status()

        print("XConnector status in VllmWorker:")
        for key, value in status.items():
            status_label = '[OK]' if value else '[NO]'
            print(f"   {status_label} {key}: {value}")

        # 如果成功，尝试手动patch当前进程中的类
        if status.get('sdk_available'):
            from integrations.dynamo.worker_injector import patch_existing_workers
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if sdk:
                # 手动patch当前进程中的Worker类
                patched_count = patch_existing_workers(sdk)
                print(f"[PATCH] Manual patch result: {patched_count} classes")

                # 修复：使用快照遍历sys.modules，避免字典大小改变错误
                _check_worker_classes_safely()

        return status

    except Exception as e:
        print(f"[ERROR] XConnector initialization in VllmWorker failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _check_worker_classes_safely():
    """安全地检查当前进程中的Worker类"""
    try:
        import inspect
        
        # 创建sys.modules的快照，避免迭代过程中字典大小改变
        modules_snapshot = list(sys.modules.items())
        
        worker_classes = []
        for module_name, module in modules_snapshot:
            if module is None:
                continue
            
            try:
                # 避免使用dir()，它可能触发新的导入
                if hasattr(module, '__dict__'):
                    attr_names = list(module.__dict__.keys())
                    for attr_name in attr_names:
                        try:
                            attr = getattr(module, attr_name, None)
                            if (attr and inspect.isclass(attr) and
                                'worker' in attr.__name__.lower() and
                                (hasattr(attr, 'recv_kv_caches') or hasattr(attr, 'send_kv_caches'))):
                                worker_classes.append(f"{module_name}.{attr.__name__}")
                        except (AttributeError, TypeError, ValueError):
                            # 忽略获取属性时的错误
                            continue
            except (AttributeError, TypeError):
                # 忽略模块没有__dict__或其他错误
                continue

        print(f"[CHECK] Worker classes in current process: {worker_classes}")
        
    except Exception as e:
        print(f"[WARN] Error checking Worker classes: {e}")


def setup_worker_environment():
    """设置Worker进程环境"""
    print("[ENV] Setting up VllmWorker process environment variables...")
    
    # 核心XConnector环境变量
    env_vars = {
        'XCONNECTOR_CONFIG_FILE': '/workspace/configs/dynamo-xconnector-offline.yaml',
        'ENABLE_XCONNECTOR': 'true',
        'XCONNECTOR_LOG_DIR': '/workspace/xconnector/log',
        'XCONNECTOR_MODE': 'worker',
        'XCONNECTOR_WORKER_TYPE': 'vllm',
        
        # Python路径
        'PYTHONPATH': f"{os.environ.get('PYTHONPATH', '')}:/workspace/xconnector",
        
        # 日志级别
        'XCONNECTOR_LOG_LEVEL': 'INFO',
        
        # 禁用某些可能引起问题的功能
        'XCONNECTOR_DISABLE_ETCD': 'true',
        'XCONNECTOR_OFFLINE_MODE': 'true',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   Set {key}={value}")


def verify_xconnector_setup():
    """验证XConnector设置"""
    print("[VERIFY] Verifying XConnector setup...")
    
    # 检查关键文件是否存在
    config_file = os.environ.get('XCONNECTOR_CONFIG_FILE')
    if config_file and os.path.exists(config_file):
        print(f"   [OK] Config file exists: {config_file}")
    else:
        print(f"   [ERROR] Config file not found: {config_file}")
    
    # 检查XConnector模块是否可导入
    try:
        from integrations.dynamo.autopatch import get_integration_status
        print("   [OK] XConnector module imported successfully")
        return True
    except ImportError as e:
        print(f"   [ERROR] Failed to import XConnector module: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("[START] VllmWorker XConnector Initialization Script (Fixed Version)")
    print("=" * 60)
    
    # 1. 设置环境
    setup_worker_environment()
    
    # 2. 验证设置
    if not verify_xconnector_setup():
        print("[EXIT] XConnector setup verification failed, exiting")
        return False
    
    # 3. 初始化XConnector
    status = init_xconnector_in_worker()
    
    if status and status.get('initialized'):
        print("\n[SUCCESS] VllmWorker XConnector initialization completed successfully!")
        return True
    else:
        print("\n[FAILURE] VllmWorker XConnector initialization failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
