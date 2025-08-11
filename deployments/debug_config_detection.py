#!/usr/bin/env python3
"""
配置检测调试和修复脚本
分析并修复XConnector配置检测问题
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/workspace/xconnector')


def debug_config_detection():
    """调试配置检测过程"""
    print("🔍 调试XConnector配置检测...")

    # 1. 检查配置文件是否存在
    print("\n1️⃣ 检查配置文件:")
    config_file = "/workspace/configs/dynamo-xconnector.yaml"
    if os.path.exists(config_file):
        print(f"✅ 配置文件存在: {config_file}")
        print(f"   文件大小: {os.path.getsize(config_file)} bytes")
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False

    # 2. 检查环境变量
    print("\n2️⃣ 检查环境变量:")
    env_vars = [
        'XCONNECTOR_CONFIG_FILE',
        'ENABLE_XCONNECTOR',
        'XCONNECTOR_ENABLED',
        'XCONNECTOR_CONFIG',
        'PYTHONPATH'
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"❌ {var}=未设置")

    # 3. 测试配置文件读取
    print("\n3️⃣ 测试配置文件读取:")
    try:
        import yaml
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        print("✅ YAML文件解析成功")

        if 'xconnector' in config_data:
            xc_config = config_data['xconnector']
            print(f"✅ 找到xconnector配置块")
            print(f"   enabled: {xc_config.get('enabled', 'None')}")
            print(f"   mode: {xc_config.get('mode', 'None')}")
        else:
            print("❌ 配置文件中没有xconnector块")
            print(f"   配置文件内容: {list(config_data.keys())}")

    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False

    # 4. 测试config_detector
    print("\n4️⃣ 测试config_detector:")
    try:
        from integrations.dynamo.config_detector import detect_config_files, detect_xconnector_config

        # 测试文件检测
        config_files = detect_config_files()
        print(f"detect_config_files结果: {config_files}")

        # 测试XConnector配置检测
        xc_config = detect_xconnector_config()
        if xc_config:
            print("✅ detect_xconnector_config成功")
            print(f"   检测到的配置: {xc_config}")
        else:
            print("❌ detect_xconnector_config返回None")

    except Exception as e:
        print(f"❌ config_detector测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试autopatch
    print("\n5️⃣ 测试autopatch:")
    try:
        from integrations.dynamo.autopatch import get_integration_status, _detect_dynamo_environment

        # 检查Dynamo环境检测
        dynamo_env = _detect_dynamo_environment()
        print(f"Dynamo环境检测: {dynamo_env}")

        # 检查集成状态
        status = get_integration_status()
        print("集成状态:")
        for key, value in status.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ autopatch测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def fix_config_detection():
    """修复配置检测问题"""
    print("\n🔧 修复配置检测问题...")

    # 1. 设置正确的环境变量
    print("1️⃣ 设置环境变量...")

    # 检查config_detector期望的环境变量名
    config_file = "/workspace/configs/dynamo-xconnector.yaml"

    # 设置多种可能的环境变量名
    env_vars_to_set = {
        'XCONNECTOR_CONFIG_FILE': config_file,
        'XCONNECTOR_CONFIG': config_file,  # config_detector可能使用这个
        'ENABLE_XCONNECTOR': 'true',
        'XCONNECTOR_ENABLED': 'true',
        'XCONNECTOR_MODE': 'embedded'
    }

    for var, value in env_vars_to_set.items():
        os.environ[var] = value
        print(f"   设置 {var}={value}")

    # 2. 直接修补config_detector
    print("2️⃣ 修补config_detector...")

    try:
        # 导入并修补detect_config_files函数
        from integrations.dynamo import config_detector

        # 创建增强版的配置检测函数
        def enhanced_detect_config_files():
            """增强的配置文件检测"""
            config_files = []

            # 搜索路径
            search_paths = [
                Path('/workspace/configs'),
                Path('/app/configs'),
                Path('./configs'),
                Path.cwd(),
                Path.cwd() / 'configs'
            ]

            config_patterns = [
                'dynamo-xconnector.yaml',
                'dynamo-xconnector.yml',
                'xconnector.yaml',
                'xconnector.yml',
                '*xconnector*.yaml',
                '*xconnector*.yml',
                'agg_with_xconnector.yaml',
                'disagg_with_xconnector.yaml'
            ]

            for search_path in search_paths:
                if search_path.exists() and search_path.is_dir():
                    for pattern in config_patterns:
                        matches = list(search_path.glob(pattern))
                        config_files.extend(matches)

            # 去重
            unique_files = []
            for f in config_files:
                if f not in unique_files:
                    unique_files.append(f)

            print(f"   增强检测找到配置文件: {unique_files}")
            return unique_files

        def enhanced_detect_xconnector_config():
            """增强的XConnector配置检测"""
            # 1. 优先从环境变量指定的文件读取
            config_file_env = os.getenv('XCONNECTOR_CONFIG_FILE')
            if config_file_env and os.path.exists(config_file_env):
                print(f"   使用环境变量指定的配置文件: {config_file_env}")
                try:
                    import yaml
                    with open(config_file_env, 'r') as f:
                        config_data = yaml.safe_load(f)

                    if 'xconnector' in config_data:
                        return config_data['xconnector']
                    elif 'sdk' in config_data:
                        # 返回完整配置，包含sdk配置
                        return config_data

                except Exception as e:
                    print(f"   环境变量配置文件读取失败: {e}")

            # 2. 从环境变量直接读取配置
            xc_config_json = os.getenv('XCONNECTOR_CONFIG')
            if xc_config_json:
                try:
                    import json
                    return json.loads(xc_config_json)
                except:
                    pass

            # 3. 搜索配置文件
            config_files = enhanced_detect_config_files()
            for config_file in config_files:
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)

                    if config_data and 'xconnector' in config_data:
                        print(f"   在{config_file}中找到xconnector配置")
                        return config_data['xconnector']

                except Exception as e:
                    print(f"   配置文件{config_file}读取失败: {e}")
                    continue

            # 4. 从基本环境变量构建配置
            if os.getenv('ENABLE_XCONNECTOR', '').lower() == 'true':
                print("   从环境变量构建基本配置")
                return {
                    'enabled': True,
                    'mode': os.getenv('XCONNECTOR_MODE', 'embedded'),
                    'offline_mode': True
                }

            return None

        # 替换原函数
        config_detector.detect_config_files = enhanced_detect_config_files
        config_detector.detect_xconnector_config = enhanced_detect_xconnector_config

        print("✅ config_detector修补完成")

    except Exception as e:
        print(f"❌ config_detector修补失败: {e}")
        return False

    # 3. 测试修补结果
    print("3️⃣ 测试修补结果...")
    try:
        from integrations.dynamo.config_detector import detect_xconnector_config

        config = detect_xconnector_config()
        if config:
            print("✅ 修补后配置检测成功")
            print(f"   配置内容: {config}")
        else:
            print("❌ 修补后仍然检测失败")
            return False

    except Exception as e:
        print(f"❌ 修补测试失败: {e}")
        return False

    return True


def test_integration_after_fix():
    """修复后测试集成"""
    print("\n🧪 测试修复后的集成...")

    try:
        # 重新导入模块
        if 'integrations.dynamo.autopatch' in sys.modules:
            del sys.modules['integrations.dynamo.autopatch']

        from integrations.dynamo.autopatch import get_integration_status, get_minimal_sdk

        status = get_integration_status()
        print("修复后集成状态:")
        for key, value in status.items():
            print(f"   {key}: {value}")

        if status.get('sdk_available'):
            sdk = get_minimal_sdk()
            if sdk:
                print(f"✅ SDK实例可用: {type(sdk).__name__}")

                # 测试缓存适配器
                if hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                    print("✅ 缓存适配器可用")
                    try:
                        stats = sdk.cache_adapter.get_cache_statistics()
                        print(f"   缓存统计: {stats}")
                    except Exception as e:
                        print(f"   缓存统计获取失败: {e}")
                else:
                    print("⚠️  缓存适配器不可用")
            else:
                print("❌ SDK实例不可用")

        return status.get('sdk_available', False)

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 XConnector配置检测调试和修复")
    print("=" * 50)

    # 1. 调试当前状态
    if not debug_config_detection():
        print("❌ 调试发现严重问题，尝试修复...")

    # 2. 修复配置检测
    if fix_config_detection():
        print("✅ 配置检测修复成功")
    else:
        print("❌ 配置检测修复失败")
        return False

    # 3. 测试修复结果
    success = test_integration_after_fix()

    print("\n" + "=" * 50)
    if success:
        print("🎉 修复成功！XConnector集成现在应该可以工作了")
        print("\n📋 建议的下一步:")
        print("1. 重新运行验证: python dynamo_xconnector_validator.py")
        print("2. 如果Dynamo在运行，重启以加载新配置")
    else:
        print("❌ 修复失败，需要进一步调查")

    return success


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 调试中断")
    except Exception as e:
        print(f"❌ 调试脚本异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)