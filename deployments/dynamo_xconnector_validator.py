#!/usr/bin/env python3
"""
手动SDK初始化脚本
当自动检测失败时，手动初始化XConnector SDK
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/workspace/xconnector')


async def manual_sdk_initialization():
    """手动初始化SDK"""
    print("🔧 手动初始化XConnector SDK...")

    try:
        # 1. 直接导入SDK
        from xconnector.sdk import MinimalSDK
        from xconnector.sdk.config import SDKConfig, SDKMode

        print("✅ SDK模块导入成功")

        # 2. 创建配置
        print("📝 创建SDK配置...")

        config = SDKConfig(
            mode=SDKMode.EMBEDDED,
            enable_kv_cache=True,
            enable_distributed=True,
            adapters=[
                {
                    "name": "lmcache",
                    "type": "cache",
                    "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                    "config": {
                        "storage_backend": "memory",
                        "max_cache_size": 1024,
                        "enable_compression": True,
                        "block_size": 16
                    },
                    "enabled": True,
                    "priority": 1
                }
            ]
        )

        print("✅ SDK配置创建成功")

        # 3. 初始化SDK
        print("🚀 初始化SDK实例...")

        sdk = MinimalSDK(config)
        success = await sdk.initialize()

        if success:
            print("✅ SDK初始化成功")

            # 4. 手动设置到autopatch模块
            try:
                import integrations.dynamo.autopatch as autopatch
                autopatch._minimal_sdk = sdk
                autopatch._integration_initialized = True
                print("✅ SDK已注册到autopatch")

            except Exception as e:
                print(f"⚠️  SDK注册警告: {e}")

            # 5. 测试SDK功能
            print("🧪 测试SDK功能...")

            if hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                print("✅ 缓存适配器可用")

                try:
                    stats = sdk.cache_adapter.get_cache_statistics()
                    print(f"📊 缓存统计: {stats}")
                except Exception as e:
                    print(f"⚠️  缓存统计获取失败: {e}")
            else:
                print("⚠️  缓存适配器不可用")

            return True

        else:
            print("❌ SDK初始化失败")
            return False

    except Exception as e:
        print(f"❌ 手动初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    print("🛠️  XConnector手动初始化")
    print("=" * 40)

    success = await manual_sdk_initialization()

    if success:
        print("\n🎉 手动初始化成功！")
        print("现在可以运行验证脚本测试功能")
    else:
        print("\n❌ 手动初始化失败")

    return success


if __name__ == "__main__":
    import asyncio

    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"❌ 脚本异常: {e}")
        sys.exit(1)