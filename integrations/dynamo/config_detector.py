# integrations/dynamo/config_detector.py
"""
配置检测器 - 修复版

根据实际目录结构优化配置检测逻辑：
1. /workspace/configs (挂载的配置目录，优先级最高)
2. /workspace/example/llm/configs (Dynamo运行目录的配置)
3. /workspace/xconnector/integrations/dynamo/configs (集成配置)
4. /workspace/xconnector/deployments (部署配置)
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# 获取logger
logger = logging.getLogger('xconnector.dynamo.config_detector')


def detect_config_files() -> List[Path]:
    """
    检测可能的配置文件

    按优先级搜索以下位置：
    1. /workspace/configs (挂载的配置，最高优先级)
    2. /workspace/example/llm/configs (Dynamo运行目录)
    3. /workspace/xconnector/integrations/dynamo/configs (集成配置)
    4. /workspace/xconnector/deployments (部署配置)
    5. 环境变量指定的路径
    6. 当前工作目录
    """
    config_files = []
    seen_files = set()  # 避免重复

    # 定义搜索路径（按优先级排序）
    search_paths = [
        Path('/workspace/configs'),  # 挂载的配置目录（最高优先级）
        Path('/workspace/example/llm/configs'),  # Dynamo运行目录
        Path('/workspace/xconnector/integrations/dynamo/configs'),  # 集成配置
        Path('/workspace/xconnector/deployments'),  # 部署配置
        Path.cwd(),  # 当前工作目录
    ]

    # 从环境变量添加额外路径
    if os.getenv('XCONNECTOR_CONFIG_PATH'):
        custom_path = Path(os.getenv('XCONNECTOR_CONFIG_PATH'))
        if custom_path not in search_paths:
            search_paths.insert(0, custom_path)  # 自定义路径优先级最高

    # 支持的配置文件名（包含XConnector配置的文件）
    config_patterns = [
        'dynamo-xconnector.yaml',
        'agg_with_xconnector.yaml',
        'disagg_with_xconnector.yaml',
        'agg_router_with_xconnector.yaml',
        'disagg_router_with_xconnector.yaml',
        '*xconnector*.yaml',  # 任何包含xconnector的yaml文件
        '*xconnector*.yml',
    ]

    # 搜索配置文件
    for search_path in search_paths:
        if not search_path.exists():
            logger.debug(f"Search path does not exist: {search_path}")
            continue

        logger.debug(f"Searching in: {search_path}")

        # 搜索特定文件名
        for pattern in config_patterns:
            if '*' in pattern:
                # 使用glob模式
                for file_path in search_path.glob(pattern):
                    if file_path.is_file() and file_path not in seen_files:
                        config_files.append(file_path)
                        seen_files.add(file_path)
                        logger.info(f"Found config file: {file_path}")
            else:
                # 直接查找文件
                file_path = search_path / pattern
                if file_path.exists() and file_path.is_file() and file_path not in seen_files:
                    config_files.append(file_path)
                    seen_files.add(file_path)
                    logger.info(f"Found config file: {file_path}")

    # 从环境变量直接指定的文件
    if os.getenv('XCONNECTOR_CONFIG_FILE'):
        config_file = Path(os.getenv('XCONNECTOR_CONFIG_FILE'))
        if config_file.exists() and config_file not in seen_files:
            config_files.insert(0, config_file)  # 环境变量指定的文件优先级最高
            logger.info(f"Found config file from env: {config_file}")

    # 从命令行参数查找
    for arg in sys.argv[1:]:
        if arg.endswith(('.yaml', '.yml')):
            config_path = Path(arg)
            if config_path.exists() and config_path not in seen_files:
                config_files.append(config_path)
                logger.info(f"Found config file from args: {config_path}")

    logger.info(f"Total config files found: {len(config_files)}")
    return config_files


def load_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载YAML文件
    """
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Successfully loaded YAML file: {file_path}")
        return data
    except ImportError:
        logger.error("PyYAML not installed, cannot load YAML files")
        return None
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {e}")
        return None


def extract_xconnector_config_from_file(config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从配置数据中提取XConnector配置

    支持多种配置格式：
    1. 直接的 xconnector 块
    2. 嵌套在其他配置中的 xconnector 块
    3. 根级别的 XConnector 相关配置
    """
    if not isinstance(config_data, dict):
        return None

    # 1. 直接查找 xconnector 块
    if 'xconnector' in config_data:
        xc_config = config_data['xconnector']
        if isinstance(xc_config, dict):
            logger.debug("Found direct xconnector block")
            return xc_config

    # 2. 查找可能包含xconnector配置的其他键
    for key in ['dynamo', 'vllm', 'worker', 'engine']:
        if key in config_data and isinstance(config_data[key], dict):
            if 'xconnector' in config_data[key]:
                logger.debug(f"Found xconnector block under {key}")
                return config_data[key]['xconnector']

    # 3. 检查是否整个文件就是XConnector配置
    # 如果包含XConnector特有的键，则认为整个文件是配置
    xconnector_keys = ['adapters', 'cache_adapter', 'kv_cache', 'enable_xconnector']
    matching_keys = [k for k in xconnector_keys if k in config_data]
    if len(matching_keys) >= 2:  # 至少匹配2个键
        logger.debug(f"Treating entire file as XConnector config (matched keys: {matching_keys})")
        return config_data

    return None


def detect_xconnector_config_from_env() -> Optional[Dict[str, Any]]:
    """
    从环境变量中检测XConnector配置
    """
    config = {}

    # 1. JSON格式的完整配置
    if os.getenv('XCONNECTOR_CONFIG'):
        try:
            config = json.loads(os.getenv('XCONNECTOR_CONFIG'))
            logger.info("Loaded XConnector config from XCONNECTOR_CONFIG env")
            return config
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in XCONNECTOR_CONFIG: {e}")

    # 2. 单独的环境变量
    env_mappings = {
        'ENABLE_XCONNECTOR': ('enabled', lambda x: x.lower() in ['true', '1', 'yes']),
        'XCONNECTOR_ENABLED': ('enabled', lambda x: x.lower() in ['true', '1', 'yes']),
        'XCONNECTOR_MODE': ('mode', str),
        'XCONNECTOR_LOG_LEVEL': ('log_level', str.upper),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            try:
                config[config_key] = converter(env_value)
                logger.debug(f"Set {config_key} from {env_var}={env_value}")
            except Exception as e:
                logger.debug(f"Error converting {env_var}: {e}")

    # 3. 如果设置了ENABLE_XCONNECTOR但没有找到配置文件，创建默认配置
    if config.get('enabled') and not config.get('adapters'):
        logger.info("ENABLE_XCONNECTOR is true, creating default config")
        config.update({
            'mode': 'embedded',
            'adapters': [
                {
                    'name': 'lmcache',
                    'type': 'cache',
                    'class_path': 'xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter',
                    'enabled': True,
                    'config': {
                        'storage_backend': 'memory',
                        'max_cache_size': 1024
                    }
                }
            ]
        })

    return config if config else None


def detect_xconnector_config_from_files() -> Optional[Dict[str, Any]]:
    """
    从配置文件中检测XConnector配置
    """
    config_files = detect_config_files()

    if not config_files:
        logger.warning("No config files found in any search path")
        return None

    # 按优先级尝试每个配置文件
    for config_file in config_files:
        try:
            logger.debug(f"Trying to load config from: {config_file}")

            # 加载YAML文件
            config_data = load_yaml_file(config_file)
            if not config_data:
                continue

            # 提取XConnector配置
            xconnector_config = extract_xconnector_config_from_file(config_data)
            if xconnector_config:
                logger.info(f"Successfully loaded XConnector config from: {config_file}")
                # 添加来源信息
                xconnector_config['_config_source'] = str(config_file)
                return xconnector_config

        except Exception as e:
            logger.debug(f"Error processing {config_file}: {e}")
            continue

    logger.warning("No XConnector config found in any config file")
    return None


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置，override_config 优先
    """
    import copy
    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # 递归合并字典
            merged[key] = merge_configs(merged[key], value)
        else:
            # 直接覆盖
            merged[key] = value

    return merged


def detect_xconnector_config() -> Optional[Dict[str, Any]]:
    """
    检测XConnector配置（主入口）

    优先级：
    1. 环境变量中的完整配置
    2. 配置文件 + 环境变量覆盖
    """
    try:
        logger.debug("=" * 60)
        logger.debug("Starting XConnector config detection...")
        logger.debug(f"Working directory: {Path.cwd()}")
        logger.debug(f"Python path: {sys.path[:3]}")

        # 从文件检测配置
        file_config = detect_xconnector_config_from_files()

        # 从环境变量检测配置
        env_config = detect_xconnector_config_from_env()

        # 合并配置
        if file_config and env_config:
            # 环境变量配置优先
            final_config = merge_configs(file_config, env_config)
            logger.info("Merged config from file and environment")
            return final_config
        elif env_config:
            logger.info("Using config from environment variables")
            return env_config
        elif file_config:
            logger.info("Using config from file")
            return file_config
        else:
            # 最后的尝试：如果ENABLE_XCONNECTOR设置为true，创建最小配置
            if os.getenv('ENABLE_XCONNECTOR', '').lower() in ['true', '1', 'yes']:
                logger.info("ENABLE_XCONNECTOR is set, creating minimal config")
                return {
                    'enabled': True,
                    'mode': 'embedded',
                    'adapters': []
                }

            logger.warning("No XConnector config found anywhere")
            return None

    except Exception as e:
        logger.error(f"Error in detect_xconnector_config: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def validate_xconnector_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证XConnector配置
    """
    errors = []

    if not isinstance(config, dict):
        errors.append("Config must be a dictionary")
        return False, errors

    # 检查enabled字段
    if 'enabled' not in config:
        # 默认为True如果配置存在
        config['enabled'] = True
        logger.debug("Added default 'enabled': True")

    # 检查mode
    if 'mode' not in config:
        config['mode'] = 'embedded'
        logger.debug("Added default 'mode': embedded")

    # 验证adapters（可选）
    if 'adapters' in config:
        if not isinstance(config['adapters'], list):
            errors.append("'adapters' must be a list")
        else:
            for i, adapter in enumerate(config['adapters']):
                if not isinstance(adapter, dict):
                    errors.append(f"Adapter {i} must be a dictionary")
                elif 'name' not in adapter:
                    errors.append(f"Adapter {i} missing 'name' field")
                elif 'type' not in adapter:
                    errors.append(f"Adapter {i} missing 'type' field")

    is_valid = len(errors) == 0
    if is_valid:
        logger.debug("Config validation passed")
    else:
        logger.warning(f"Config validation failed: {errors}")

    return is_valid, errors


# 导出公共接口
__all__ = [
    'detect_xconnector_config',
    'detect_config_files',
    'load_yaml_file',
    'validate_xconnector_config',
    'detect_xconnector_config_from_env',
    'detect_xconnector_config_from_files'
]