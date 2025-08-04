# xconnector/integrations/dynamo/config_detector.py
"""
配置检测器

负责从多个来源检测和解析XConnector配置：
1. 环境变量配置
2. YAML配置文件中的xconnector配置块
3. 命令行参数中的配置文件

设计原则：
- 简单实用：只实现必要的检测功能
- 多源融合：支持多种配置来源
- 容错优先：配置解析失败时优雅降级
- 轻量级：最小依赖，快速执行
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

    检测策略：
    1. 当前目录的常见配置文件
    2. 命令行参数中的配置文件
    3. 环境变量指定的配置文件

    Returns:
        List[Path]: 找到的配置文件路径列表
    """
    config_files = []

    # 1. 常见的Dynamo配置文件名
    common_config_names = [
        'dynamo_config.yaml',
        'dynamo.yaml',
        'config.yaml',
        'agg_with_xconnector.yaml',
        'disagg_with_xconnector.yaml',
        'agg_router_with_xconnector.yaml',
        'disagg_router_with_xconnector.yaml'
    ]

    # 在当前目录查找
    current_dir = Path.cwd()
    for config_name in common_config_names:
        config_path = current_dir / config_name
        if config_path.exists() and config_path.is_file():
            config_files.append(config_path)
            logger.debug(f"Found config file: {config_path}")

    # 2. 从命令行参数中查找配置文件
    for arg in sys.argv[1:]:  # 跳过脚本名
        if arg.endswith(('.yaml', '.yml', '.json')):
            config_path = Path(arg)
            if config_path.exists() and config_path not in config_files:
                config_files.append(config_path)
                logger.debug(f"Found config file from args: {config_path}")
        elif arg.startswith('--config='):
            config_path = Path(arg.split('=', 1)[1])
            if config_path.exists() and config_path not in config_files:
                config_files.append(config_path)
                logger.debug(f"Found config file from --config: {config_path}")

    # 3. 从环境变量查找
    env_configs = [
        'DYNAMO_CONFIG',
        'CONFIG_FILE',
        'XCONNECTOR_CONFIG_FILE'
    ]

    for env_var in env_configs:
        config_path_str = os.getenv(env_var)
        if config_path_str:
            config_path = Path(config_path_str)
            if config_path.exists() and config_path not in config_files:
                config_files.append(config_path)
                logger.debug(f"Found config file from {env_var}: {config_path}")

    return config_files


def load_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载YAML文件

    Args:
        file_path: YAML文件路径

    Returns:
        Optional[Dict]: 解析后的配置字典，失败时返回None
    """
    try:
        # 尝试导入yaml
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML config files")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if isinstance(config, dict):
            logger.debug(f"Successfully loaded YAML config from {file_path}")
            return config
        else:
            logger.warning(f"YAML file {file_path} does not contain a dictionary")
            return None

    except Exception as e:
        logger.debug(f"Failed to load YAML file {file_path}: {e}")
        return None


def extract_xconnector_config_from_file(config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从配置文件数据中提取XConnector配置

    Args:
        config_data: 完整的配置数据

    Returns:
        Optional[Dict]: XConnector配置，未找到时返回None
    """
    try:
        # 直接查找xconnector配置块
        if 'xconnector' in config_data:
            xconnector_config = config_data['xconnector']
            if isinstance(xconnector_config, dict):
                logger.debug("Found xconnector config block")
                return xconnector_config

        # 查找Common配置中的xconnector
        if 'Common' in config_data and isinstance(config_data['Common'], dict):
            common_config = config_data['Common']
            if 'xconnector' in common_config:
                xconnector_config = common_config['xconnector']
                if isinstance(xconnector_config, dict):
                    logger.debug("Found xconnector config in Common block")
                    return xconnector_config

        return None

    except Exception as e:
        logger.debug(f"Error extracting xconnector config: {e}")
        return None


def detect_xconnector_config_from_files() -> Optional[Dict[str, Any]]:
    """
    从配置文件中检测XConnector配置

    Returns:
        Optional[Dict]: XConnector配置，未找到时返回None
    """
    config_files = detect_config_files()

    if not config_files:
        logger.debug("No config files found")
        return None

    # 按优先级遍历配置文件
    for config_file in config_files:
        try:
            # 只处理YAML文件（Dynamo主要使用YAML）
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = load_yaml_file(config_file)
                if config_data:
                    xconnector_config = extract_xconnector_config_from_file(config_data)
                    if xconnector_config:
                        logger.info(f"Found XConnector config in {config_file}")
                        return xconnector_config

        except Exception as e:
            logger.debug(f"Error processing config file {config_file}: {e}")
            continue

    return None


def detect_xconnector_config_from_env() -> Optional[Dict[str, Any]]:
    """
    从环境变量中检测XConnector配置

    Returns:
        Optional[Dict]: XConnector配置，未找到时返回None
    """
    config = {}

    try:
        # 1. 完整的JSON配置
        xconnector_config_json = os.getenv('XCONNECTOR_CONFIG')
        if xconnector_config_json:
            try:
                config = json.loads(xconnector_config_json)
                logger.debug("Found XConnector config in XCONNECTOR_CONFIG (JSON)")
                return config
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in XCONNECTOR_CONFIG: {e}")

        # 2. 分离的环境变量配置
        env_mappings = {
            'ENABLE_XCONNECTOR': ('enabled', lambda x: x.lower() == 'true'),
            'XCONNECTOR_MODE': ('mode', str),
            'XCONNECTOR_LOG_LEVEL': ('log_level', str.upper),
            'XCONNECTOR_GRACEFUL_DEGRADATION': ('graceful_degradation', lambda x: x.lower() == 'true'),
        }

        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    config[config_key] = converter(env_value)
                    logger.debug(f"Found {config_key} from {env_var}")
                except Exception as e:
                    logger.debug(f"Error converting {env_var}={env_value}: {e}")

        # 3. 简单的适配器配置
        if os.getenv('ENABLE_LMCACHE', '').lower() == 'true':
            config.setdefault('adapters', []).append({
                'name': 'lmcache',
                'type': 'cache',
                'enabled': True
            })
            logger.debug("Added LMCache adapter from ENABLE_LMCACHE")

        return config if config else None

    except Exception as e:
        logger.debug(f"Error detecting config from environment: {e}")
        return None


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        Dict: 合并后的配置
    """
    try:
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并字典
                merged[key] = merge_configs(merged[key], value)
            elif key == 'adapters' and isinstance(value, list):
                # 特殊处理adapters列表
                merged.setdefault('adapters', []).extend(value)
            else:
                # 直接覆盖
                merged[key] = value

        return merged

    except Exception as e:
        logger.debug(f"Error merging configs: {e}")
        return base_config


def detect_xconnector_config() -> Optional[Dict[str, Any]]:
    """
    检测XConnector配置（主入口函数）

    按优先级从多个来源检测配置：
    1. 环境变量（最高优先级）
    2. 配置文件

    Returns:
        Optional[Dict]: 合并后的XConnector配置，未找到时返回None
    """
    try:
        logger.debug("Starting XConnector config detection...")

        # 从配置文件检测
        file_config = detect_xconnector_config_from_files()

        # 从环境变量检测
        env_config = detect_xconnector_config_from_env()

        # 合并配置（环境变量优先）
        if file_config and env_config:
            final_config = merge_configs(file_config, env_config)
            logger.info("Merged XConnector config from files and environment")
            return final_config
        elif env_config:
            logger.info("Using XConnector config from environment")
            return env_config
        elif file_config:
            logger.info("Using XConnector config from files")
            return file_config
        else:
            logger.debug("No XConnector config found")
            return None

    except Exception as e:
        logger.error(f"Error detecting XConnector config: {e}")
        return None


def validate_xconnector_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证XConnector配置

    Args:
        config: XConnector配置字典

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 检查enabled字段
        if 'enabled' not in config:
            errors.append("Missing 'enabled' field")
        elif not isinstance(config['enabled'], bool):
            errors.append("'enabled' field must be boolean")

        # 检查适配器配置
        if 'adapters' in config:
            adapters = config['adapters']
            if not isinstance(adapters, list):
                errors.append("'adapters' must be a list")
            else:
                for i, adapter in enumerate(adapters):
                    if not isinstance(adapter, dict):
                        errors.append(f"Adapter {i} must be a dictionary")
                        continue

                    required_fields = ['name', 'type']
                    for field in required_fields:
                        if field not in adapter:
                            errors.append(f"Adapter {i} missing required field: {field}")

        is_valid = len(errors) == 0
        return is_valid, errors

    except Exception as e:
        errors.append(f"Validation error: {e}")
        return False, errors


# 导出
__all__ = [
    'detect_xconnector_config',
    'detect_config_files',
    'load_yaml_file',
    'validate_xconnector_config'
]