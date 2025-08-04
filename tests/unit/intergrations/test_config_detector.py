# tests/unit/integrations/dynamo/test_config_detector.py
"""
配置检测器单元测试

测试 config_detector.py 中的所有配置检测逻辑：
- 文件检测和解析
- 环境变量检测
- 配置合并和验证
- 错误处理机制
"""

import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from integrations.dynamo.config_detector import (
    detect_config_files,
    load_yaml_file,
    extract_xconnector_config_from_file,
    detect_xconnector_config_from_files,
    detect_xconnector_config_from_env,
    merge_configs,
    detect_xconnector_config,
    validate_xconnector_config
)


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录的fixture"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_yaml_config():
    """模拟YAML配置内容"""
    return {
        "Common": {
            "model": "/data/model/test-model",
            "xconnector": {
                "enabled": True,
                "mode": "embedded",
                "log_level": "INFO",
                "adapters": [
                    {
                        "name": "lmcache",
                        "type": "cache",
                        "enabled": True,
                        "config": {"storage_backend": "memory"}
                    }
                ]
            }
        },
        "VllmWorker": {
            "enable-prefix-caching": True
        }
    }


@pytest.fixture
def clean_env():
    """清理环境变量的fixture"""
    env_vars_to_clean = [
        'XCONNECTOR_CONFIG', 'ENABLE_XCONNECTOR', 'XCONNECTOR_MODE',
        'XCONNECTOR_LOG_LEVEL', 'XCONNECTOR_GRACEFUL_DEGRADATION', 'ENABLE_LMCACHE',
        'DYNAMO_CONFIG', 'CONFIG_FILE', 'XCONNECTOR_CONFIG_FILE'
    ]

    # 保存原始值
    original_values = {}
    for var in env_vars_to_clean:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # 恢复原始值
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class TestDetectConfigFiles:
    """测试配置文件检测功能"""

    def test_detect_common_config_files(self, temp_config_dir):
        """测试检测常见配置文件"""
        # 创建测试配置文件
        config_files = [
            'dynamo_config.yaml',
            'config.yaml',
            'agg_with_xconnector.yaml'
        ]

        for filename in config_files:
            (temp_config_dir / filename).write_text("test: config")

        with patch('pathlib.Path.cwd', return_value=temp_config_dir):
            found_files = detect_config_files()

        assert len(found_files) == len(config_files)
        found_names = {f.name for f in found_files}
        assert found_names == set(config_files)

    def test_detect_config_from_command_line_args(self, temp_config_dir):
        """测试从命令行参数中检测配置文件"""
        config_file = temp_config_dir / "custom_config.yaml"
        config_file.write_text("test: config")

        with patch.object(sys, 'argv', ['script.py', str(config_file)]):
            found_files = detect_config_files()

        assert len(found_files) == 1
        assert found_files[0] == config_file

    def test_detect_config_from_environment_variables(self, temp_config_dir, clean_env):
        """测试从环境变量中检测配置文件"""
        config_file = temp_config_dir / "env_config.yaml"
        config_file.write_text("test: config")

        os.environ['DYNAMO_CONFIG'] = str(config_file)

        found_files = detect_config_files()

        assert config_file in found_files

    def test_no_config_files_found(self, temp_config_dir):
        """测试没有找到配置文件的情况"""
        with patch('pathlib.Path.cwd', return_value=temp_config_dir):
            found_files = detect_config_files()

        assert len(found_files) == 0


class TestLoadYamlFile:
    """测试YAML文件加载功能"""

    def test_load_valid_yaml_file(self, temp_config_dir, mock_yaml_config):
        """测试加载有效的YAML文件"""
        config_file = temp_config_dir / "test_config.yaml"

        # 写入YAML内容
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_yaml_config, f)

        result = load_yaml_file(config_file)

        assert result is not None
        assert isinstance(result, dict)
        assert "Common" in result
        assert result["Common"]["model"] == "/data/model/test-model"

    def test_load_invalid_yaml_file(self, temp_config_dir):
        """测试加载无效的YAML文件"""
        config_file = temp_config_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        result = load_yaml_file(config_file)

        assert result is None

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        result = load_yaml_file(Path("/nonexistent/file.yaml"))

        assert result is None

    @patch('yaml.safe_load')
    def test_yaml_import_error(self, mock_yaml_load, temp_config_dir):
        """测试PyYAML不可用的情况"""
        config_file = temp_config_dir / "test.yaml"
        config_file.write_text("test: config")

        # 模拟导入错误
        with patch('builtins.__import__', side_effect=ImportError("No module named 'yaml'")):
            result = load_yaml_file(config_file)

        assert result is None


class TestExtractXConnectorConfig:
    """测试从配置数据中提取XConnector配置"""

    def test_extract_from_root_level(self):
        """测试从根级别提取xconnector配置"""
        config_data = {
            "xconnector": {
                "enabled": True,
                "mode": "embedded"
            }
        }

        result = extract_xconnector_config_from_file(config_data)

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"

    def test_extract_from_common_block(self, mock_yaml_config):
        """测试从Common配置块中提取xconnector配置"""
        result = extract_xconnector_config_from_file(mock_yaml_config)

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"
        assert len(result["adapters"]) == 1

    def test_extract_no_xconnector_config(self):
        """测试没有xconnector配置的情况"""
        config_data = {
            "Common": {
                "model": "/data/model/test"
            },
            "VllmWorker": {
                "enable-prefix-caching": True
            }
        }

        result = extract_xconnector_config_from_file(config_data)

        assert result is None

    def test_extract_invalid_xconnector_config(self):
        """测试无效的xconnector配置"""
        config_data = {
            "xconnector": "invalid_string_instead_of_dict"
        }

        result = extract_xconnector_config_from_file(config_data)

        assert result is None


class TestDetectXConnectorConfigFromEnv:
    """测试从环境变量检测XConnector配置"""

    def test_detect_from_json_env_var(self, clean_env):
        """测试从JSON环境变量检测配置"""
        config_json = json.dumps({
            "enabled": True,
            "mode": "embedded",
            "adapters": [{"name": "lmcache", "type": "cache"}]
        })

        os.environ['XCONNECTOR_CONFIG'] = config_json

        result = detect_xconnector_config_from_env()

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"
        assert len(result["adapters"]) == 1

    def test_detect_from_separate_env_vars(self, clean_env):
        """测试从分离的环境变量检测配置"""
        os.environ['ENABLE_XCONNECTOR'] = 'true'
        os.environ['XCONNECTOR_MODE'] = 'embedded'
        os.environ['XCONNECTOR_LOG_LEVEL'] = 'debug'
        os.environ['XCONNECTOR_GRACEFUL_DEGRADATION'] = 'true'

        result = detect_xconnector_config_from_env()

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"
        assert result["log_level"] == "DEBUG"
        assert result["graceful_degradation"] is True

    def test_detect_lmcache_adapter_from_env(self, clean_env):
        """测试从环境变量检测LMCache适配器"""
        os.environ['ENABLE_LMCACHE'] = 'true'

        result = detect_xconnector_config_from_env()

        assert result is not None
        assert "adapters" in result
        assert len(result["adapters"]) == 1
        assert result["adapters"][0]["name"] == "lmcache"
        assert result["adapters"][0]["type"] == "cache"

    def test_detect_invalid_json_env_var(self, clean_env):
        """测试无效的JSON环境变量"""
        os.environ['XCONNECTOR_CONFIG'] = '{"invalid": json}'

        result = detect_xconnector_config_from_env()

        # 因为JSON解析失败，但可能还有其他环境变量被检测到
        # 所以我们检查返回值不包含从JSON解析的配置
        if result:
            assert 'invalid' not in result  # 确保无效的JSON没有被解析
        else:
            assert result is None or result == {}

    def test_no_env_vars_set(self, clean_env):
        """测试没有设置环境变量的情况"""
        # 确保没有设置任何相关的环境变量
        env_vars_to_check = [
            'XCONNECTOR_CONFIG', 'ENABLE_XCONNECTOR', 'XCONNECTOR_MODE',
            'XCONNECTOR_LOG_LEVEL', 'XCONNECTOR_GRACEFUL_DEGRADATION', 'ENABLE_LMCACHE'
        ]

        for var in env_vars_to_check:
            if var in os.environ:
                del os.environ[var]

        result = detect_xconnector_config_from_env()

        assert result is None


class TestMergeConfigs:
    """测试配置合并功能"""

    def test_merge_simple_configs(self):
        """测试合并简单配置"""
        base_config = {
            "enabled": False,
            "mode": "service",
            "log_level": "INFO"
        }

        override_config = {
            "enabled": True,
            "log_level": "DEBUG"
        }

        result = merge_configs(base_config, override_config)

        assert result["enabled"] is True  # 被覆盖
        assert result["mode"] == "service"  # 保持原值
        assert result["log_level"] == "DEBUG"  # 被覆盖

    def test_merge_nested_configs(self):
        """测试合并嵌套配置"""
        base_config = {
            "enabled": True,
            "performance": {
                "async_mode": False,
                "batch_size": 10
            }
        }

        override_config = {
            "performance": {
                "async_mode": True,
                "timeout": 30
            }
        }

        result = merge_configs(base_config, override_config)

        assert result["enabled"] is True
        assert result["performance"]["async_mode"] is True  # 被覆盖
        assert result["performance"]["batch_size"] == 10  # 保持原值
        assert result["performance"]["timeout"] == 30  # 新增

    def test_merge_adapters_list(self):
        """测试合并adapters列表"""
        base_config = {
            "enabled": True,
            "adapters": [
                {"name": "lmcache", "type": "cache"}
            ]
        }

        override_config = {
            "adapters": [
                {"name": "vllm", "type": "inference"}
            ]
        }

        result = merge_configs(base_config, override_config)

        assert len(result["adapters"]) == 2
        adapter_names = {adapter["name"] for adapter in result["adapters"]}
        assert adapter_names == {"lmcache", "vllm"}


class TestValidateXConnectorConfig:
    """测试XConnector配置验证"""

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = {
            "enabled": True,
            "mode": "embedded",
            "adapters": [
                {
                    "name": "lmcache",
                    "type": "cache",
                    "config": {"storage_backend": "memory"}
                }
            ]
        }

        is_valid, errors = validate_xconnector_config(config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_enabled_field(self):
        """测试缺少enabled字段的配置"""
        config = {
            "mode": "embedded"
        }

        is_valid, errors = validate_xconnector_config(config)

        assert is_valid is False
        assert "Missing 'enabled' field" in errors

    def test_validate_invalid_enabled_type(self):
        """测试enabled字段类型错误"""
        config = {
            "enabled": "true"  # 应该是boolean
        }

        is_valid, errors = validate_xconnector_config(config)

        assert is_valid is False
        assert "'enabled' field must be boolean" in errors

    def test_validate_invalid_adapters_type(self):
        """测试adapters字段类型错误"""
        config = {
            "enabled": True,
            "adapters": "not_a_list"
        }

        is_valid, errors = validate_xconnector_config(config)

        assert is_valid is False
        assert "'adapters' must be a list" in errors

    def test_validate_invalid_adapter_structure(self):
        """测试适配器结构错误"""
        config = {
            "enabled": True,
            "adapters": [
                "not_a_dict",  # 应该是字典
                {"name": "lmcache"}  # 缺少type字段
            ]
        }

        is_valid, errors = validate_xconnector_config(config)

        assert is_valid is False
        assert "Adapter 0 must be a dictionary" in errors
        assert "Adapter 1 missing required field: type" in errors


class TestDetectXConnectorConfigMain:
    """测试主配置检测函数"""

    def test_config_priority_env_over_file(self, temp_config_dir, clean_env):
        """测试环境变量优先级高于文件配置"""
        # 创建文件配置
        config_file = temp_config_dir / "dynamo_config.yaml"
        file_config = {
            "xconnector": {
                "enabled": False,  # 文件中禁用
                "mode": "service"
            }
        }

        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(file_config, f)

        # 设置环境变量配置
        os.environ['ENABLE_XCONNECTOR'] = 'true'  # 环境变量中启用
        os.environ['XCONNECTOR_MODE'] = 'embedded'

        with patch('pathlib.Path.cwd', return_value=temp_config_dir):
            result = detect_xconnector_config()

        assert result is not None
        assert result["enabled"] is True  # 环境变量优先
        assert result["mode"] == "embedded"  # 环境变量优先

    def test_no_config_found(self, temp_config_dir, clean_env):
        """测试没有找到任何配置的情况"""
        # 额外确保清理所有可能的环境变量
        env_vars_to_check = [
            'XCONNECTOR_CONFIG', 'ENABLE_XCONNECTOR', 'XCONNECTOR_MODE',
            'XCONNECTOR_LOG_LEVEL', 'XCONNECTOR_GRACEFUL_DEGRADATION', 'ENABLE_LMCACHE'
        ]

        for var in env_vars_to_check:
            if var in os.environ:
                del os.environ[var]

        with patch('pathlib.Path.cwd', return_value=temp_config_dir):
            result = detect_xconnector_config()

        assert result is None

    def test_file_config_only(self, temp_config_dir, clean_env, mock_yaml_config):
        """测试只有文件配置的情况"""
        config_file = temp_config_dir / "dynamo_config.yaml"

        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_yaml_config, f)

        with patch('pathlib.Path.cwd', return_value=temp_config_dir):
            result = detect_xconnector_config()

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"
        assert len(result["adapters"]) == 1

    def test_env_config_only(self, clean_env):
        """测试只有环境变量配置的情况"""
        os.environ['ENABLE_XCONNECTOR'] = 'true'
        os.environ['XCONNECTOR_MODE'] = 'embedded'

        result = detect_xconnector_config()

        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == "embedded"