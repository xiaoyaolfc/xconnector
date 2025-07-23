# conftest.py 更新
# tests/conftest.py
import pytest
import logging
import asyncio
from unittest.mock import MagicMock
import torch


def pytest_configure(config):
    """配置 pytest，设置日志级别为 DEBUG 以便查看详细日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """设置日志捕获"""
    caplog.set_level(logging.DEBUG)
    yield


@pytest.fixture(scope="session")
def event_loop():
    """创建会话级别的事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_torch_tensors():
    """提供模拟的 torch 张量"""
    return {
        "kv_cache": torch.randn(2, 8, 64, 64),
        "hidden_states": torch.randn(1, 10, 768),
        "attention_mask": torch.ones(1, 10),
        "position_ids": torch.arange(10).unsqueeze(0)
    }


@pytest.fixture
def sample_requests():
    """提供示例请求数据"""
    return [
        {"id": "req_001", "prompt": "What is AI?", "tokens": [1, 2, 3, 4, 5]},
        {"id": "req_002", "prompt": "Hello world", "tokens": [6, 7, 8]},
        {"id": "req_003", "prompt": "How are you?", "tokens": [9, 10, 11, 12]},
    ]


# 运行脚本
# run_tests.py
# !/usr/bin/env python3
"""
测试运行脚本

提供不同级别的测试运行选项
"""
import subprocess
import sys
import argparse


def run_unit_tests():
    """运行单元测试"""
    cmd = ["pytest", "tests/unit/", "-m", "not slow", "-v"]
    return subprocess.run(cmd)


def run_integration_tests():
    """运行集成测试"""
    cmd = ["pytest", "tests/e2e/", "-m", "integration", "-v"]
    return subprocess.run(cmd)


def run_e2e_tests():
    """运行端到端测试"""
    cmd = ["pytest", "tests/e2e/", "-m", "e2e", "-v"]
    return subprocess.run(cmd)


def run_cache_tests():
    """运行缓存相关测试"""
    cmd = ["pytest", "-m", "cache", "-v"]
    return subprocess.run(cmd)


def run_inference_tests():
    """运行推理相关测试"""
    cmd = ["pytest", "tests/e2e/test_inference_request_flow.py", "-v"]
    return subprocess.run(cmd)


def run_full_tests():
    """运行完整测试套件"""
    cmd = ["pytest", "tests/e2e/test_full_integration.py", "-v"]
    return subprocess.run(cmd)


def run_all_tests():
    """运行所有测试"""
    cmd = ["pytest", "tests/", "-v"]
    return subprocess.run(cmd)


def run_coverage():
    """运行测试覆盖率"""
    cmd = [
        "pytest",
        "--cov=xconnector",
        "--cov-report=html",
        "--cov-report=term",
        "tests/"
    ]
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="XConnector Test Runner")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "e2e", "cache", "inference",
            "full", "all", "coverage"
        ],
        help="Type of tests to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    test_functions = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "e2e": run_e2e_tests,
        "cache": run_cache_tests,
        "inference": run_inference_tests,
        "full": run_full_tests,
        "all": run_all_tests,
        "coverage": run_coverage,
    }

    print(f"Running {args.test_type} tests...")
    result = test_functions[args.test_type]()

    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()