# setup.py
"""
XConnector SDK Python包配置
"""

from setuptools import setup, find_packages
import os


# 读取README文件
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "XConnector SDK for distributed inference systems"


# 读取版本号
def get_version():
    version_file = os.path.join("xconnector", "__version__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            exec(f.read())
            return locals()["__version__"]
    return "1.0.0"


setup(
    name="xconnector",
    version=get_version(),
    author="xiaoyaolfc",
    author_email="your-email@example.com",
    description="XConnector SDK for distributed inference systems",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/xconnector",

    # 包配置
    packages=find_packages(),
    include_package_data=True,

    # Python版本要求
    python_requires=">=3.8",

    # 核心依赖
    install_requires=[
        "torch>=1.12.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.0",
        "asyncio-mqtt>=0.10.0",  # 如果需要MQTT
    ],

    # 可选依赖
    extras_require={
        "lmcache": ["lmcache>=0.1.0"],
        "vllm": ["vllm>=0.2.0"],
        "dynamo": ["ai-dynamo>=0.1.0"],
        "redis": ["redis>=4.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "all": [
            "lmcache>=0.1.0",
            "vllm>=0.2.0",
            "redis>=4.0.0",
        ]
    },

    # 入口点
    entry_points={
        "console_scripts": [
            "xconnector=xconnector.cli:main",
        ],
        # 插件入口点
        "xconnector.adapters": [
            "lmcache=xconnector.adapters.cache.lmcache_adapter:LMCacheAdapter",
            "vllm=xconnector.adapters.inference.vllm_adapter:VLLMAdapter",
            "dynamo=xconnector.adapters.distributed.dynamo_adapter:DynamoAdapter",
        ],
    },

    # 分类标签
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # 项目URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/xconnector/issues",
        "Source": "https://github.com/your-org/xconnector",
        "Documentation": "https://xconnector.readthedocs.io/",
    },
)

# pyproject.toml (现代Python包配置)
"""
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "xconnector"
description = "XConnector SDK for distributed inference systems"
authors = [{name = "xiaoyaolfc", email = "your-email@example.com"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
]
dynamic = ["version"]

dependencies = [
    "torch>=1.12.0",
    "psutil>=5.8.0", 
    "pyyaml>=5.4.0",
]

[project.optional-dependencies]
lmcache = ["lmcache>=0.1.0"]
vllm = ["vllm>=0.2.0"]
dynamo = ["ai-dynamo>=0.1.0"]
all = ["lmcache>=0.1.0", "vllm>=0.2.0"]

[project.entry-points."xconnector.adapters"]
lmcache = "xconnector.adapters.cache.lmcache_adapter:LMCacheAdapter"
vllm = "xconnector.adapters.inference.vllm_adapter:VLLMAdapter"

[tool.setuptools_scm]
write_to = "xconnector/__version__.py"
"""

# xconnector/__version__.py (自动生成)
__version__ = "1.0.0"

# MANIFEST.in (包含额外文件)
"""
include README.md
include LICENSE
include xconnector/configs/*.yaml
include xconnector/configs/*.json
recursive-include xconnector *.py
recursive-include xconnector *.yaml
recursive-include xconnector *.json
global-exclude *.pyc
global-exclude __pycache__
"""

# requirements.txt (开发依赖)
"""
# 核心依赖
torch>=1.12.0
psutil>=5.8.0
pyyaml>=5.4.0

# 可选依赖 - 根据需要安装
# lmcache>=0.1.0
# vllm>=0.2.0
# redis>=4.0.0

# 开发依赖
pytest>=7.0.0
pytest-asyncio>=0.20.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.991
"""