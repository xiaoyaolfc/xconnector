#!/usr/bin/env python3
"""
Deploy XConnector with AI-Dynamo

This script handles the deployment of XConnector integrated with AI-Dynamo
"""

import subprocess
import sys
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XConnectorDeployer:
    """Handle XConnector deployment with Dynamo"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "configs"
        self.dynamo_home = Path(os.environ.get("DYNAMO_HOME", ""))
        self.xconnector_home = Path(os.environ.get("XCONNECTOR_HOME", ""))

        self._validate_environment()

    def _validate_environment(self):
        """Validate required environment variables and paths"""
        if not self.dynamo_home.exists():
            raise RuntimeError(
                "DYNAMO_HOME not set or invalid. "
                "Please set DYNAMO_HOME to your ai-dynamo directory"
            )

        if not self.xconnector_home.exists():
            raise RuntimeError(
                "XCONNECTOR_HOME not set or invalid. "
                "Please set XCONNECTOR_HOME to your xconnector directory"
            )

        logger.info(f"DYNAMO_HOME: {self.dynamo_home}")
        logger.info(f"XCONNECTOR_HOME: {self.xconnector_home}")

    def install_xconnector(self):
        """Install XConnector package"""
        logger.info("Installing XConnector...")

        cmd = [sys.executable, "-m", "pip", "install", "-e", str(self.xconnector_home)]

        try:
            subprocess.run(cmd, check=True)
            logger.info("XConnector installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install XConnector: {e}")
            raise

    def prepare_configs(self, worker_config: str) -> Dict[str, Path]:
        """Prepare configuration files"""
        configs = {}

        # XConnector service config
        xconnector_config = self.config_dir / "xconnector_config.yaml"
        if not xconnector_config.exists():
            logger.error(f"XConnector config not found: {xconnector_config}")
            raise FileNotFoundError(xconnector_config)
        configs["xconnector"] = xconnector_config

        # Worker config
        if worker_config.startswith("/"):
            worker_config_path = Path(worker_config)
        else:
            worker_config_path = self.dynamo_home / "examples/vllm_v0/configs" / worker_config

        if not worker_config_path.exists():
            logger.error(f"Worker config not found: {worker_config_path}")
            raise FileNotFoundError(worker_config_path)

        # Create modified worker config
        modified_config = self._create_xconnector_worker_config(worker_config_path)
        configs["worker"] = modified_config

        return configs

    def _create_xconnector_worker_config(self, original_config: Path) -> Path:
        """Create worker config with XConnector enabled"""
        with open(original_config, 'r') as f:
            config = yaml.safe_load(f)

        # Add XConnector configuration
        if "VllmWorker" not in config:
            config["VllmWorker"] = {}

        config["VllmWorker"]["enable_xconnector"] = True

        # Additional XConnector settings
        config["VllmWorker"]["xconnector"] = {
            "adapters": ["vllm", "lmcache", "dynamo"],
            "routing_strategy": "least_loaded",
            "cache_coordination": True
        }

        # Save modified config
        modified_path = self.config_dir / f"{original_config.stem}_xconnector.yaml"
        with open(modified_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created modified config: {modified_path}")
        return modified_path

    def deploy_xconnector_service(self, config_path: Path):
        """Deploy XConnector as a Dynamo service"""
        logger.info("Deploying XConnector service...")

        # Change to Dynamo directory for proper module resolution
        os.chdir(self.dynamo_home)

        # Add XConnector to Python path
        sys.path.insert(0, str(self.xconnector_home))

        cmd = [
            "dynamo", "serve",
            "xconnector.integration.dynamo.xconnector_service:XConnectorService",
            "-f", str(config_path)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy XConnector service: {e}")
            raise

    def deploy_workers(self, config_path: Path, graph: str = "graphs.disagg:Frontend"):
        """Deploy workers with XConnector integration"""
        logger.info("Deploying workers with XConnector...")

        # Change to vllm_v0 example directory
        example_dir = self.dynamo_home / "examples/vllm_v0"
        os.chdir(example_dir)

        # Ensure XConnector is in Python path
        if str(self.xconnector_home) not in sys.path:
            sys.path.insert(0, str(self.xconnector_home))

        cmd = [
            "dynamo", "serve",
            graph,
            "-f", str(config_path)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy workers: {e}")
            raise

    def validate_deployment(self) -> bool:
        """Validate the deployment"""
        logger.info("Validating deployment...")

        checks = [
            ("XConnector Service", "http://localhost:8080/xconnector/get_status"),
            ("Worker Status", "http://localhost:8000/health"),
        ]

        import requests
        import time

        # Wait for services to start
        time.sleep(5)

        all_healthy = True
        for name, url in checks:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ {name} is healthy")
                else:
                    logger.error(f"✗ {name} returned status {response.status_code}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"✗ {name} check failed: {e}")
                all_healthy = False

        return all_healthy


def main():
    parser = argparse.ArgumentParser(
        description="Deploy XConnector with AI-Dynamo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy only XConnector service
  %(prog)s --mode service

  # Deploy workers with XConnector integration
  %(prog)s --mode workers --worker-config disagg.yaml

  # Full deployment (service + workers)
  %(prog)s --mode all --worker-config disagg.yaml

  # Deploy with custom graph
  %(prog)s --mode all --graph graphs.agg:Frontend --worker-config agg.yaml
        """
    )

    parser.add_argument(
        "--mode",
        choices=["service", "workers", "all"],
        default="all",
        help="Deployment mode"
    )

    parser.add_argument(
        "--worker-config",
        default="disagg.yaml",
        help="Worker configuration file (relative to dynamo examples/vllm_v0/configs/)"
    )

    parser.add_argument(
        "--graph",
        default="graphs.disagg:Frontend",
        help="Dynamo graph to deploy"
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Install XConnector package before deployment"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate deployment after completion"
    )

    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for deployment configs"
    )

    args = parser.parse_args()

    # Initialize deployer
    deployer = XConnectorDeployer(args.base_dir)

    try:
        # Install if requested
        if args.install:
            deployer.install_xconnector()

        # Prepare configurations
        configs = deployer.prepare_configs(args.worker_config)

        # Deploy based on mode
        if args.mode in ["service", "all"]:
            deployer.deploy_xconnector_service(configs["xconnector"])

        if args.mode in ["workers", "all"]:
            deployer.deploy_workers(configs["worker"], args.graph)

        # Validate if requested
        if args.validate:
            if deployer.validate_deployment():
                logger.info("✓ Deployment validation successful!")
            else:
                logger.error("✗ Deployment validation failed!")
                sys.exit(1)

        logger.info("Deployment completed successfully!")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()