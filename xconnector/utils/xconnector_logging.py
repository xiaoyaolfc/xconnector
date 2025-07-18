# xconnector/utils/logging.py
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import threading

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 全局日志配置
_logger_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': 'xconnector/log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'console_output': True,
    'file_output': True
}

# 线程锁，确保日志文件创建的线程安全
_lock = threading.Lock()

# 已创建的日志器缓存
_loggers: Dict[str, logging.Logger] = {}


def setup_logging_config(config: Optional[Dict[str, Any]] = None) -> None:
    """
    设置全局日志配置

    Args:
        config: 日志配置字典
    """
    global _logger_config

    if config:
        _logger_config.update(config)

    # 确保日志目录存在
    log_dir = Path(_logger_config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)


def _generate_log_filename(logger_name: str) -> str:
    """
    生成唯一的日志文件名

    Args:
        logger_name: 日志器名称

    Returns:
        str: 日志文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将模块名中的点替换为下划线，避免文件名问题
    safe_name = logger_name.replace('.', '_').replace('/', '_')
    return f"{safe_name}_{timestamp}.log"


def _create_file_handler(logger_name: str) -> logging.Handler:
    """
    创建文件处理器

    Args:
        logger_name: 日志器名称

    Returns:
        logging.Handler: 文件处理器
    """
    log_dir = Path(_logger_config['log_dir'])
    log_filename = _generate_log_filename(logger_name)
    log_filepath = log_dir / log_filename

    # 使用RotatingFileHandler支持日志轮转
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_filepath),
        maxBytes=_logger_config['max_file_size'],
        backupCount=_logger_config['backup_count'],
        encoding='utf-8'
    )

    # 设置格式
    formatter = logging.Formatter(
        fmt=_logger_config['format'],
        datefmt=_logger_config['date_format']
    )
    file_handler.setFormatter(formatter)

    return file_handler


def _create_console_handler() -> logging.Handler:
    """
    创建控制台处理器

    Returns:
        logging.Handler: 控制台处理器
    """
    console_handler = logging.StreamHandler(sys.stdout)

    # 设置格式
    formatter = logging.Formatter(
        fmt=_logger_config['format'],
        datefmt=_logger_config['date_format']
    )
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    获取日志器实例

    Args:
        name: 日志器名称，通常使用 __name__
        level: 日志级别，如果为None则使用全局配置

    Returns:
        logging.Logger: 日志器实例
    """
    with _lock:
        # 如果已经创建过，直接返回
        if name in _loggers:
            return _loggers[name]

        # 创建新的日志器
        logger = logging.getLogger(name)

        # 设置日志级别
        log_level = level or _logger_config['level']
        logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))

        # 防止重复添加处理器
        if not logger.handlers:
            # 添加控制台处理器
            if _logger_config['console_output']:
                console_handler = _create_console_handler()
                logger.addHandler(console_handler)

            # 添加文件处理器
            if _logger_config['file_output']:
                try:
                    file_handler = _create_file_handler(name)
                    logger.addHandler(file_handler)
                except Exception as e:
                    # 如果文件处理器创建失败，至少保证控制台输出
                    logger.error(f"Failed to create file handler: {e}")

        # 防止日志传播到根日志器
        logger.propagate = False

        # 缓存日志器
        _loggers[name] = logger

        return logger


def create_logger(name: str,
                  level: Optional[str] = None,
                  console_output: bool = True,
                  file_output: bool = True,
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    创建自定义日志器

    Args:
        name: 日志器名称
        level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        log_file: 指定日志文件名（可选）

    Returns:
        logging.Logger: 日志器实例
    """
    logger = logging.getLogger(name)

    # 设置日志级别
    log_level = level or _logger_config['level']
    logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))

    # 清除已有的处理器
    logger.handlers.clear()

    # 添加控制台处理器
    if console_output:
        console_handler = _create_console_handler()
        logger.addHandler(console_handler)

    # 添加文件处理器
    if file_output:
        try:
            if log_file:
                log_dir = Path(_logger_config['log_dir'])
                log_dir.mkdir(parents=True, exist_ok=True)
                log_filepath = log_dir / log_file

                file_handler = logging.handlers.RotatingFileHandler(
                    filename=str(log_filepath),
                    maxBytes=_logger_config['max_file_size'],
                    backupCount=_logger_config['backup_count'],
                    encoding='utf-8'
                )
            else:
                file_handler = _create_file_handler(name)

            formatter = logging.Formatter(
                fmt=_logger_config['format'],
                datefmt=_logger_config['date_format']
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.error(f"Failed to create file handler: {e}")

    # 防止日志传播到根日志器
    logger.propagate = False

    return logger


def set_log_level(logger_name: str, level: str) -> None:
    """
    设置指定日志器的日志级别

    Args:
        logger_name: 日志器名称
        level: 日志级别
    """
    if logger_name in _loggers:
        logger = _loggers[logger_name]
        logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))


def list_loggers() -> Dict[str, str]:
    """
    列出所有已创建的日志器及其级别

    Returns:
        Dict[str, str]: 日志器名称到级别的映射
    """
    return {
        name: logging.getLevelName(logger.level)
        for name, logger in _loggers.items()
    }


def get_log_files() -> list:
    """
    获取所有日志文件列表

    Returns:
        list: 日志文件路径列表
    """
    log_dir = Path(_logger_config['log_dir'])
    if not log_dir.exists():
        return []

    log_files = []
    for file_path in log_dir.glob("*.log*"):
        log_files.append(str(file_path))

    return sorted(log_files)


def cleanup_old_logs(keep_days: int = 7) -> None:
    """
    清理旧的日志文件

    Args:
        keep_days: 保留天数
    """
    log_dir = Path(_logger_config['log_dir'])
    if not log_dir.exists():
        return

    cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

    for file_path in log_dir.glob("*.log*"):
        try:
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                print(f"Deleted old log file: {file_path}")
        except Exception as e:
            print(f"Failed to delete log file {file_path}: {e}")


def configure_uvicorn_logging() -> None:
    """
    配置Uvicorn日志器，使其与XConnector日志系统一致
    """
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")

    # 设置日志级别
    uvicorn_logger.setLevel(LOG_LEVELS.get(_logger_config['level'].upper(), logging.INFO))
    uvicorn_access_logger.setLevel(LOG_LEVELS.get(_logger_config['level'].upper(), logging.INFO))

    # 创建文件处理器
    if _logger_config['file_output']:
        try:
            uvicorn_file_handler = _create_file_handler("uvicorn")
            uvicorn_logger.addHandler(uvicorn_file_handler)

            uvicorn_access_file_handler = _create_file_handler("uvicorn.access")
            uvicorn_access_logger.addHandler(uvicorn_access_file_handler)
        except Exception as e:
            print(f"Failed to setup uvicorn file logging: {e}")


# 初始化日志系统
def initialize_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    初始化日志系统

    Args:
        config: 日志配置字典
    """
    setup_logging_config(config)

    # 创建日志目录
    log_dir = Path(_logger_config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # 根日志器设置为WARNING，避免第三方库的冗余日志

    print(f"XConnector logging initialized. Log directory: {log_dir.absolute()}")


# 在模块加载时自动初始化
initialize_logging()


# 便捷函数
def debug(msg: str, logger_name: str = "xconnector") -> None:
    """记录DEBUG级别日志"""
    get_logger(logger_name).debug(msg)


def info(msg: str, logger_name: str = "xconnector") -> None:
    """记录INFO级别日志"""
    get_logger(logger_name).info(msg)


def warning(msg: str, logger_name: str = "xconnector") -> None:
    """记录WARNING级别日志"""
    get_logger(logger_name).warning(msg)


def error(msg: str, logger_name: str = "xconnector") -> None:
    """记录ERROR级别日志"""
    get_logger(logger_name).error(msg)


def critical(msg: str, logger_name: str = "xconnector") -> None:
    """记录CRITICAL级别日志"""
    get_logger(logger_name).critical(msg)


# 使用示例

# if __name__ == "__main__":
#     # 测试日志系统
#     logger = get_logger(__name__)
#
#     logger.debug("This is a debug message")
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")
#     logger.critical("This is a critical message")
#
#     # 自定义日志器
#     custom_logger = create_logger("custom", level="DEBUG")
#     custom_logger.info("Custom logger message")
#
#     # 列出所有日志器
#     print("Available loggers:", list_loggers())
#
#     # 列出日志文件
#     print("Log files:", get_log_files())