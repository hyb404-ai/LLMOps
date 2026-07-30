"""配置选择入口。"""

from config.default_config import DefaultConfig


def get_config():
    """返回当前运行环境的配置类。"""
    return DefaultConfig
