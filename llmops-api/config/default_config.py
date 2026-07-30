"""默认配置；敏感信息通过环境变量覆盖。"""

import os


class DefaultConfig:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
