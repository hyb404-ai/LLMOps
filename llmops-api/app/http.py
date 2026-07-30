"""HTTP 应用工厂。"""

from flask import Flask

from config.config import get_config
from internal.router.router import api


def create_app() -> Flask:
    """创建并配置 Flask 应用。"""
    app = Flask(__name__)
    app.config.from_object(get_config())
    app.register_blueprint(api, url_prefix="/api/v1")
    return app
