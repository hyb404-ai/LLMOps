"""API 路由注册。"""
from flask import Flask, Blueprint
from injector import inject
from dataclasses import dataclass

from internal.handler import AppHandler


@inject
@dataclass
class Router:
    """路由"""

    app_handler: AppHandler

    def __init__(self, app_handler: AppHandler):
        self.app_handler = app_handler

    def register_router(self, app: Flask):
        """注册路由"""
        # 1. 创建一个蓝图
        bp = Blueprint("llmops", __name__, url_prefix="/api")

        # 2. 将url与对应的控制器方法绑定
        bp.add_url_rule("/ping", view_func=self.app_handler.ping, methods=["GET"])
        bp.add_url_rule("/health", view_func=self.app_handler.health, methods=["GET"])

        # 3. 在应用上注册蓝图
        app.register_blueprint(bp)
