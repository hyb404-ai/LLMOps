"""API 路由注册。"""

from flask import Blueprint, jsonify

api = Blueprint("api", __name__)


@api.get("/health")
def health_check():
    """服务健康检查。"""
    return jsonify({"status": "ok"})
