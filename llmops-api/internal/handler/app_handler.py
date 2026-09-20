#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AppHandler - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:14
"""
import os
import uuid
from dataclasses import dataclass

from flask import request
from injector import inject
from openai import OpenAI

from internal.schema import CompleteForm
from internal.service import AppService
from pkg.response import success_json, validate_error_json, success_message

@inject
@dataclass
class AppHandler:
    """应用控制器"""

    app_service: AppService

    def create_app(self):
        """调用服务创建新的APP记录"""
        app = self.app_service.create_app()
        return success_message(f"应用已经成功创建，id为{app.id}")

    def get_app(self, id: uuid.UUID):
        print(f"get_app:{id}")
        app = self.app_service.get_app(id)
        return success_message(f"应用已经成功获取，名字是{app.name}")

    def update_app(self, id: uuid.UUID):
        app = self.app_service.update_app(id)
        return success_message(f"应用已经成功修改，修改的名字是:{app.name}")

    def delete_app(self, id: uuid.UUID):
        app = self.app_service.delete_app(id)
        return success_message(f"应用已经成功删除，id为:{app.id}")

    def completion(self):
        """聊天接口"""

        # 1. 提取从接口中获取的输入
        req = CompleteForm()
        if not req.validate():
            return validate_error_json(req.errors)

        query = request.json.get("query")

        # 2. 构建OPENAI客户端，并发起请求
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

        # 3. 得到请求响应，人后将OpenAI的响应传递给前端
        completion = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是 MiniMax 开发的聊天机器人，请根据用户的输入回复对应的信息"},
                {"role": "user", "content": query}
            ]
        )
        content = completion.choices[0].message.content
        # resp = Response(code=HttpCode.SUCCESS, message="", data={"content": content})
        # return jsonify(resp), 200
        return success_json(data={"content": content})

    def ping(self):
        return {"ping": "pong"}

    def health(self):
        return {"health": "health"}
