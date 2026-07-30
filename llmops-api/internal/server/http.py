#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
http - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:39
"""
import os

from flask import Flask

from config import Config
from internal.exception import CustomException
from internal.router import Router
from pkg.response import json, Response, HttpCode
from pkg.sqlalchemy import SQLAlchemy


class Http(Flask):
    """http 服务引擎"""

    def __init__(self,
                 *args,
                 config: Config,
                 db: SQLAlchemy,
                 router: Router,
                 **kwargs):
        # 1.调用父类构造函数初始化
        super().__init__(*args, **kwargs)

        # 2.初始化应用配置
        self.config.from_object(config)

        # 3.注册绑定异常错误处理
        self.register_error_handler(Exception, self._register_error_handler)

        # 4.初始化flask扩展
        db.init_app(self)

        # 注册应用路由
        router.register_router(self)

    def _register_error_handler(self, error: Exception):
        # 1.异常信息是不是我们的自定义异常，如果是可以提取message和code等信息
        if isinstance(error, CustomException):
            return json(Response(
                code=error.code,
                message=error.message,
                data=error.data if error.data is not None else {},
            ))
        # 2.如果不是我们的自定义异常，则有可能是程序、数据库抛出的异常，也可以提取信息，设置为FAIL状态码
        if self.debug or os.getenv("FLASK_ENV") == "development":
            raise error
        else:
            return json(Response(
                code=HttpCode.FAIL,
                message=str(error),
                data={},
            ))
