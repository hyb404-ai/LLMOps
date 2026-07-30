#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
http - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:39
"""
from flask import Flask
from internal.router import Router


class Http(Flask):
    """http 服务引擎"""

    def __init__(self, *args, router: Router, **kwargs):
        super().__init__(*args, **kwargs)
        # 注册应用路由
        router.register_router(self)
