#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AppHandler - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:14
"""


class AppHandler:
    """应用控制器"""

    def ping(self):
        return {"ping": "pong"}

    def health(self):
        return {"health": "health"}
