#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
default_config - 模块描述

作者: huangyoubin
创建日期: 2026/7/31 16:08
"""
# 应用默认配置项
DEFAULT_CONFIG = {
    # wft配置
    "WTF_CSRF_ENABLED": "False",

    # SQLAlchemy数据库配置
    "SQLALCHEMY_DATABASE_URI": "",
    "SQLALCHEMY_POOL_SIZE": 30,
    "SQLALCHEMY_POOL_RECYCLE": 3600,
    "SQLALCHEMY_ECHO": "True",
}
