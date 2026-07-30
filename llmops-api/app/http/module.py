#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module - 模块描述

作者: huangyoubin
创建日期: 2026/7/31 16:19
"""
from injector import Module, Binder

from internal.extension.database_extension import db
from pkg.sqlalchemy import SQLAlchemy


class ExtensionModule(Module):
    """扩展模块的依赖注入"""

    def configure(self, binder: Binder) -> None:
        binder.bind(SQLAlchemy, to=db)
