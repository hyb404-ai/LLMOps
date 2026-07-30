#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:43
"""
from injector import Injector
import dotenv

from module import ExtensionModule
from config import Config
from internal.router import Router
from internal.server import Http
from pkg.sqlalchemy import SQLAlchemy

# 将env加载到环境变量中
dotenv.load_dotenv()

config = Config()

injector = Injector([ExtensionModule])
app = Http(__name__,
           config=config,
           db=injector.get(SQLAlchemy),
           router=injector.get(Router))

if __name__ == "__main__":
    app.run(debug=True)
