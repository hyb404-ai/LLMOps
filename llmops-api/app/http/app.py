#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 14:43
"""
from injector import Injector

from internal.router import Router
from internal.server import Http

injector = Injector()
app = Http(__name__, router=injector.get(Router))

if __name__ == "__main__":
    app.run(debug=True)
