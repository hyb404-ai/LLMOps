# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
injector - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 12:27
"""
from injector import Injector, inject


class A:
    def __init__(self, name: str):
        self.name = name


@inject
class B:
    def __init__(self, a: A):
        self.a = a

    def print(self):
        print(f"name: {self.a.name}")


def configure(binder):
    """为需要运行时参数的依赖提供具体实例。"""
    binder.bind(A, to=A(name="llmops"))


def child_configure(binder):
    binder.bind(A, to=A(name="child-llmops"))


if __name__ == "__main__":
    parent_injector = Injector(configure)
    injector = Injector(modules=child_configure, parent=parent_injector)
    b = injector.get(B)
    b.print()
