# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
injector - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 12:27

===================================================================================
知识点讲解：Injector 依赖注入框架
===================================================================================

1. 依赖注入（Dependency Injection，DI）简介
   - 一种设计模式，将对象的依赖关系外部化
   - 对象不主动创建依赖，而是由外部容器注入
   - 核心目的：降低耦合、提升可测试性、便于扩展

2. 为什么需要依赖注入
   - 不使用 DI：类内部 new 依赖对象，硬编码耦合
   - 使用 DI：依赖由容器管理，可灵活替换实现
   - 单元测试时可以注入 Mock 对象
   - 修改依赖实现无需改动业务代码

3. Injector 库简介
   - Python 的依赖注入框架，类似 Java 的 Guice
   - 基于类型注解（type hints）自动解析依赖
   - 支持模块化配置、作用域管理、父子容器

4. @inject 装饰器详解
   - 标记类或函数需要依赖注入
   - Injector 会读取 __init__ 的类型注解
   - 自动解析并注入对应类型的实例
   - 本示例中 B 依赖 A，通过类型注解 a: A 声明

5. Injector 容器
   - Injector(modules=...)：创建注入容器
   - get(类型)：从容器获取指定类型的实例
   - 自动递归解析依赖链
   - 默认单例作用域（同一容器多次 get 返回同一实例）

6. binder 与绑定配置
   - configure(binder) 函数用于配置依赖绑定
   - binder.bind(接口/类型, to=具体实现)
   - 绑定方式：
     * to=实例：绑定到具体实例（本示例）
     * to=类：绑定到类（由容器实例化）
     * to=工厂函数：绑定到工厂方法

7. 为什么需要手动绑定 A
   - A 的 __init__ 需要 name 参数（运行时参数）
   - Injector 无法自动推断 name 的值
   - 必须通过 binder.bind(A, to=A(name="llmops")) 提供实例
   - 对比：B 只依赖类型 A，可以自动注入

8. 父子容器（Parent-Child Injector）
   - Injector(modules=..., parent=父容器)
   - 子容器优先使用自己的绑定
   - 找不到时向父容器查找
   - 应用场景：
     * 全局配置在父容器，请求级配置在子容器
     * 多租户场景：不同租户不同配置
     * 测试时覆盖部分依赖

9. 本示例的绑定覆盖分析
   - 父容器绑定：A(name="llmops")
   - 子容器绑定：A(name="child-llmops")
   - injector.get(B) 时，B 依赖的 A 从子容器获取
   - 输出结果：name: child-llmops（子容器覆盖父容器）

10. 在 LLMOps 项目中的应用
    - Flask 应用中管理 Service、Repository 等组件
    - 数据库连接、LLM 客户端等资源的统一管理
    - 配置对象的注入
    - 便于单元测试时替换真实依赖

11. 作用域（Scope）
    - NoScope（默认）：每次 get 创建新实例
    - SingletonScope：容器内单例
    - ThreadLocalScope：线程内单例
    - RequestScope：请求内单例（Web 应用）

12. 最佳实践
    - 优先使用构造函数注入（而非属性注入）
    - 使用类型注解明确依赖关系
    - 需要运行时参数的依赖通过 binder 手动绑定
    - 模块化组织绑定配置（按功能拆分 configure 函数）
    - 避免在类内部直接 new 依赖对象
    - 注意避免循环依赖

13. 常见陷阱
    - 类型注解缺失导致注入失败
    - 循环依赖导致递归错误
    - 忘记加 @inject 装饰器
    - 局部模块名与第三方包同名导致导入冲突
      （本文件命名为 injector_demo.py 而非 injector.py）

===================================================================================
"""
from injector import Injector, inject


# 被依赖的类
# 注意：__init__ 需要 name 参数（运行时参数）
# 因此不能被 Injector 自动实例化，必须通过 binder 手动绑定
class A:
    def __init__(self, name: str):
        self.name = name


# 依赖 A 的类
# @inject 装饰器标记该类需要依赖注入
# Injector 会读取 __init__ 的类型注解（a: A），自动注入 A 的实例
@inject
class B:
    def __init__(self, a: A):
        # a 参数由 Injector 自动注入，无需手动传递
        self.a = a

    def print(self):
        # 访问注入的依赖对象
        print(f"name: {self.a.name}")


# 父容器的绑定配置函数
# binder 参数由 Injector 提供，用于注册依赖绑定
def configure(binder):
    """为需要运行时参数的依赖提供具体实例。"""
    # bind(类型, to=实例) 将类型 A 绑定到具体实例
    # 因为 A 需要 name 参数，无法自动实例化，所以手动创建实例
    binder.bind(A, to=A(name="llmops"))


# 子容器的绑定配置函数
# 覆盖父容器中 A 的绑定
def child_configure(binder):
    # 子容器使用不同的 A 实例（name 不同）
    binder.bind(A, to=A(name="child-llmops"))


if __name__ == "__main__":
    # 创建父容器
    # modules 参数接收配置函数（或函数列表/Module 类）
    parent_injector = Injector(modules=configure)

    # 创建子容器，指定 parent 为父容器
    # 依赖查找顺序：先查子容器，找不到再查父容器
    # 这里子容器覆盖了 A 的绑定
    injector = Injector(modules=child_configure, parent=parent_injector)

    # 从容器获取 B 的实例
    # 执行流程：
    #   1. Injector 检测到 B 有 @inject 装饰器
    #   2. 读取 B.__init__ 的类型注解，发现需要 A 类型的实例
    #   3. 在子容器中查找 A 的绑定，找到 A(name="child-llmops")
    #   4. 将 A 实例注入到 B 的构造函数
    #   5. 返回构造完成的 B 实例
    b = injector.get(B)

    # 输出：name: child-llmops
    # 验证子容器的绑定覆盖了父容器的绑定
    b.print()
