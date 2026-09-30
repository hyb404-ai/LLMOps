#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 11:52
@Author  : thezehui@gmail.com
@File    : 2.StructuredTool类方法创建工具.py

===================================================================================
知识点讲解：使用 StructuredTool 类方法创建自定义工具

1. 组件定位：自定义工具的三种构建方式（回顾）
   - LangChain 提供 @tool 装饰器、StructuredTool.from_function()、BaseTool 子类三种方式
   - 本文件讲第 2 种：StructuredTool.from_function()，比 @tool 提供更多配置项

2. StructuredTool 简介
   - 比 @tool 装饰器更灵活的工具创建方式
   - 支持同时定义同步和异步版本的工具（核心优势）
   - 提供更细粒度的配置控制，且无需修改原函数即可将其转换为工具

3. StructuredTool.from_function() 参数详解
   - func：同步执行函数，必需参数
   - coroutine：异步执行函数（可选），支持 async/await
   - name：工具名称（必需），建议使用描述性名称
   - description：工具描述（必需），帮助 LLM 理解工具用途
   - return_direct：是否直接返回结果（可选，默认 False）
   - args_schema：参数 Pydantic 模型（可选但强烈推荐）
   - handle_tool_error：错误处理策略（可选）

4. 同步与异步支持（核心优势）
   - func：处理同步调用 invoke()
   - coroutine：处理异步调用 ainvoke()
   - 可以只提供 func，也可以同时提供两者
   - 异步版本适用于 I/O 密集型操作（网络请求、数据库查询）
   - @tool 装饰器做不到同步/异步共存，这是两者关键区别

5. 与 @tool 装饰器的区别（三选一原则的边界）
   - @tool：适合简单工具，代码简洁，但不支持同步/异步共存
   - StructuredTool：适合需要同时支持同步/异步的工具，可将已有函数转换为工具且不修改原函数，配置更灵活
   - BaseTool 继承：适合复杂工具，需要维护状态

6. 典型输出示例与观察点
   - 配置 func + coroutine + args_schema 后，工具的 args 同样暴露带 description 的 JSON Schema：
        {'a': {'title': 'A', 'description': '第一个数字', 'type': 'integer'}, ...}
   - 与 @tool 不同，name / description 由参数显式传入，而不是取自函数名 / docstring
   - invoke({"a": 2, "b": 8}) 走 func，ainvoke(...) 走 coroutine，二者逻辑应保持一致

7. 参数模式的重要性
   - args_schema 使用 Pydantic 模型定义参数结构，提供类型验证和自动转换
   - 生成 JSON Schema 供 LLM 理解参数，Field() 的 description 是 LLM 理解参数的关键

8. 使用场景
   - 需要同时支持同步和异步调用的工具
   - 封装现有函数为工具，不希望修改原函数
   - 需要精细控制工具属性的场景、工具需要复杂配置的场景

9. 最佳实践
   - 同步和异步函数逻辑应保持一致
   - 明确指定 name 和 description
   - 始终提供 args_schema 以提高 LLM 调用准确性
   - 异步函数使用 async/await 语法，适用于 I/O 密集型操作避免阻塞

10. 代码量对比
    - 与 BaseTool 继承相比代码量相近，但无需定义类，更加轻量
    - 适合不需要维护状态的场景

===================================================================================
"""
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# 定义工具参数的 Pydantic 模型
# 这个模型既用于参数验证，也用于生成 JSON Schema 供 LLM 使用
class MultiplyInput(BaseModel):
    a: int = Field(description="第一个数字")  # 第一个乘数
    b: int = Field(description="第二个数字")  # 第二个乘数


# 同步版本的乘法函数
# 这个函数会被用于处理同步调用（invoke()）
def multiply(a: int, b: int) -> int:
    """将传递的两个数字相乘"""
    return a * b


# 异步版本的乘法函数
# 这个函数会被用于处理异步调用（ainvoke()）
# 在实际应用中，异步函数通常用于 I/O 密集型操作
async def amultiply(a: int, b: int) -> int:
    """将传递的两个数字相乘"""
    return a * b


# 使用 StructuredTool.from_function() 创建工具
# 参数说明：
#   - func：同步执行函数
#   - coroutine：异步执行函数（可选）
#   - name：工具名称，LLM 通过这个名称识别工具
#   - description：工具描述，帮助 LLM 理解何时使用该工具
#   - return_direct：True 表示工具结果直接返回，不经过 LLM 后处理
#   - args_schema：参数模式，使用 Pydantic 模型定义
calculator = StructuredTool.from_function(
    func=multiply,  # 同步函数
    coroutine=amultiply,  # 异步函数
    name="multiply_tool",  # 工具名称
    description="将传递的两个数字相乘",  # 工具描述
    return_direct=True,  # 直接返回结果
    args_schema=MultiplyInput,  # 参数模式
)

# 打印工具的相关信息
# 这些属性会被 LLM 用于选择和调用工具
print("名称: ", calculator.name)  # multiply_tool
print("描述: ", calculator.description)  # 将传递的两个数字相乘
print("参数: ", calculator.args)  # JSON Schema 格式的参数定义
print("直接返回: ", calculator.return_direct)  # True

# 调用工具
# invoke() 调用同步函数（multiply）
# ainvoke() 会调用异步函数（amultiply）
print(calculator.invoke({"a": 2, "b": 8}))  # 输出: 16
