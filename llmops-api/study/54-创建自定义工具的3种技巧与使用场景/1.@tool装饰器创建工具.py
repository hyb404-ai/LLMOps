#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 11:18
@Author  : thezehui@gmail.com
@File    : 1.@tool装饰器创建工具.py

===================================================================================
知识点讲解：使用 @tool 装饰器创建自定义工具

1. 组件定位：自定义工具的三种构建方式
   - 在 Agent / 函数调用场景中，需要向 LLM 提供工具列表；LangChain 内置工具不一定贴合业务，更多场合使用自定义工具
   - LangChain 提供 3 种构建自定义工具的技巧：@tool 装饰器、StructuredTool.from_function() 类方法、BaseTool 子类，各有优缺点与适用场景（本文件讲第 1 种）

2. @tool 装饰器简介
   - 定义自定义工具的最简单方式，可快速将函数改造成 LLM 工具
   - 默认使用函数名称作为工具名称，但可通过传递字符串作为第一个参数覆盖
   - 使用函数的文档字符串作为工具描述，因此被装饰的函数必须提供 docstring
   - 底层把函数包装成 StructuredTool 实例（StructuredTool 与 BaseTool 见后续文件）

3. @tool 装饰器参数详解
   - name：工具名称（可选，默认使用函数名）
   - description：工具描述（可选，默认使用函数 docstring）
   - return_direct：是否直接返回结果，不经过 LLM 后处理（默认 False）
   - args_schema：参数模式，使用 Pydantic 模型定义（可选但强烈推荐）
   - parse_docstring：是否解析 Google-style 文档字符串（默认 False）

4. Pydantic 参数模式的作用
   - 明确定义每个参数的类型和描述
   - Field() 的 description 会被 LLM 用于理解参数含义，是 LLM 正确生成参数的关键
   - 提供参数验证和类型转换，并生成标准 JSON Schema 供 LLM 使用

5. 函数签名要求
   - 参数类型提示（type hints）必须提供
   - 返回值类型提示推荐提供
   - 函数文档字符串用作工具描述，被装饰函数必须提供 docstring

6. return_direct 参数详解
   - True：工具结果直接返回给用户，不经过 LLM 加工
   - False：工具结果返回给 LLM，由 LLM 整合后输出
   - 适用场景：计算器、数据库查询等确定性结果可设为 True

7. parse_docstring 参数的作用
   - 设为 True 可解析 Google-style 文档字符串，自动提取 Args 部分的参数描述，无需单独定义 args_schema
   - 示例：
       @tool(parse_docstring=True)
       def foo(bar: str, baz: int) -> str:
           '''The foo.

           Args:
               bar: The bar.
               baz: The baz.
           '''

8. @tool 装饰器的限制
   - 不能同时装饰同步和异步版本，只可单独装饰：
       @tool
       async def amultiply(a: int, b: int) -> int:
           '''Multiply two numbers.'''
           return a * b
   - 若需同时支持同步和异步，使用 StructuredTool.from_function()

9. 工具调用方式与属性
   - invoke()：同步调用，传入字典格式的参数（键名对应 Pydantic 模型字段名）
   - ainvoke()：异步调用（需要函数支持 async）
   - 工具属性 name / description / args（JSON Schema）/ return_direct 是 LLM 选择和调用工具的依据

10. 典型输出示例与观察点
    - 未传 args_schema 时，参数只有类型无 description：
        {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
    - 传入 args_schema 后，参数带上 description，LLM 才能正确理解含义：
        {'a': {'title': 'A', 'description': '第一个数字', 'type': 'integer'}, ...}
    - 观察点：args 即工具暴露给 LLM 的 JSON Schema，description 的有无直接影响 LLM 调用准确率

11. 使用场景（三选一原则）
    - 快速原型开发：将现有函数快速转换为工具
    - 简单工具：单一功能的工具（计算、转换、查询）
    - 无状态工具：不需要保存实例状态的工具
    - 参数相对简单的函数

12. 最佳实践
    - 始终提供清晰的函数文档字符串
    - 使用 Pydantic 模型明确参数模式（强烈推荐）
    - 参数描述要详细，帮助 LLM 正确理解
    - 函数应具有单一职责，工具名称避免与内置工具冲突

===================================================================================
"""
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# 定义工具参数的 Pydantic 模型
# 这个模型会被转换为 JSON Schema，供 LLM 理解参数结构
class MultiplyInput(BaseModel):
    # Field() 的 description 参数非常重要，它会告诉 LLM 这个参数的含义
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")


# 使用 @tool 装饰器创建工具
# 参数说明：
#   - "multiply_tool"：工具名称，LLM 通过这个名称识别工具
#   - return_direct=True：工具结果直接返回，不经过 LLM 后处理
#   - args_schema=MultiplyInput：使用 Pydantic 模型定义参数模式
@tool("multiply_tool", return_direct=True, args_schema=MultiplyInput)
def multiply(a: int, b: int) -> int:
    """将传递的两个数字相乘"""  # 这个文档字符串会作为工具的描述
    return a * b


# 打印工具的相关信息
# 这些属性是 LLM 选择和调用工具的依据
print("名称: ", multiply.name)  # 工具的唯一标识符
print("描述: ", multiply.description)  # 工具的功能描述（来自 docstring）
print("参数: ", multiply.args)  # 工具的参数 JSON Schema
print("直接返回: ", multiply.return_direct)  # 是否跳过 LLM 后处理

# 调用工具
# invoke() 方法接收字典格式的参数，键名对应 Pydantic 模型的字段名
print(multiply.invoke({"a": 2, "b": 8}))  # 输出: 16
