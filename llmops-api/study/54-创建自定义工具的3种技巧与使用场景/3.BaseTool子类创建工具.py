#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 12:25
@Author  : thezehui@gmail.com
@File    : 3.BaseTool子类创建工具.py

===================================================================================
知识点讲解：通过继承 BaseTool 创建自定义工具

1. 组件定位：所有工具都是 BaseTool 子类
   - 在 LangChain 中，所有工具都是 BaseTool 子类
   - 使用 @tool 或 StructuredTool.from_function() 创建的工具，底层都是包装成 StructuredTool，本质上也是 BaseTool 子类
   - 三种方式中 BaseTool 继承最灵活、最强大

2. BaseTool 继承方式简介
   - 最灵活、功能最强大的工具创建方式
   - 适合复杂工具，需要维护内部状态或复杂逻辑
   - 支持完全自定义工具行为

3. BaseTool 子类必须定义的属性
   - name：工具名称（类属性）
   - description：工具描述（类属性）
   - args_schema：参数模式，使用 Pydantic 模型（类属性，类型注解 Type[BaseModel]）
   - _run()：核心执行方法（必须实现）

4. _run() 方法详解
   - 工具的核心执行逻辑，接收 *args 和 **kwargs 参数
   - 参数会根据 args_schema 进行验证和转换
   - 通过 kwargs.get("参数名") 获取参数（建议用 kwargs 而非 args）
   - 必须返回字符串或可序列化的数据

5. 可选的方法和属性
   - _arun()：异步执行方法，支持 async/await
   - return_direct：是否直接返回结果（类属性，默认 False）
   - handle_tool_error：错误处理策略（类属性或方法）
   - __init__()：自定义初始化逻辑

6. 与其他方式的对比（三选一原则）
   - @tool 装饰器：最简单，适合无状态工具
   - StructuredTool：适中，支持同步/异步
   - BaseTool 继承：最灵活，适合复杂工具

7. BaseTool 继承的优势
   - 可在类中维护状态（如连接池、缓存）
   - 支持复杂初始化逻辑，可重写更多方法自定义行为
   - 适合封装为可复用的工具类，便于添加辅助方法

8. 典型输出示例与观察点
   - 子类实例化后，工具属性同样可被 LLM 读取：
        name = multiply_tool
        description = 将传递的两个数字相乘
        args = {'a': {'title': 'A', 'description': '第一个数字', 'type': 'integer'}, ...}
        return_direct = False
   - 与 @tool / StructuredTool 的唯一区别：name / description / args_schema 是「类属性」，_run 是实例方法
   - invoke({"a": 2, "b": 8}) 会调用 _run()，输出 16

9. 使用场景（何时选择 BaseTool）
   - 工具需要维护内部状态（如数据库连接）
   - 工具逻辑复杂，需要多个辅助方法
   - 需要自定义错误处理逻辑、需要在多个 Agent 中复用
   - 目前没有已实现的代码（从零开始设计）

10. 最佳实践
    - 在类文档字符串中描述工具用途
    - 使用类型注解提高可读性，合理处理异常返回有意义的错误信息
    - 如需异步支持，实现 _arun() 方法
    - 工具名称、描述、参数描述要清晰详细

11. BaseTool 的完整继承关系
    - Runnable（可运行组件基类）-> BaseTool（工具基类）-> StructuredTool（结构化工具）/ 自定义工具类

===================================================================================
"""
from typing import Any, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# 定义工具参数的 Pydantic 模型
class MultiplyInput(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")


# 通过继承 BaseTool 创建自定义工具类
# 这种方式适合需要维护状态或复杂逻辑的工具
class MultiplyTool(BaseTool):
    """乘法计算工具"""  # 类文档字符串，描述工具用途

    # 工具名称（必须定义）
    # LLM 通过这个名称识别和选择工具
    name = "multiply_tool"

    # 工具描述（必须定义）
    # 帮助 LLM 理解何时使用该工具
    description = "将传递的两个数字相乘后返回"

    # 参数模式（必须定义）
    # 使用 Type[BaseModel] 类型注解
    args_schema: Type[BaseModel] = MultiplyInput

    # 核心执行方法（必须实现）
    # 这个方法包含工具的实际逻辑
    # 参数说明：
    #   - *args：位置参数（通常不使用）
    #   - **kwargs：关键字参数，包含经过验证的工具参数
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """将传入的a和b相乘后返回"""
        # 从 kwargs 中获取参数
        # 参数已经根据 args_schema 进行了验证和类型转换
        return kwargs.get("a") * kwargs.get("b")

    # 可选：实现异步执行方法
    # async def _arun(self, *args: Any, **kwargs: Any) -> Any:
    #     """异步版本的执行方法"""
    #     return kwargs.get("a") * kwargs.get("b")


# 实例化工具
# BaseTool 子类需要实例化后才能使用
calculator = MultiplyTool()

# 打印工具的相关信息
print("名称: ", calculator.name)  # multiply_tool
print("描述: ", calculator.description)  # 将传递的两个数字相乘后返回
print("参数: ", calculator.args)  # JSON Schema 格式的参数定义
print("直接返回: ", calculator.return_direct)  # False（默认值）

# 调用工具
# invoke() 方法会调用 _run() 方法
# 参数以字典形式传入，键名对应 Pydantic 模型的字段名
print(calculator.invoke({"a": 2, "b": 8}))  # 输出: 16
