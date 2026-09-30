#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/12 9:51
@Author  : thezehui@gmail.com
@File    : 1.错误捕获.py

===================================================================================
知识点讲解：工具调用出错捕获（try-except 策略）
===================================================================================

1. 为什么要处理函数调用错误
   - 执行函数调用时错误几乎不可避免：LLM 可能传错/漏传参数，工具内部也可能抛错
   - 若不处理，程序会非常脆弱，一次异常就中断整个链
   - 在链中构建错误处理/捕获，可显著降低故障概率、提升健壮性

2. 工具调用常见错误类型
   - 参数缺失：LLM 未生成必需参数（如本例的 dict_arg）
   - 参数类型错误：类型不匹配（字符串传给 int 等）
   - 参数验证失败：Pydantic 校验不通过
   - 工具执行异常：网络错误、API 错误、业务逻辑错误
   - 工具不存在：LLM 生成了不存在的工具名

3. 本示例的失败场景
   - complex_tool 定义了三个参数：int_arg、float_arg、dict_arg，返回 int_arg * float_arg
   - 用户查询只提供两个值 "5 和 2.1"，LLM 只生成了前两个参数
   - 缺失 dict_arg，Pydantic 校验失败，抛出 ValidationError：
       pydantic.v1.error_wrappers.ValidationError: 1 validation error for complex_toolSchema
       dict_arg
         field required (type=value_error.missing)

4. try-except 策略的实现
   - 用 try 包裹工具调用（complex_tool.invoke(tool_args, config)）
   - 捕获 Exception，返回一段描述性错误信息
   - 错误信息包含三要素：实际传入的参数、异常类型 type(e)、异常消息 str(e)
   - 返回的错误串形如：
       "调用工具时使用以下参数:\n\n{tool_args}\n\n引发了以下错误:\n\n{type(e)}: {e}"

5. RunnableConfig 的作用
   - LangChain 运行时配置对象，含 callbacks、metadata、tags 等
   - 透传给工具的 invoke(config=config)，保持回调与追踪链路完整
   - 支持 LangSmith 等可观测性工具

6. 链的组成
   - llm_with_tools：绑定工具的 LLM，生成 tool_calls
   - lambda msg: msg.tool_calls[0]["args"]：提取第一个工具调用的参数字典
   - try_except_tool：安全执行工具并捕获异常
   - 组合：llm_with_tools | 提取参数 | try_except_tool

7. 错误信息的价值：让 LLM 自我纠正（最推荐）
   - 本策略最推荐的做法是「在工具内部捕获错误，并独立返回错误信息」
   - 关键在于：把返回的错误消息重新喂回 LLM，LLM 会重新判断并继续函数调用，
     自动补全缺失参数（如补上 dict_arg），让程序正常执行
   - 因此 try-except 并非只能「被动报错」，配合 LLM 重试即可转化为自我纠正

8. 策略优缺点
   - 优点：实现简单、代码清晰；不中断链执行；错误信息可返回用户或 LLM
   - 缺点：单独使用时不会自动重试/纠正；用户可能先看到错误提示

9. 适用场景与最佳实践
   - 适用：需要记录错误但暂不自动恢复、错误信息需展示给用户、作为其他策略的兜底、调试阶段定位问题
   - 实践：生产环境结合日志记录；错误信息不要暴露敏感信息；
     区分不同异常类型分别处理；配合重试或回退策略提高成功率

===================================================================================
"""
from typing import Any

import dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 定义一个"复杂工具"，需要三个参数
# 注意：这个工具故意设计了容易出错的场景
#   - 需要三个参数，但用户查询可能只提供两个
#   - 函数体只使用了前两个参数（dict_arg 未使用）
#   - 返回值类型标注为 int，但实际返回 float
@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg


# 安全的工具调用包装函数
# 参数说明：
#   - tool_args：工具参数字典，来自 LLM 生成的 tool_calls
#   - config：LangChain 运行时配置，保持链路追踪完整
def try_except_tool(tool_args: dict, config: RunnableConfig) -> Any:
    try:
        # 尝试执行工具
        # 传递 config 参数以保持回调和追踪链路
        return complex_tool.invoke(tool_args, config=config)
    except Exception as e:
        # 捕获所有异常，返回描述性错误信息
        # 错误信息包含三个关键要素：
        #   1. 实际传入的参数（便于调试和 LLM 纠正）
        #   2. 异常类型（type(e)）
        #   3. 异常消息（e 的字符串表示）
        return f"调用工具时使用以下参数:\n\n{tool_args}\n\n引发了以下错误:\n\n{type(e)}: {e}"


# 1.创建大语言模型并绑定工具
# temperature=0 提高输出的确定性
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)
# bind_tools() 将工具绑定到 LLM，使其能够生成 tool_calls
llm_with_tools = llm.bind_tools([complex_tool])

# 2.创建链并执行工具
# 链的执行流程：
#   步骤1：llm_with_tools 生成包含 tool_calls 的 AIMessage
#   步骤2：lambda 提取第一个 tool_call 的 args（参数字典）
#   步骤3：try_except_tool 安全执行工具，捕获可能的异常
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool

# 3.调用链
# 注意：用户查询只提供了 5 和 2.1 两个值
# 但 complex_tool 需要三个参数（缺少 dict_arg）
# 预期结果：工具调用失败，返回错误信息而非中断程序
print(chain.invoke("使用复杂工具，对应参数为5和2.1"))
