#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/12 10:24
@Author  : thezehui@gmail.com
@File    : 3.携带错误信息的重试.py

===================================================================================
知识点讲解：携带错误信息的重试（自我纠正链）
===================================================================================

1. 自我纠正机制简介
   - 最智能的错误处理策略：把错误信息反馈给 LLM，让其自主纠正
   - 模拟人类「试错—学习—改进」，显著提高工具调用成功率

2. 核心实现思路
   - 首次调用工具、捕获异常
   - 将异常信息包装为消息（AIMessage + ToolMessage + HumanMessage）
   - 把错误消息追加到提示词中
   - LLM 看到错误后生成修正参数，重新调用工具

3. 自定义异常类 CustomToolException
   - 继承 Exception，携带完整上下文
   - tool_call：原始工具调用请求（含 id、name、args）
   - exception：原始异常对象
   - 目的：在 fallback 中获取完整错误上下文

4. exception_key 参数
   - with_fallbacks() 的关键参数，将捕获的异常对象注入 fallback 链输入
   - 输入变为 {原始输入..., exception_key: 异常对象}，使 fallback 能访问异常

5. exception_to_messages() 函数（核心）
   - 从输入中提取异常（inputs.pop("exception")，避免影响提示词渲染）
   - 构造三条消息模拟「一次失败的工具调用」：
     * AIMessage(content="", tool_calls=[...])：重现 LLM 之前的错误请求
     * ToolMessage(tool_call_id=..., content=str(exception))：工具返回的错误响应
     * HumanMessage：提示「请纠正并重试，不要重复犯错」
   - 文档注释点明：这些历史消息让模型知道自己上次犯了错
   - 将消息列表赋给 last_output 字段

6. placeholder 消息占位符
   - ChatPromptTemplate 中的 ("placeholder", "{last_output}") 用于动态插入消息列表
   - 首次调用时为空（无错误信息）；重试时包含错误消息序列

7. 消息序列设计原理
   - AIMessage（含 tool_calls）：告诉 LLM「你之前这样调用了」
   - ToolMessage（含错误）：告诉 LLM「调用结果是这个错误」
   - HumanMessage（提示）：告诉 LLM「请纠正并重试」
   - 符合 LLM 对话格式要求

8. tool_choice 参数的作用
   - bind_tools(tools=[...], tool_choice="complex_tool") 强制 LLM 调用指定工具
   - 避免 LLM 选择不调用工具或调用错误工具；确保每次都生成 tool_calls

9. 链的完整结构与执行流程
   - 主链：prompt | llm | tool_custom_exception
   - 自我纠正链：主链.with_fallbacks([exception_to_messages | 主链], exception_key="exception")
   - 流程：首次 prompt（last_output 空）→ llm → 工具失败 → 抛 CustomToolException
     → with_fallbacks 捕获并注入 exception → exception_to_messages 生成 last_output
     → 重试：prompt（含错误）→ llm 补齐 dict_arg → 工具成功
   - 课程最终输出：10.5（即 5 * 2.1，参数补全后成功）

10. 三种策略对比 / 适用场景 / 注意事项
   - try-except：捕获错误返回信息（被动）；with_fallbacks：切换备用方案（主动切换）；
     自我纠正：反馈错误让 LLM 修正（主动学习）
   - 适用：参数易错的复杂工具、需高成功率的关键流程、LLM 推理能力强、可容忍额外延迟
   - 注意：增加一次 LLM 调用，成本与延迟翻倍；需 LLM 具备理解错误能力；
     可设多级 fallback 多次重试；错误信息要清晰；限制重试次数避免死循环；
     结合 temperature=0 提高确定性

===================================================================================
"""
from typing import Any

import dotenv
from langchain_core.messages import ToolCall, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 自定义异常类，携带工具调用的完整上下文
# 目的：在 fallback 链中能够获取原始的 tool_call 和异常信息
class CustomToolException(Exception):
    def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
        super().__init__()
        # 保存原始的工具调用请求（包含 id、name、args）
        self.tool_call = tool_call
        # 保存原始异常对象
        self.exception = exception


# 定义复杂工具，需要三个参数
# 这个工具容易因参数缺失而调用失败
@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg


# 工具调用包装函数，抛出自定义异常
# 参数说明：
#   - msg：LLM 生成的 AIMessage（包含 tool_calls）
#   - config：LangChain 运行时配置
def tool_custom_exception(msg: AIMessage, config: RunnableConfig) -> Any:
    try:
        # 提取第一个工具调用的参数并执行工具
        return complex_tool.invoke(msg.tool_calls[0]["args"], config=config)
    except Exception as e:
        # 抛出自定义异常，携带 tool_call 和原始异常
        # 这样 fallback 链就能获取完整的错误上下文
        raise CustomToolException(msg.tool_calls[0], e)


# 将异常信息转换为消息列表
# 这是自我纠正机制的核心函数
# 参数：inputs 包含原始输入 + exception_key 注入的异常对象
def exception_to_messages(inputs: dict) -> dict:
    # 1.从inputs中分离出异常信息
    # pop() 移除 exception 键，避免影响后续的提示词渲染
    exception = inputs.pop("exception")

    # 2.根据异常信息组装占位消息列表
    # 这三条消息模拟了"一次失败的工具调用"的完整对话
    messages = [
        # AIMessage：重现 LLM 之前的错误调用请求
        # content="" 表示没有文本内容，只有工具调用
        AIMessage(content="", tool_calls=[exception.tool_call]),

        # ToolMessage：模拟工具返回的错误信息
        # tool_call_id 必须与 AIMessage 中的 tool_call id 一致
        ToolMessage(tool_call_id=exception.tool_call["id"], content=str(exception.exception)),

        # HumanMessage：明确提示 LLM 纠正错误
        # "请不要重复犯错" 强化了纠正的意图
        HumanMessage(content="最后一次工具调用引发了异常，请尝试使用更正的参数再次调用该工具，请不要重复犯错"),
    ]

    # 将消息列表赋值给 last_output 字段
    # 对应提示词模板中的 ("placeholder", "{last_output}")
    inputs["last_output"] = messages
    return inputs


# 1.创建prompt
# 关键点：("placeholder", "{last_output}") 用于动态插入消息列表
#   - 首次调用：last_output 为空（无错误信息）
#   - 重试调用：last_output 包含错误消息序列
prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}"),
    ("placeholder", "{last_output}")
])

# 2.创建大语言模型并绑定工具
# 参数说明：
#   - temperature=0：提高输出确定性
#   - tools=[complex_tool]：绑定工具
#   - tool_choice="complex_tool"：强制调用指定工具
#     * 避免 LLM 选择不调用工具
#     * 确保每次都会生成 tool_calls
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0).bind_tools(
    tools=[complex_tool], tool_choice="complex_tool",
)

# 3.创建链并执行工具
# 主链：提示词 -> LLM -> 工具调用（可能抛出 CustomToolException）
chain = prompt | llm | tool_custom_exception

# 自我纠正链
# with_fallbacks() 参数说明：
#   - [exception_to_messages | chain]：fallback 链
#     * exception_to_messages 将异常转换为消息
#     * 然后重新执行主链（此时提示词包含错误信息）
#   - exception_key="exception"：将异常对象注入到 fallback 链的输入中
#     * fallback 链的输入变为：{"query": ..., "exception": CustomToolException}
self_correcting_chain = chain.with_fallbacks(
    [exception_to_messages | chain], exception_key="exception"
)

# 4.调用自我纠正链完成任务
# 执行流程：
#   1. 首次尝试：LLM 可能只生成 int_arg 和 float_arg（缺少 dict_arg）
#   2. 工具调用失败，抛出 CustomToolException
#   3. with_fallbacks 捕获异常，注入到输入中
#   4. exception_to_messages 生成包含错误信息的消息列表
#   5. 重新调用 LLM，LLM 看到错误后补充 dict_arg 参数
#   6. 工具调用成功，返回结果
print(self_correcting_chain.invoke({"query": "使用复杂工具，对应参数为5和2.1"}))
