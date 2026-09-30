#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/14 21:58
@Author  : thezehui@gmail.com
@File    : 1.XMLAgent示例.py

===================================================================================
知识点讲解：内置的其他 Agent 类型介绍与中间件（Middleware）
===================================================================================

1. LangChain 0.x 时代的多种 Agent 类型（均基于「推理-行动-观察」思想）
   - 各类 Agent 的设计思想都是 ReAct（推理-行动-观察）：先推理要做什么，调用工具行动，再观察结果
   - create_react_agent：ReACT 文本协议（Thought/Action/Observation）
   - create_xml_agent：XML 标签协议（本章重点）
   - create_json_chat_agent：JSON 格式协议
   - create_tool_calling_agent / create_openai_tools_agent：原生函数调用
   - create_structured_chat_agent：结构化对话
   - create_self_ask_with_search_agent：自问自答 + 搜索
   - 之所以有多种类型，是因为不同模型擅长不同格式（如 Claude 擅长 XML、部分模型擅长 JSON），需配套不同 prompt

2. XMLAgent 的设计思路
   - 使用 XML 标签作为工具调用协议，相比纯文本关键词更结构化、解析更可靠
   - <tool>工具名</tool>：指定要调用的工具
   - <tool_input>参数</tool_input>：传递工具参数
   - <observation>结果</observation>：工具返回结果
   - <final_answer>答案</final_answer>：最终答案
   - 适用于不支持原生函数调用的模型（如早期 Claude），如今这类模型已可被原生函数调用取代

3. XML 协议实例与 ReACT 对照
   - 模型按 prompt 约定输出，例如：
       <tool>search</tool><tool_input>weather in SF</tool_input>
       <observation>64 degrees</observation>
       <final_answer>The weather in SF is 64 degrees</final_answer>
   - XML：结构化标签，解析更可靠；ReACT：纯文本关键词（Thought/Action/Observation），解析容易出错
   - 两者都是「模拟」函数调用的变通方案，现代模型原生支持函数调用后二者都已过时

4. LangChain 1.x 的架构统一
   - 所有 Agent 类型收敛到 create_agent
   - 不再提供 create_xml_agent 等专用构造函数
   - 差异化能力通过 middleware（中间件）表达
   - 底层统一使用 LangGraph + 原生工具调用

5. Middleware（中间件）机制简介
   - LangChain 1.x 的核心扩展机制，在 Agent 执行流程中插入自定义逻辑
   - 可拦截、修改、限制 Agent 行为，替代旧版 AgentExecutor 的各种参数配置

6. 常用内置 Middleware
   - ModelCallLimitMiddleware：限制模型调用次数（替代 AgentExecutor 的 max_iterations）
   - SummarizationMiddleware：自动摘要长对话
   - HumanInTheLoopMiddleware：人工审批工具调用
   - PIIMiddleware：敏感信息脱敏
   - ToolCallLimitMiddleware：限制工具调用次数
   - TodoListMiddleware：任务清单管理

7. ModelCallLimitMiddleware 参数详解
   - thread_limit：单个线程（会话）的模型调用次数上限
   - run_limit：单次运行的模型调用次数上限
   - exit_behavior：达到上限时的行为，"end" 正常结束返回当前结果 / "error" 抛出异常
   - 作用：防止 Agent 陷入无限循环、控制成本

8. 为什么需要调用次数限制
   - Agent 可能因推理错误陷入循环，每次循环都调用 LLM，成本累积
   - 防止单个请求消耗过多资源、保护 API 配额

9. Middleware 的执行时机
   - before_model / after_model：模型调用前后
   - before_tool / after_tool：工具调用前后
   - 不同 middleware 实现不同的钩子

10. 自定义 Middleware 与最佳实践
   - 继承 AgentMiddleware 基类，实现需要的钩子方法，可修改状态、拦截执行、注入逻辑，支持组合多个 middleware
   - 使用 create_agent 统一构建 Agent，通过 middleware 表达差异化需求
   - 生产环境必须设置调用次数限制，组合多个 middleware 实现复杂控制
   - 优先选择支持原生函数调用的模型

11. 各类 Agent 的异同点（横向对比）
   - 相同点：都拥有 input 与 agent_scratchpad 两个输入变量（原始问题与智能体草稿）；都是单 Agent 自我执行、无法多 Agent 协作；都可通过切换 prompt 与 create_xxx_agent() 快速切换；都基于「推理-行动-观察」，只是 prompt 不同；都用同一个 LLM 做推理与答案生成，不支持多 LLM 分工
   - 差异点：提示词风格不同（有的把 tools 写进 prompt、有的用文本/消息提示）；输出解析器不一致（多数取决于 prompt，有的支持多工具有的不支持）；输入编码方式不一致（是否支持历史记忆输入）

12. 典型输出示例与观察点
   - 对「马拉松的世界记录是多少？」运行 XMLAgent，得到：
       <tool>google_serper</tool><tool_input>马拉松 世界记录 ...</tool_input>
       <final_answer>截至2023年，马拉松的世界纪录是2小时01分09秒，由埃利乌德·基普乔格于2018年创造。</final_answer>
       输出：{'output': '截至2023年，马拉松的世界纪录是2小时01分09秒，由埃利乌德·基普乔格于2018年创造。'}
   - 观察点：模型先产出一个 <tool> 调用再由 executor 执行，最终 <final_answer> 成为链的输出；这正是 ReAct「推理-行动-观察」在 XML 协议下的具体落地

===================================================================================
"""
import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 加载环境变量（需要 SERPER_API_KEY 和 OPENAI_API_KEY）
dotenv.load_dotenv()


# Google 搜索工具的参数模式
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


# DALL-E 图像生成工具的参数模式
class DallEArgsSchema(BaseModel):
    query: str = Field(description="输入应该是生成图像的文本提示(prompt)")


# 1.定义工具与工具列表
# 工具1：Google 搜索，用于获取实时信息
google_serper = GoogleSerperRun(
    name="google_serper",
    description=(
        "一个低成本的谷歌搜索API。"
        "当你需要回答有关时事的问题时，可以调用该工具。"
        "该工具的输入是搜索查询语句。"
    ),
    args_schema=GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper(),
)

# 工具2：DALL-E 图像生成
# 这里显式指定了 args_schema，自定义参数描述
dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"),
    args_schema=DallEArgsSchema,
)

# 工具列表
tools = [google_serper, dalle]

# 2.创建大语言模型
llm = ChatOpenAI(model="deepseek-v4-pro")

# 3.LangChain 1.x 写法：不同 Agent 类型统一收敛到 create_agent，
# 差异化能力由 middleware 表达(此处用调用次数上限替代 AgentExecutor 的 max_iterations)
# 参数说明：
#   - model：大语言模型实例
#   - tools：工具列表
#   - system_prompt：系统提示词
#   - middleware：中间件列表，用于扩展和控制 Agent 行为
#
# ModelCallLimitMiddleware 参数说明：
#   - thread_limit=5：单个会话线程最多调用模型 5 次
#     * 防止 Agent 陷入无限循环
#     * 替代旧版 AgentExecutor 的 max_iterations 参数
#   - exit_behavior="end"：达到上限时正常结束，返回当前结果
#     * "end"：优雅退出，返回已有结果
#     * "error"：抛出异常，中断执行
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Help the user answer any questions.",
    middleware=[ModelCallLimitMiddleware(thread_limit=5, exit_behavior="end")],
)

# 执行 Agent
# 输入：需要实时信息的问题（会触发 google_serper 工具调用）
# 执行流程：
#   1. Agent 理解问题需要实时信息
#   2. 调用 google_serper 搜索马拉松世界记录
#   3. 整合搜索结果生成回答
#   4. 如果推理步骤超过 5 次模型调用，middleware 会终止执行
result = agent.invoke({"messages": [{"role": "user", "content": "马拉松的世界记录是多少？"}]})
print(result["messages"][-1].content)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# XMLAgent 用 <tool>/<tool_input>/<observation> 文本协议兼容不支持原生函数调用的模型，
# 1.x 已不再提供 create_xml_agent
#
# XMLAgent 的工作原理：
#   1. 提示词中用 XML 标签定义工具调用协议
#   2. LLM 输出 <tool>工具名</tool><tool_input>参数</tool_input>
#   3. AgentExecutor 解析 XML 标签，提取工具名和参数
#   4. 执行工具，将结果包装为 <observation>结果</observation>
#   5. 追加到 agent_scratchpad，重新调用 LLM
#   6. LLM 输出 <final_answer>答案</final_answer> 时结束
#
# 已废弃的原因：
#   - 现代模型原生支持函数调用，无需文本协议
#   - XML 解析仍可能失败（标签不完整、嵌套错误等）
#   - create_agent 统一了所有 Agent 类型
#
# from langchain_classic.agents import create_xml_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate
#
# prompt = ChatPromptTemplate.from_messages([
#     ("human", """You are a helpful assistant. Help the user answer any questions.
#
# You have access to the following tools:
#
# {tools}
#
# In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
# For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:
#
# <tool>search</tool><tool_input>weather in SF</tool_input>
# <observation>64 degrees</observation>
#
# When you are done, respond with a final answer between <final_answer></final_answer>. For example:
#
# <final_answer>The weather in SF is 64 degrees</final_answer>
#
# Begin!
#
# Previous Conversation:
# {chat_history}
#
# Question: {input}
# {agent_scratchpad}"""),
# ])
#
# agent = create_xml_agent(prompt=prompt, llm=llm, tools=tools)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# print(agent_executor.invoke({"input": "马拉松的世界记录是多少？", "chat_history": ""}))
