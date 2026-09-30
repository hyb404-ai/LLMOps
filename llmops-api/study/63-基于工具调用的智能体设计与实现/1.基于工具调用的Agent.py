#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/14 21:31
@Author  : thezehui@gmail.com
@File    : 1.基于工具调用的Agent.py

===================================================================================
知识点讲解：基于工具调用（Tool Calling）的智能体设计与实现
===================================================================================

1. ReACT Agent 的脆弱性（设计动机）
   - ReACT 把 tools 描述、agent_scratchpad、工具结果、推理全部塞进同一 prompt，靠文本解析提取 LLM 的规范输出决定下一步
   - 这种模式随 LLM 输出随机性、不同模型性能差异而异常脆弱
   - ReACT 早期针对「文本补全模型」设计（给一段话让 LLM 补全后续文本）

2. Tool Calling Agent 的演进与核心优势
   - 思想不变，但迁移到「聊天消息 + 工具调用」架构，使用更稳定的消息列表 + 结构化工具调用
   - 输出规范通过检测 LLM 输出「文本内容」还是「工具调用参数」判断下一步，性能更稳定
   - 支持一次性生成多个工具的参数（并行工具调用），能力更强
   - 要求模型支持函数/工具调用能力

3. LangChain 1.x 的统一收敛
   - 旧版：create_react_agent、create_tool_calling_agent、create_openai_functions_agent、create_xml_agent 等多种 Agent
   - 新版：统一为 create_agent，底层基于 LangGraph 实现，默认原生工具调用，差异化能力用 middleware 表达
   - （旧版也可走 create_tool_calling_agent + AgentExecutor(verbose=True) 的写法，步骤与本例一致：工具列表→Prompt→LLM→Agent→Executor）

4. create_agent() 完整参数说明
   - model：大语言模型实例
   - tools：工具列表
   - system_prompt：系统提示词（替代提示模板）
   - checkpointer：状态持久化器（实现多轮对话记忆）
   - middleware：中间件列表（扩展 Agent 行为）
   - response_format：结构化输出格式
   - state_schema：自定义状态结构

5. checkpointer 与多轮对话记忆
   - 作用：持久化 Agent 的对话状态，替代旧版的 chat_history 占位符和 Memory 组件
   - InMemorySaver：内存存储（开发测试用）
   - SqliteSaver：SQLite 存储（单机持久化）
   - PostgresSaver：PostgreSQL 存储（生产环境）

6. thread_id 与会话隔离
   - config = {"configurable": {"thread_id": "用户标识"}}
   - 每个 thread_id 对应一个独立的对话线程
   - 同一 thread_id 的多次调用会共享消息历史
   - 不同 thread_id 之间完全隔离，实现多用户、多会话场景

7. 多工具协作
   - 本示例包含两个工具：google_serper（Google 搜索，获取实时信息）、openai_dalle（DALL-E 图像生成）
   - Agent 根据用户意图自主选择工具
   - 支持一次调用多个工具（并行工具调用）

8. Agent 的自主决策能力
   - 用户输入「帮我绘制一幅鲨鱼在天上游泳的场景」→ 判断为图像生成 → 调用 openai_dalle
   - 「马拉松世界记录是多少」→ 调用 google_serper
   - 「你好」→ 不调用工具，直接回答

9. args_schema 的可选性
   - 内置工具通常自带默认的 args_schema
   - 可以自定义 args_schema 覆盖默认参数描述
   - 本示例中 DallEArgsSchema 被注释，使用工具默认参数模式
   - 自定义时要确保参数名与工具实际参数一致

10. 消息历史的自动管理
    - checkpointer 自动保存每轮对话的消息
    - 下次调用时自动加载历史消息
    - Agent 可以理解上下文（如「再画一张」）
    - 无需手动管理 chat_history

11. 输入输出格式
    - 输入：{"messages": [{"role": "user", "content": "..."}]}
    - 配置：config={"configurable": {"thread_id": "..."}}
    - 输出：{"messages": [...]}，最终答案在 messages[-1].content

12. 适用场景与最佳实践
    - 适用：多工具协作的智能助手、需要多轮对话记忆的场景、需要区分不同用户会话的应用、生产级 Agent 应用
    - 最佳实践：生产环境使用持久化 checkpointer（SqliteSaver/PostgresSaver）；thread_id 使用用户 ID 或会话 ID；工具描述要清晰避免 Agent 选错工具；使用 middleware 限制资源消耗；监控工具调用次数和成本；考虑添加工具调用的权限控制

===================================================================================
"""
import dotenv
from langchain.agents import create_agent
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

# 加载环境变量（需要 SERPER_API_KEY 和 OPENAI_API_KEY）
dotenv.load_dotenv()


# Google 搜索工具的参数模式
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


# DALL-E 图像生成工具的参数模式
# 注意：本示例中未使用（被注释），使用工具的默认参数模式
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

# 工具2：DALL-E 图像生成，用于文生图
dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"),
    # args_schema=DallEArgsSchema,  # 可选：自定义参数模式，需与工具实际参数一致
)

# 工具列表，Agent 会根据用户意图自主选择调用
tools = [google_serper, dalle]

# 2.创建大语言模型
# 模型必须支持函数调用（tool calling）能力
llm = ChatOpenAI(model="deepseek-v4-pro")

# 3.LangChain 1.x 写法：create_agent 直接返回可执行的 LangGraph 图，
# system_prompt 替代提示模板，checkpointer 替代 chat_history 占位符
# 参数说明：
#   - model：大语言模型实例
#   - tools：工具列表，Agent 可自主选择调用
#   - system_prompt：系统提示词，定义 Agent 角色
#     * 替代旧版需要手写的含 agent_scratchpad 的提示模板
#   - checkpointer：状态持久化器
#     * InMemorySaver()：内存存储，进程重启后丢失（开发测试用）
#     * 生产环境推荐：SqliteSaver、PostgresSaver
#     * 作用：自动保存和加载对话历史，实现多轮对话记忆
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是由DeepSeek开发的聊天机器人，善于帮助用户解决问题。",
    checkpointer=InMemorySaver(),
)

# 配置对象，用于指定会话线程
# thread_id 的作用：
#   - 标识一个独立的对话线程（会话）
#   - 同一 thread_id 的多次调用共享消息历史
#   - 不同 thread_id 之间完全隔离
#   - 实践中通常使用用户 ID 或会话 ID
config = {"configurable": {"thread_id": "muxiaoke"}}

# 执行 Agent
# 参数说明：
#   - 第一个参数：输入消息（messages 结构）
#   - config：运行时配置，包含 thread_id
# 执行流程：
#   1. checkpointer 加载 thread_id 对应的历史消息
#   2. Agent 理解用户意图（图像生成需求）
#   3. Agent 选择调用 openai_dalle 工具
#   4. 工具生成图像并返回 URL
#   5. Agent 整合工具结果生成最终回答
#   6. checkpointer 保存本轮对话到 thread_id
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我绘制一幅鲨鱼在天上游泳的场景"}]},
    config=config,
)
# messages[-1] 是 Agent 的最终回答
print(result["messages"][-1].content)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# 需要手写含 agent_scratchpad 占位符的提示模板，再由 AgentExecutor 驱动工具调用循环
#
# 旧版的问题：
#   1. 需要手写包含 4 个部分的提示模板
#   2. {chat_history} 需要配合 Memory 组件手动管理
#   3. {agent_scratchpad} 由 AgentExecutor 填充中间步骤
#   4. AgentExecutor 黑盒驱动循环，难以扩展和调试
#
# from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是由DeepSeek开发的聊天机器人，善于帮助用户解决问题。"),  # 系统提示
#     ("placeholder", "{chat_history}"),      # 历史对话，需 Memory 组件管理
#     ("human", "{input}"),                   # 用户输入
#     ("placeholder", "{agent_scratchpad}"),  # 中间步骤，由 AgentExecutor 填充
# ])
#
# agent = create_tool_calling_agent(prompt=prompt, llm=llm, tools=tools)
# # AgentExecutor 驱动工具调用循环，verbose=True 打印执行过程
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# print(agent_executor.invoke({"input": "帮我绘制一幅鲨鱼在天上游泳的场景"}))
