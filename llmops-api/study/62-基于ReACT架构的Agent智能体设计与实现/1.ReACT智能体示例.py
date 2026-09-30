#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/13 22:42
@Author  : thezehui@gmail.com
@File    : 1.ReACT智能体示例.py

===================================================================================
知识点讲解：基于 ReACT 架构的 Agent 智能体设计与实现
===================================================================================

1. 什么是 Agent（为何需要）
   - 当工具的使用次数与顺序取决于输入时，我们希望让 LLM 自己决定调用工具的次数和顺序，Agent 能做到这一点
   - Agent 是能利用 LLM 和其他工具执行复杂任务的系统，用来处理 LLM 无法直接解决、需多步骤或外部数据源的任务
   - 最基础工作流程只有 5 步：输入理解 → 计划定制 → 工具调用 → 结果整合 → 反馈循环（迭代到满足为止）
   - 三个组成模块：Tools（工具集）、Executor（执行计划的逻辑）、Prompt Templates（指导理解与处理的模板）

2. ReACT 架构简介
   - ReACT = Reasoning（推理）+ Acting（行动），由 Yao 等人在 2022 年论文（arXiv 2210.03629）首次提出
   - 目前绝大部分 Agent 架构都是在 ReACT 上衍生的
   - 核心思想：让 LLM 交替进行「思考」与「行动」，通过多轮循环逐步解决复杂问题
   - 对比函数调用：Agent 流程与之非常接近，只是多了「执行工具」和「观察结果」两步

3. ReACT 的经典循环流程（文本协议）
   Question（问题）
     -> Thought（思考：我需要做什么）
     -> Action（行动：调用哪个工具）
     -> Action Input（行动输入：工具参数）
     -> Observation（观察：工具返回结果）
     -> Thought（继续思考...）
     -> ... 循环 N 次 ...
     -> Final Answer（最终答案）
   - 旧版 ReACT 通过解析输出中的特定关键词（Thought / Action / Observation / Final Answer）来提取后续步骤
   - 经典提示词模板见 LangChain Hub 的 hwchase17/react

4. ReACT 智能体的缺陷（为何要从 0.x 迁移）
   - 核心缺陷：底层靠解析响应中的特定关键词来推进步骤，而 LLM 输出极不稳定
   - 例：问「你好，你是？」本应直接回答，但若模型没输出 "Final Answer" 关键字、只给了正文，
     组装后只剩 "Thought: ..." 而没有后续 Action/Final Answer，解析器识别不了步骤，必然抛错
     （课程实测即便 GPT-4o 也会犯）
   - 若想把 Prompt 改成中文，通常还需同步修改输出解析器，使用代价非常大
   - 因此 1.x 改用原生 tool_calls 承载推理过程，从根本上规避了文本解析的脆弱性

5. LangChain 1.x 的重大变化
   - 旧版（0.x）：create_react_agent + AgentExecutor
     * 手写包含 Thought/Action/Observation 的文本提示模板
     * 需要 agent_scratchpad 占位符存储中间步骤
     * AgentExecutor 解析 LLM 文本输出提取 Action，依赖文本解析，容易出错
   - 新版（1.x）：create_agent（基于 LangGraph）
     * 使用原生工具调用（tool_calls），无需文本解析
     * 不需要 agent_scratchpad，状态由 LangGraph 管理
     * 更稳定、更易调试

6. Agent 与 Chain 的区别
   - Chain：固定执行流程，步骤预先定义
   - Agent：动态决策，LLM 自主决定调用哪些工具、调用几次
   - Agent 具备自主规划和多步推理能力

7. create_agent() 参数详解
   - model：大语言模型实例（ChatModel）
   - tools：工具列表，Agent 可调用的工具
   - system_prompt：系统提示词（替代旧版提示模板，无需 {tools}/{tool_names}/{agent_scratchpad}）
   - checkpointer：状态持久化器（用于多轮对话记忆）
   - middleware：中间件列表（扩展 Agent 行为，如限制迭代次数）
   - response_format：结构化输出格式（可选）

8. 输入输出格式（messages 结构）
   - 输入：{"messages": [{"role": "user", "content": "问题"}]}
   - 输出：{"messages": [...完整消息历史...]}
   - 最终答案：result["messages"][-1].content
   - 消息历史包含：HumanMessage、AIMessage（可能含 tool_calls）、ToolMessage

9. 流式输出与过程观察
   - agent.stream(input, stream_mode="updates")
   - stream_mode 选项：
     * "updates"：每个节点的状态更新（推荐调试用）
     * "values"：每步的完整状态
     * "messages"：逐 token 流式输出
   - 可以实时观察 Agent 的推理和工具调用过程

10. LangGraph 的优势
   - 图结构表达 Agent 循环，逻辑清晰
   - 内置状态管理和持久化
   - 支持人工介入（human-in-the-loop）、中断与恢复
   - 更好的可观测性

11. 旧版 ReACT 文本协议（附录对照）
   - {tools}：工具描述文本；{tool_names}：工具名称列表
   - {input}：用户问题；{agent_scratchpad}：中间步骤记录
   - render_text_description_and_args()：渲染工具描述

12. 适用场景
   - 需要多步推理的复杂问题
   - 需要调用多个工具协作的任务
   - 开放式问答（不确定需要哪些工具）
   - 实时信息检索和整合

13. 注意事项与最佳实践
   - Agent 可能陷入循环，需设置迭代上限（用 middleware，见第 64 章）
   - 每轮循环都会调用 LLM，成本较高；推理过程不可完全预测
   - 优先使用 LangChain 1.x 的 create_agent
   - 工具描述要清晰，帮助 Agent 正确选择；system_prompt 明确角色与能力边界
   - 使用 checkpointer 实现多轮对话记忆（见第 63 章）；使用 stream 观察执行过程便于调试

===================================================================================
"""
import dotenv
from langchain.agents import create_agent
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 加载环境变量（需要 SERPER_API_KEY）
dotenv.load_dotenv()


# Google 搜索工具的参数模式
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


# 1.定义工具与工具列表
# GoogleSerperRun 是 LangChain 内置的 Google 搜索工具
# 工具描述很重要，它帮助 Agent 判断何时调用该工具
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
# 工具列表，Agent 可以从中选择调用
tools = [google_serper]

# 2.创建大语言模型
# temperature=0 提高推理的确定性和稳定性
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# 3.LangChain 1.x 写法：create_agent 基于 LangGraph 内置 ReACT 循环，
# 推理过程由原生工具调用承载，不再需要 agent_scratchpad 与文本解析
# 参数说明：
#   - model：大语言模型实例
#   - tools：Agent 可调用的工具列表
#   - system_prompt：系统提示词，定义 Agent 的角色和行为
#     * 替代了旧版需要手写的 ReACT 文本提示模板
#     * 不需要 {tools}、{tool_names}、{agent_scratchpad} 占位符
# 返回值：可执行的 LangGraph 图对象（CompiledGraph）
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个乐于助人的智能助手，可以调用工具回答有关时事的问题。",
)

# 4.执行智能体并检索，输入输出统一为 messages 结构
# 输入格式：{"messages": [{"role": "user", "content": "问题"}]}
#   - role 可选值："user"、"assistant"、"system"
#   - 也可以直接传入 LangChain 的 Message 对象
result = agent.invoke({"messages": [{"role": "user", "content": "你好，你是？"}]})
# 输出格式：{"messages": [完整的消息历史列表]}
#   - 包含：HumanMessage、AIMessage（可能含 tool_calls）、ToolMessage
#   - messages[-1] 是 Agent 的最终回答
print(result["messages"][-1].content)

# 也可以流式观察每一步的推理与工具调用
# stream_mode="updates" 会输出每个节点的状态更新
# 便于观察 Agent 的推理过程：何时调用工具、工具返回什么、如何整合结果
# for chunk in agent.stream(
#         {"messages": [{"role": "user", "content": "马拉松最新的世界纪录是多少?"}]},
#         stream_mode="updates",
# ):
#     print(chunk)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# 手写 ReACT 文本提示模板，由 AgentExecutor 解析 Thought/Action/Observation 文本并循环调用工具
#
# 旧版的问题：
#   1. 需要手写复杂的 ReACT 提示模板
#   2. 依赖文本解析提取 Action 和 Action Input，容易解析失败
#   3. 需要 agent_scratchpad 占位符手动管理中间步骤
#   4. 状态管理由 AgentExecutor 黑盒处理，难以调试和扩展
#
# from langchain_classic.agents import create_react_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.tools import render_text_description_and_args
#
# prompt = ChatPromptTemplate.from_template(
#     "Answer the following questions as best you can. You have access to the following tools:\n\n"
#     "{tools}\n\n"                         # 工具描述，由 tools_renderer 生成
#     "Use the following format:\n\n"
#     "Question: the input question you must answer\n"
#     "Thought: you should always think about what to do\n"          # 思考步骤
#     "Action: the action to take, should be one of [{tool_names}]\n" # 选择工具
#     "Action Input: the input to the action\n"                       # 工具参数
#     "Observation: the result of the action\n"                       # 工具结果
#     "... (this Thought/Action/Action Input/Observation can repeat N times)\n"  # 循环
#     "Thought: I now know the final answer\n"
#     "Final Answer: the final answer to the original input question\n\n"
#     "Begin!\n\n"
#     "Question: {input}\n"
#     "Thought:{agent_scratchpad}"          # 中间步骤记录，由 AgentExecutor 填充
# )
#
# agent = create_react_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools,
#     tools_renderer=render_text_description_and_args,  # 工具描述渲染函数
# )
#
# # AgentExecutor 负责驱动 ReACT 循环：解析输出 -> 调用工具 -> 追加 Observation -> 重新调用 LLM
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# print(agent_executor.invoke({"input": "你好，你是？"}))
