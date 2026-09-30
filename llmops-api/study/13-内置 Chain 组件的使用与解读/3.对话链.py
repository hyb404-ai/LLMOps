#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 9:40
@Author  : thezehui@gmail.com
@File    : 3.对话链.py

===================================================================================
知识点讲解：对话链与消息历史管理（ConversationChain 的现代替代方案）
===================================================================================

1. 对话链要解决的核心问题
   - LLM 本身无状态，每次调用之间不会记住之前内容
   - 多轮对话需将历史消息一并送入模型，才能实现"记忆"效果
   - 对话链职责：自动完成历史消息的读取、注入与写回

2. 从 ConversationChain 到 RunnableWithMessageHistory
   - 0.x 的 ConversationChain 内置 ConversationBufferMemory，历史读写对用户不可见、隐式存储
   - 隐式存储问题：难以定制、难以支持多用户会话隔离、难以持久化
   - 1.x 的 RunnableWithMessageHistory 采用显式设计，由开发者提供历史存储函数
   - 课程示例：ConversationChain(llm=ChatOpenAI(model="gpt-4o"))，invoke 返回 dict 含 input / history / response 三个键，history 由内部 Memory 自动维护（旧写法，已不推荐）

3. RunnableWithMessageHistory 的三个关键参数
   - runnable：被包装的底层链，必须包含 MessagesPlaceholder 用于接收历史
   - get_session_history：回调函数，根据 session_id 返回 BaseChatMessageHistory
   - input_messages_key / history_messages_key：指定输入与历史在 dict 中的键名

4. MessagesPlaceholder 的作用
   - 在提示词模板中预留一个"消息列表插槽"
   - invoke 时由 RunnableWithMessageHistory 自动填入历史消息列表
   - 位置关键：通常放在 system 之后、human 之前，保证对话顺序正确

5. session_id 与会话隔离
   - 通过 config={"configurable": {"session_id": "..."}} 指定当前会话
   - 不同 session_id 对应不同历史记录，实现多用户 / 多会话隔离
   - 生产环境中 session_id 通常来自用户 ID 或会话 UUID

6. BaseChatMessageHistory 的常见实现
   - InMemoryChatMessageHistory：内存存储，进程重启后丢失，适合演示与测试
   - FileChatMessageHistory：文件存储，可跨进程持久化
   - RedisChatMessageHistory / SQLChatMessageHistory：生产级持久化方案

7. 完整的数据流转过程
   - 调用前：从 store 读取 session_id 对应的历史 → 填入 history 插槽
   - 执行中：{"input": ..., "history": [...]} → prompt → llm → parser → str
   - 调用后：将本轮 HumanMessage 与 AIMessage 自动追加写回历史

===================================================================================
"""
import dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量（API Key 等）
dotenv.load_dotenv()

# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，实现 Runnable 协议
llm = ChatOpenAI(model="deepseek-v4-pro")

# ==================== LangChain 1.x 写法：LCEL 链 + 显式历史管理 ====================

# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：构建包含历史消息插槽的对话提示词模板
#   参数：messages - 消息定义列表，支持元组和 MessagesPlaceholder 混用
#   返回：ChatPromptTemplate 实例
#   消息顺序的意义：system 定义角色 → history 提供上下文 → human 为当前提问
prompt = ChatPromptTemplate.from_messages([
    # system 消息：固定的角色设定，不参与历史记录
    ("system", "你是DeepSeek开发的聊天机器人，请根据对话上下文回复用户问题"),

    # MessagesPlaceholder(variable_name: str) -> MessagesPlaceholder
    #   作用：在模板中预留消息列表插槽，用于插入动态数量的历史消息
    #   参数：variable_name - 变量名，必须与 history_messages_key 一致
    #   填充来源：RunnableWithMessageHistory 自动从 get_session_history 读取
    MessagesPlaceholder("history"),

    # human 消息：当前轮次的用户输入，变量名需与 input_messages_key 一致
    ("human", "{input}"),
])

# LCEL 管道构建基础链
# 数据流转：dict → prompt → Messages → llm → AIMessage → parser → str
chain = prompt | llm | StrOutputParser()

# store: dict[str, BaseChatMessageHistory]
#   作用：全局会话历史仓库，键为 session_id，值为对应的历史对象
#   说明：这是显式存储的体现，开发者完全掌控历史的存放位置与生命周期
#   生产替代：可替换为 Redis、数据库等持久化方案
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据会话 id 获取消息历史，替代 Memory 组件的隐式存储

    这是 RunnableWithMessageHistory 要求的工厂函数（factory function）。

    参数：
        session_id: str - 会话唯一标识，由 config.configurable.session_id 传入

    返回：
        BaseChatMessageHistory - 该会话对应的历史存储对象
        该对象需实现 messages 属性（读）和 add_message 方法（写）

    调用时机：
        每次 invoke/stream 时被 RunnableWithMessageHistory 自动调用两次
        - 执行前：读取历史填入 MessagesPlaceholder
        - 执行后：写回本轮的 Human 与 AI 消息

    实现要点：
        必须保证同一 session_id 返回同一个历史对象实例，否则历史会丢失
    """
    if session_id not in store:
        # InMemoryChatMessageHistory() -> InMemoryChatMessageHistory
        #   作用：创建基于内存列表的消息历史容器
        #   特点：读写快但不持久化，进程退出后数据丢失
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# RunnableWithMessageHistory(
#     runnable: Runnable,
#     get_session_history: Callable[..., BaseChatMessageHistory],
#     input_messages_key: str = None,
#     history_messages_key: str = None,
#     output_messages_key: str = None,
# ) -> RunnableWithMessageHistory
#   作用：为任意 Runnable 链包装自动的历史管理能力
#   参数：
#     runnable - 被包装的底层链（此处为 prompt | llm | parser）
#     get_session_history - 根据 session_id 返回历史对象的工厂函数
#     input_messages_key - 输入 dict 中表示"当前用户消息"的键名，需与 prompt 中的变量对应
#     history_messages_key - 输入 dict 中表示"历史消息列表"的键名，需与 MessagesPlaceholder 一致
#     output_messages_key - 输出 dict 中表示"AI 回复"的键名（输出为 str 时可省略）
#   返回：包装后的 Runnable，仍支持 invoke/batch/stream
#   核心行为：
#     1. 从 config 中提取 session_id
#     2. 调用 get_session_history(session_id) 获取历史对象
#     3. 读取 history.messages 填入 history_messages_key
#     4. 执行底层 runnable
#     5. 将本轮 HumanMessage 与 AIMessage 追加写回历史
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# config: dict
#   作用：运行时配置，configurable.session_id 用于定位会话历史
#   说明：不同 session_id 之间历史完全隔离，实现多用户支持
config = {"configurable": {"session_id": "muxiaoke"}}

# 第一轮对话
# chain_with_history.invoke(input: dict, config: dict) -> str
#   作用：执行带历史管理的对话链
#   参数：
#     input - 包含 input_messages_key 的字典（此处为 {"input": "..."}）
#     config - 运行时配置，必须包含 configurable.session_id
#   返回：LLM 生成的字符串回复
#   执行流程：
#     1. 读取 store["muxiaoke"].messages → [] （首轮为空）
#     2. {"input": "...", "history": []} → prompt → llm → parser → str
#     3. 写回：store["muxiaoke"] 追加 HumanMessage("你好，我是慕小课...") 与 AIMessage(回复)
content = chain_with_history.invoke(
    {"input": "你好，我是慕小课，我喜欢打篮球还有游泳，你喜欢什么运动呢？"},
    config=config,
)
print(content)

# 第二轮对话（验证历史记忆效果）
# 关键差异：此时 store["muxiaoke"].messages 已包含第一轮的两条消息
#   1. 读取历史 → [HumanMessage("你好，我是慕小课..."), AIMessage("...")]
#   2. 历史被填入 MessagesPlaceholder("history")
#   3. 模型能看到用户之前提到的"打篮球"和"游泳"，因此可以正确统计
#   4. 本轮消息继续追加写回历史
content = chain_with_history.invoke(
    {"input": "根据上下文信息，请统计一下我的运动爱好有什么?"},
    config=config,
)
print(content)

# ==================== 最佳实践与扩展写法 ====================
# 1. 使用持久化历史存储（跨进程保留对话）
# from langchain_community.chat_message_histories import FileChatMessageHistory
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     return FileChatMessageHistory(f"./chat_history_{session_id}.txt")
#
# 2. 流式输出对话内容（提升交互体验）
# for chunk in chain_with_history.stream({"input": "你好"}, config=config):
#     print(chunk, end="", flush=True)
#
# 3. 多会话隔离演示
# config_a = {"configurable": {"session_id": "user_a"}}
# config_b = {"configurable": {"session_id": "user_b"}}
# # 两个 session 的历史互不影响
#
# 4. 限制历史长度（避免上下文窗口溢出）
# from langchain_core.runnables import RunnablePassthrough
# def trim_history(data: dict) -> dict:
#     # 只保留最近 10 条消息
#     data["history"] = data["history"][-10:]
#     return data
# chain = RunnablePassthrough.assign(...) | trim_history | prompt | llm | StrOutputParser()

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# ConversationChain 内部自带 ConversationBufferMemory，历史读写对使用者不可见
# 缺点：
#   - 无法支持多用户会话隔离（Memory 绑定在 Chain 实例上）
#   - 历史存储位置不透明，难以持久化与调试
#   - 与 LCEL 生态不兼容，无法灵活组合
# from langchain_classic.chains.conversation.base import ConversationChain
#
# chain = ConversationChain(llm=llm)
#
# content = chain.invoke({"input": "你好，我是慕小课，我喜欢打篮球还有游泳，你喜欢什么运动呢？"})
# print(content)
#
# content = chain.invoke({"input": "根据上下文信息，请统计一下我的运动爱好有什么?"})
# print(content)
