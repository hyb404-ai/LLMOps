#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/27 11:30
@Author  : thezehui@gmail.com
@File    : 1.RunnableWithMessageHistory使用示例.py

===================================================================================
知识点讲解：RunnableWithMessageHistory 简化多轮对话代码
===================================================================================

1. RunnableWithMessageHistory 的设计定位
   - 是 LangChain 1.x 推荐的记忆（Memory）管理方案，替代 0.x 的 Memory 组件
   - 本质是一个装饰器（Wrapper）：包装任意 Runnable，为其注入历史读写逻辑
   - 遵循"显式优于隐式"原则，历史存储由开发者通过回调函数掌控

2. 它自动完成的三件事
   - 读取历史：调用 get_session_history(session_id) 拿到历史消息列表
   - 注入历史：把消息列表填入 history_messages_key 指定的变量
   - 写回历史：执行完成后自动追加本轮的 HumanMessage 与 AIMessage

3. 与直接手写历史管理的对比
   - 手写方式需要在每次调用前后手动 append 消息，容易遗漏且代码重复
   - RunnableWithMessageHistory 将这部分模板代码收敛到框架内部
   - 同时保留了存储实现的自由度（内存/文件/Redis/数据库）

4. FileChatMessageHistory 的特点
   - 将消息以 JSON 格式序列化写入本地文本文件
   - 优点：跨进程持久化，程序重启后对话历史依然可用
   - 缺点：无并发控制，不适合高并发生产环境（生产推荐 Redis/SQL）

5. 流式输出与历史写回的配合
   - 使用 stream() 时，框架会先逐块 yield 内容给调用方
   - 流结束后（生成器耗尽时）才把完整的 AI 回复写回历史
   - 因此必须完整消费生成器，否则历史可能不完整

6. 关键参数的对应关系（三处必须一致）
   - prompt 中的 MessagesPlaceholder("history")
   - RunnableWithMessageHistory 的 history_messages_key="history"
   - prompt 中的 {query} 与 input_messages_key="query"

7. 数据流转过程（单轮）
   - {"query": 用户输入} + config{session_id}
   - → 读取 FileChatMessageHistory.messages → list[BaseMessage]
   - → {"query": ..., "history": [...]} → prompt → llm → StrOutputParser → str
   - → 追加写回：HumanMessage(query) + AIMessage(完整回复)

===================================================================================
"""
import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量（API Key 等）
dotenv.load_dotenv()

# store: dict
#   作用：session_id 到历史对象的缓存映射
#   目的：保证同一会话在多次调用间复用同一个 history 实例，避免重复创建文件句柄
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取（或创建）对应的文件型消息历史对象

    参数：
        session_id: str - 会话唯一标识，由 config.configurable.session_id 注入

    返回：
        BaseChatMessageHistory - 具备 messages 读属性与 add_message 写方法的历史对象

    调用时机：
        RunnableWithMessageHistory 在每次 invoke/stream 时自动调用

    FileChatMessageHistory(file_path: str) -> FileChatMessageHistory
      作用：创建基于本地文件的消息历史存储
      参数：file_path - 历史文件路径，不存在时自动创建
      返回：FileChatMessageHistory 实例
      存储格式：JSON 数组，每个元素为序列化后的 BaseMessage
      特点：进程重启后历史依然保留，适合单机演示与轻量场景
    """
    if session_id not in store:
        # 按 session_id 生成独立文件，实现会话隔离与持久化
        store[session_id] = FileChatMessageHistory(f"./chat_history_{session_id}.txt")
    return store[session_id]


# 1.构建提示模板与大语言模型
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：构建含历史插槽的对话提示词模板
#   参数：messages - 消息定义列表（元组或 MessagesPlaceholder）
#   返回：ChatPromptTemplate 实例
prompt = ChatPromptTemplate.from_messages([
    # system 消息：角色设定，每次调用都固定存在，不写入历史
    ("system", "你是一个强大的聊天机器人，请根据用户的需求回复问题。"),

    # MessagesPlaceholder(variable_name: str) -> MessagesPlaceholder
    #   作用：预留历史消息列表插槽
    #   参数：variable_name - 变量名，必须与 history_messages_key 完全一致
    #   注意：放在 system 之后、human 之前，保证对话时序正确
    MessagesPlaceholder("history"),

    # human 消息：本轮用户输入，变量名需与 input_messages_key 一致
    ("human", "{query}"),
])

# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例作为推理引擎
llm = ChatOpenAI(model="deepseek-v4-pro")

# 2.构建链
# LCEL 管道：dict → prompt → Messages → llm → AIMessage → parser → str
# 说明：这是不含历史管理的"纯逻辑链"，历史能力在下一步由包装器注入
chain = prompt | llm | StrOutputParser()

# 3.包装链
# RunnableWithMessageHistory(
#     runnable: Runnable,
#     get_session_history: Callable[..., BaseChatMessageHistory],
#     input_messages_key: str = None,
#     history_messages_key: str = None,
#     output_messages_key: str = None,
# ) -> RunnableWithMessageHistory
#   作用：为基础链附加自动的历史读取与写回能力
#   参数：
#     chain - 被包装的底层 Runnable
#     get_session_history - 历史对象工厂函数，签名为 (session_id) -> BaseChatMessageHistory
#     input_messages_key="query" - 指明输入 dict 中哪个键是"当前用户消息"
#     history_messages_key="history" - 指明历史消息要填入哪个变量
#   返回：包装后的 Runnable，接口与原链一致（invoke/batch/stream）
#   写回规则：
#     - 将 input[input_messages_key] 包装为 HumanMessage 写入历史
#     - 将链的输出包装为 AIMessage 写入历史
with_message_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# 多轮对话交互循环
while True:
    # 4.获取用户的输入
    # input(prompt: str) -> str
    #   作用：阻塞等待用户从标准输入键入内容
    #   返回：用户输入的字符串（不含末尾换行）
    query = input("Human: ")

    # 输入 q 时退出程序
    # exit(code: int) -> NoReturn
    #   作用：终止当前进程，0 表示正常退出
    if query == "q":
        exit(0)

    # 5.运行链并传递配置信息
    # with_message_chain.stream(input: dict, config: dict) -> Iterator[str]
    #   作用：流式执行带历史的对话链，逐块产出 LLM 生成内容
    #   参数：
    #     input - {"query": 用户输入}，键名需与 input_messages_key 一致
    #     config - 运行时配置，configurable.session_id 决定使用哪份历史
    #   返回：生成器，逐个 yield 文本片段
    #   执行流程：
    #     1. 从 config 提取 session_id="muxiaoke"
    #     2. 调用 get_session_history("muxiaoke") 读取 ./chat_history_muxiaoke.txt
    #     3. 历史消息填入 MessagesPlaceholder("history")
    #     4. 逐块流式返回 LLM 输出
    #     5. 流结束后把 HumanMessage(query) 与 AIMessage(完整回复) 追加写回文件
    response = with_message_chain.stream(
        {"query": query},
        config={"configurable": {"session_id": "muxiaoke"}}
    )

    # flush=True 立即刷新输出缓冲，end="" 避免换行，实现打字机效果
    print("AI: ", flush=True, end="")

    # 必须完整消费生成器，否则历史写回可能不触发或内容不完整
    for chunk in response:
        print(chunk, flush=True, end="")
    print("")

# ==================== 最佳实践与扩展写法 ====================
# 1. 生产环境使用 Redis 持久化（支持并发与分布式）
# from langchain_community.chat_message_histories import RedisChatMessageHistory
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     return RedisChatMessageHistory(session_id, url="redis://localhost:6379/0")
#
# 2. 使用数据库存储（便于查询与审计）
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     return SQLChatMessageHistory(session_id, connection_string="sqlite:///chat.db")
#
# 3. 支持多维度会话键（如 user_id + conversation_id）
# from langchain_core.runnables import ConfigurableFieldSpec
# with_message_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_history,          # 签名改为 (user_id, conversation_id)
#     input_messages_key="query",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(id="user_id", annotation=str, name="用户 ID", is_shared=True),
#         ConfigurableFieldSpec(id="conversation_id", annotation=str, name="会话 ID", is_shared=True),
#     ],
# )
# # 调用时：config={"configurable": {"user_id": "u1", "conversation_id": "c1"}}
#
# 4. 使用 invoke 替代 stream（无需实时输出时）
# content = with_message_chain.invoke(
#     {"query": query},
#     config={"configurable": {"session_id": "muxiaoke"}}
# )
# print("AI:", content)
#
# 5. 清空某个会话的历史
# get_session_history("muxiaoke").clear()
