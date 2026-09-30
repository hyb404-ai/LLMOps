#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 9:22
@Author  : thezehui@gmail.com
@File    : 1.对话消息历史组件基础.py

===================================================================================
知识点讲解：ChatMessageHistory 对话消息历史组件
===================================================================================

1. 记忆功能的两个核心问题与设计动机
   - 把记忆功能集成到 LLM 应用，涉及两个核心问题：存储的历史信息是什么？如何检索与处理历史信息？
   - 一个记忆类由「历史消息的存储」与「记忆的处理逻辑」两部分组成，LangChain 采用同样的思路
   - 朴素写法里「检索历史」和「存储历史」逻辑常混在一起；LangChain 用专门组件把「存储」职责单独抽离，便于替换与测试

2. 什么是 BaseChatMessageHistory
   - LangChain 封装的管理历史信息的抽象基类，所有扩展的消息历史组件（含自定义组件）均继承它
   - 职责单一：只管消息的增（add_*）、查（messages）、删（clear），不参与 Prompt 拼装
   - 在 LangChain 1.x 中，它取代了旧的 Memory 体系，成为记忆能力的唯一存储底座

3. BaseChatMessageHistory 的核心接口
   - messages: List[BaseMessage]        属性，读取全部历史消息
   - add_user_message / add_ai_message   追加人类 / AI 消息
   - add_message(message)                追加任意类型的单条消息
   - add_messages(messages)              批量追加消息列表
   - clear()                             清空全部历史
   - aget_messages / aadd_messages       对应的异步版本

4. 消息类型体系（langchain_core.messages）
   - SystemMessage  系统指令，设定 AI 角色与行为准则
   - HumanMessage   人类用户输入
   - AIMessage      AI 回复，可携带 tool_calls
   - ToolMessage    工具执行结果，需与 tool_call_id 对应
   - FunctionMessage 旧版函数调用结果（已被 ToolMessage 取代）
   - 所有类型都继承自 BaseMessage，含 content、additional_kwargs、response_metadata

5. InMemoryChatMessageHistory 的特点
   - 位于 langchain_core.chat_history，是最轻量的内存实现，也是 langchain_core 内置的对话消息历史类
   - 底层就是一个 Python list，进程退出后数据即丢失
   - 适用：单元测试、教学演示、单次会话的临时缓存
   - 不适用：多进程/多实例部署、需要持久化的生产环境

6. 常见的持久化实现（langchain_community.chat_message_histories）
   - 第三方集成的记忆组件均通过 langchain_community 导入（与内置的 InMemory 不同）
   - FileChatMessageHistory        存到本地 JSON 文件（见本章第 2 个示例）
   - RedisChatMessageHistory       存到 Redis，支持 TTL 过期，生产最常用
   - SQLChatMessageHistory / PostgresChatMessageHistory / MongoDBChatMessageHistory  存到各类数据库
   - 它们的 API 完全一致，切换存储只需替换构造函数，业务代码无需改动

7. 与 Memory 组件的区别（版本演进）
   - LangChain 0.x：Memory 既存历史，又隐式改写链的输入输出，行为不透明
   - LangChain 1.x：ChatMessageHistory 只存历史，注入 Prompt 由 MessagesPlaceholder + RunnableWithMessageHistory 显式完成
   - 好处：职责清晰、行为可预测、便于测试与替换存储后端

8. 典型使用方式
   - 手动注入：chain.invoke({"query": q, "history": chat_history.messages})
   - 自动读写：用 RunnableWithMessageHistory 包装链，按 session_id 自动存取

9. 注意事项
   - messages 返回的是消息对象列表，不是字符串；转字符串用 get_buffer_string(messages)
   - 该组件本身不做任何裁剪，历史会无限增长，需配合 trim_messages 控制 token
   - InMemoryChatMessageHistory 非线程安全，并发写入需自行加锁

===================================================================================
"""
from langchain_core.chat_history import InMemoryChatMessageHistory

# InMemoryChatMessageHistory() -> InMemoryChatMessageHistory
#   作用：创建一个基于内存列表的对话消息历史存储实例
#   参数：无必填参数（可选传 messages=[...] 提供初始消息）
#   返回：InMemoryChatMessageHistory 实例，实现了 BaseChatMessageHistory 接口
#   存储位置：实例内部的 self.messages 列表，进程退出即丢失
chat_history = InMemoryChatMessageHistory()

# add_user_message(message: str | HumanMessage) -> None
#   作用：向历史中追加一条人类消息
#   参数：message - 传入 str 时会被自动包装成 HumanMessage(content=message)，
#                   也可以直接传入 HumanMessage 对象
#   返回：None
#   副作用：messages 列表末尾新增一个 HumanMessage 元素
chat_history.add_user_message("你好，我是慕小课，你是谁？")

# add_ai_message(message: str | AIMessage) -> None
#   作用：向历史中追加一条 AI 回复消息
#   参数：message - 传入 str 时自动包装成 AIMessage(content=message)
#   返回：None
#   副作用：messages 列表末尾新增一个 AIMessage 元素
#   说明：一轮完整问答对应两条消息（Human + AI），
#         等价于 LangChain 0.x 中 memory.save_context({"input":...}, {"output":...})
chat_history.add_ai_message("你好，我是DeepSeek，有什么可以帮到您的？")

# messages 属性 -> List[BaseMessage]
#   作用：读取当前存储的全部历史消息
#   返回：消息对象列表，本例输出
#         [HumanMessage(content='你好，我是慕小课，你是谁？'),
#          AIMessage(content='你好，我是DeepSeek，有什么可以帮到您的？')]
#   用途：可直接作为 MessagesPlaceholder 的入参传给 Prompt 模板
print(chat_history.messages)

# ===================================================================================
# 最佳实践与其他用法
# ===================================================================================
#
# 1. 批量追加消息（比逐条 add 更高效，持久化实现下可减少 IO 次数）：
#    from langchain_core.messages import HumanMessage, AIMessage
#    chat_history.add_messages([
#        HumanMessage(content="今天天气怎么样？"),
#        AIMessage(content="今天晴，气温 25 度。"),
#    ])
#
# 2. 追加任意类型消息（如系统消息）：
#    from langchain_core.messages import SystemMessage
#    chat_history.add_message(SystemMessage(content="你是一个专业的技术顾问"))
#
# 3. 把消息列表转成纯文本（计算 token 或拼装原生 SDK 请求时常用）：
#    from langchain_core.messages import get_buffer_string
#    print(get_buffer_string(chat_history.messages))
#    # 输出："Human: 你好，我是慕小课，你是谁？\nAI: 你好，我是DeepSeek..."
#
# 4. 清空历史（开启新会话时使用）：
#    chat_history.clear()
#    print(chat_history.messages)  # []
#
# 5. 接入 Prompt 模板（推荐写法，配合 MessagesPlaceholder）：
#    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#    prompt = ChatPromptTemplate.from_messages([
#        ("system", "你是一个聊天机器人"),
#        MessagesPlaceholder("history"),
#        ("human", "{query}"),
#    ])
#    chain = prompt | llm | StrOutputParser()
#    chain.invoke({"query": "我是谁？", "history": chat_history.messages})
#
# 6. 切换成持久化存储（API 完全一致，仅替换构造函数）：
#    from langchain_community.chat_message_histories import RedisChatMessageHistory
#    chat_history = RedisChatMessageHistory(session_id="user_1001", url="redis://localhost:6379")
#
# 7. 常见错误与解决方案：
#    - 错误：TypeError，把 messages 当成字符串直接拼进 f-string
#      原因：messages 是 BaseMessage 对象列表，不是字符串
#      解决：用 get_buffer_string(messages) 转换，或交给 MessagesPlaceholder 处理
#
#    - 问题：重启程序后历史丢失
#      原因：InMemoryChatMessageHistory 只存在内存中
#      解决：改用 FileChatMessageHistory / RedisChatMessageHistory 等持久化实现
#
#    - 问题：历史越来越长导致超出上下文窗口
#      原因：该组件不做任何自动裁剪
#      解决：使用 trim_messages 裁剪（见第 10 章）或摘要压缩（见第 11 章）
