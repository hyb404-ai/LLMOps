#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 23:53
@Author  : thezehui@gmail.com
@File    : 1.BaseChatMemory运行流程解析.py

===================================================================================
知识点讲解：Memory 运行流程与 LangChain 1.x 的记忆范式演进
===================================================================================

1. LangChain 0.x 的 Memory 体系（已废弃）
   - 核心抽象：BaseMemory → BaseChatMemory → 各种具体 Memory 子类
   - BaseMemory 基类封装方法：memory_variables、load_memory_variables、aload_memory_variables、save_context、asave_context、clear、aclear
   - 两个核心方法（子类关键）：load_memory_variables(inputs) 链执行前调用，把历史注入链输入；save_context(inputs, outputs) 链执行后调用，写回本轮问答
   - 关键字段：chat_memory（内部持有的 BaseChatMessageHistory，真正的存储载体）、memory_variables（声明注入变量名）、input_key / output_key（从输入输出字典取对应值）、return_messages（True 返回消息列表，False 返回拼接纯文本）
   - 问题：Memory 隐式改写链的输入输出，行为不透明，难测试难调试

2. 衍生子类：SimpleMemory 与 BaseChatMemory
   - SimpleMemory：实现了记忆方法但不存储任何内容，可在"不需要记忆又不想改代码结构"时平替
   - BaseChatMemory：内置其他记忆组件的基类，针对聊天模型对话场景封装了历史

3. LangChain 1.x 的新范式（本示例采用）
   - 公式：记忆 = 消息历史存储（BaseChatMessageHistory）+ 链上显式注入
   - 存储层：InMemory / File / Redis 等，只管存取
   - 注入层：MessagesPlaceholder 占位，RunnableWithMessageHistory 自动读写
   - 裁剪层：trim_messages、SummarizationMiddleware
   - 好处：每一步显式可见、职责单一、易替换与单元测试

4. 新旧 API 对应关系（迁移对照表）
   旧写法                                      新写法
   memory.save_context({"input":q},{"output":a})  add_user_message(q)+add_ai_message(a)
   memory.load_memory_variables({})                chat_history.messages
   memory.clear()                                  chat_history.clear()
   memory.chat_memory                              chat_history 本身
   ConversationBufferWindowMemory                  trim_messages(strategy="last")
   ConversationSummaryBufferMemory                 SummarizationMiddleware
   ConversationEntityMemory                        with_structured_output 显式抽取

5. BaseChatMemory 为什么是抽象基类（源码要点）
   - memory_variables 与 load_memory_variables 是抽象方法，无默认实现
   - 其内部 save_context 本质就是 self.chat_memory.add_messages([HumanMessage, AIMessage])
   - clear 本质调用 self.chat_memory.clear()
   - 不同记忆策略注入的变量名与加载逻辑差异大，必须由子类各自定义，故不能直接实例化

6. 一轮问答对应两条消息
   - add_user_message + add_ai_message 合起来等价于旧版一次 save_context
   - 顺序重要：先 user 后 ai，否则对话顺序错乱
   - 批量写法 add_messages([HumanMessage(...), AIMessage(...)]) 更高效，持久化下可合并两次 IO 为一次

7. 历史在链上的三种注入方式
   - 手动注入：chain.invoke({"query": q, "chat_history": chat_history.messages})
   - 自动读写：RunnableWithMessageHistory 按 session_id 自动存取（推荐）
   - Agent 场景：LangGraph 的 checkpointer 按 thread_id 持久化全部状态

8. clear() 的使用场景
   - 用户点"新建对话"重置上下文；会话超时或换话题时主动清理
   - 注意：对持久化实现（FileChatMessageHistory）clear 会真正删除磁盘数据

9. 注意事项
   - messages 返回的是消息对象列表而非字符串，不要直接拼进 f-string
   - 该组件不做任何自动裁剪，长对话必须配合 trim_messages 或摘要
   - 多用户场景务必按 session_id / user_id 隔离历史，防止记忆串号

===================================================================================
"""
from langchain_core.chat_history import InMemoryChatMessageHistory

# LangChain 1.x 写法：Memory 体系已被 BaseChatMessageHistory 取代，
# 记忆 = "消息历史存储" + "链上显式注入"，不再由 Memory 隐式改写链的输入输出
#
# InMemoryChatMessageHistory() -> InMemoryChatMessageHistory
#   作用：创建基于内存列表的消息历史存储
#   参数：无必填参数（可选 messages=[...] 传入初始消息）
#   返回：InMemoryChatMessageHistory 实例，实现 BaseChatMessageHistory 接口
#   对应旧版：等价于旧版 Memory 内部持有的 memory.chat_memory 对象
chat_history = InMemoryChatMessageHistory()

# 1.还没写入任何上下文，此时历史为空
# messages 属性 -> List[BaseMessage]
#   作用：读取当前全部历史消息
#   返回：消息对象列表，此时为空列表 []
#   对应旧版：等价于 memory.load_memory_variables({}) 返回 {"chat_history": []}
print("保存前:", chat_history.messages)

# 2.一轮问答对应两条消息，等价于旧版的 save_context
# add_user_message(message: str | HumanMessage) -> None
#   作用：追加一条人类消息到历史
#   参数：message - str 会被自动包装成 HumanMessage(content=message)
#   返回：None
#   副作用：messages 列表末尾新增一个 HumanMessage
chat_history.add_user_message("你好，我是慕小课你是谁")

# add_ai_message(message: str | AIMessage) -> None
#   作用：追加一条 AI 回复消息到历史
#   参数：message - str 会被自动包装成 AIMessage(content=message)
#   返回：None
#   副作用：messages 列表末尾新增一个 AIMessage
#   等价旧版：上面两行合起来 == memory.save_context(
#             {"query": "你好，我是慕小课你是谁"},
#             {"output": "你好，我是DeepSeek,有什么可以帮到您的"})
chat_history.add_ai_message("你好，我是DeepSeek,有什么可以帮到您的")

# 也可以一次性批量追加
# add_messages(messages: Sequence[BaseMessage]) -> None
#   作用：批量追加多条消息
#   参数：messages - BaseMessage 对象序列
#   返回：None
#   优势：持久化实现下只触发一次 IO，比逐条 add 更高效
# chat_history.add_messages([HumanMessage("..."), AIMessage("...")])

# 读取写入后的历史，输出：
# [HumanMessage(content='你好，我是慕小课你是谁'), AIMessage(content='你好，我是DeepSeek,有什么可以帮到您的')]
print("保存后:", chat_history.messages)

# 3.链上使用时，历史直接作为 MessagesPlaceholder 的入参传入
# 手动注入写法：chat_history.messages 作为 chat_history 变量传给链
# content = chain.invoke({"query": "你好", "chat_history": chat_history.messages})
#
# 数据流转：chat_history.messages → MessagesPlaceholder("chat_history")
#           → Prompt 渲染出含历史的完整消息列表 → llm → AIMessage
#
# 更推荐交给 RunnableWithMessageHistory 自动读写(见第 13 章对话链示例)
# 它会在链执行前自动读取历史注入，执行后自动把本轮问答写回，无需手动 add_*

# 4.清空历史
# clear() -> None
#   作用：清空全部历史消息
#   参数：无
#   返回：None
#   副作用：messages 列表被置空；持久化实现下会真正删除存储的数据
#   对应旧版：等价于 memory.clear()
#   使用场景：用户新建对话、切换话题、会话超时重置
chat_history.clear()
print("清空后:", chat_history.messages)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# BaseChatMemory 是抽象基类(ABC)，memory_variables / load_memory_variables
# 由子类实现，所以这里定义一个最小实现，用来观察它的运行流程
#
# 阅读要点（对照上面的新写法理解演进）：
#   - memory_variables 声明注入变量名，新写法中由 MessagesPlaceholder 的名字承担
#   - load_memory_variables 读历史，新写法中直接用 chat_history.messages
#   - save_context 写历史，新写法中由 add_user_message + add_ai_message 承担
#   - input_key / output_key 指定从哪个键取值，新写法中由
#     RunnableWithMessageHistory 的 input_messages_key / history_messages_key 承担
#
# from langchain_classic.memory.chat_memory import BaseChatMemory
#
#
# class SimpleChatMemory(BaseChatMemory):
#     @property
#     def memory_variables(self) -> list:
#         # 该记忆会向链的输入里注入哪些变量名
#         # 抽象属性，必须由子类实现
#         return ["chat_history"]
#
#     def load_memory_variables(self, inputs: dict) -> dict:
#         # 从 chat_memory 读取历史，return_messages 决定返回消息对象还是纯文本
#         # 抽象方法，必须由子类实现
#         # 参数：inputs: dict - 链的原始输入，部分 Memory 会据此做条件加载（如向量检索记忆）
#         # 返回：dict - 键名必须与 memory_variables 声明的一致
#         return {self.memory_variables[0]: list(self.chat_memory.messages)}
#
#
# # SimpleChatMemory(input_key, output_key, return_messages) -> 实例
# #   input_key: str - save_context 时从 inputs 的哪个键取用户输入
# #   output_key: str - save_context 时从 outputs 的哪个键取 AI 回复
# #   return_messages: bool - True 返回消息对象列表，False 返回拼接好的字符串
# memory = SimpleChatMemory(input_key="query", output_key="output", return_messages=True)
#
# # load_memory_variables(inputs: dict) -> dict
# #   作用：加载记忆变量，链执行前由框架自动调用
# #   返回：{"chat_history": []}
# memory_variable = memory.load_memory_variables({})
# print("保存前:", memory_variable)
#
# # save_context(inputs: dict, outputs: dict) -> None
# #   作用：保存一轮问答，链执行后由框架自动调用
# #   参数：inputs - 按 input_key 取值；outputs - 按 output_key 取值
# #   返回：None
# #   副作用：内部转成 HumanMessage/AIMessage 存入 chat_memory
# memory.save_context({"query": "你好，我是慕小课你是谁"}, {"output": "你好，我是DeepSeek,有什么可以帮到您的"})
#
# memory_variable = memory.load_memory_variables({})
# print("保存后:", memory_variable)

# ===================================================================================
# 最佳实践与常见问题
# ===================================================================================
#
# 1. 推荐的链上自动读写写法（无需手动 add_*，见第 10 章完整示例）：
#    from langchain_core.runnables.history import RunnableWithMessageHistory
#    store = {}
#    def get_session_history(session_id: str):
#        if session_id not in store:
#            store[session_id] = InMemoryChatMessageHistory()
#        return store[session_id]
#
#    chain_with_history = RunnableWithMessageHistory(
#        chain, get_session_history,
#        input_messages_key="query", history_messages_key="history",
#    )
#    chain_with_history.invoke({"query": "你好"},
#                              config={"configurable": {"session_id": "u1"}})
#
# 2. 历史转纯文本（计算 token、拼原生 SDK 请求时使用）：
#    from langchain_core.messages import get_buffer_string
#    print(get_buffer_string(chat_history.messages))
#
# 3. 控制历史长度（见第 10 章）：
#    from langchain_core.messages import trim_messages
#    trimmed = trim_messages(chat_history.messages, max_tokens=200,
#                            strategy="last", token_counter=llm)
#
# 4. 常见错误与解决方案：
#    - 错误：TypeError: Can't instantiate abstract class BaseChatMemory
#      原因：BaseChatMemory 是 ABC，memory_variables 等抽象成员未实现
#      解决：定义子类实现抽象方法，或直接改用 1.x 的 ChatMessageHistory 方案
#
#    - 错误：ModuleNotFoundError: No module named 'langchain_classic'
#      原因：LangChain 1.x 已把旧 Memory 迁移到独立包
#      解决：pip install langchain-classic，但更建议迁移到新范式
#
#    - 问题：历史顺序错乱，模型理解混乱
#      原因：add_ai_message 写在了 add_user_message 之前
#      解决：严格保证「先 user 后 ai」的写入顺序
#
#    - 问题：多个用户共用了同一份历史
#      原因：全局只创建了一个 chat_history 实例
#      解决：按 session_id 维护 dict[str, BaseChatMessageHistory] 做隔离
