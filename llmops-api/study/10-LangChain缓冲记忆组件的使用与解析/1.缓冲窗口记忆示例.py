#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/23 1:55
@Author  : thezehui@gmail.com
@File    : 1.缓冲窗口记忆示例.py

===================================================================================
知识点讲解：缓冲窗口记忆（trim_messages + RunnableWithMessageHistory）
===================================================================================

1. 什么是缓冲窗口记忆（Buffer Window Memory）
   - 只保留「最近 N 轮」或「最近 N 个 token」的对话，更早的对话直接丢弃
   - 目的：让注入 Prompt 的历史长度恒定可控，避免 token 随轮次无限增长
   - 代价：早期信息被彻底丢弃，模型会对很久以前说过的内容「失忆」
   - 与摘要记忆的区别：窗口记忆「丢弃」旧信息，摘要记忆「压缩」旧信息（见第 11 章）

2. LangChain 1.x 的实现拆分（本示例核心范式）
   旧版由一个 Memory 组件同时负责「存储 + 裁剪 + 注入」，职责混杂且行为隐式。
   1.x 把它拆成三个正交的组件，每个只做一件事：
   - 存储：BaseChatMessageHistory（InMemoryChatMessageHistory / Redis / File...）
   - 裁剪：trim_messages(...)，一个纯函数式的 Runnable
   - 注入：MessagesPlaceholder（Prompt 占位）+ RunnableWithMessageHistory（自动读写）
   好处：可以自由替换任意一层，且每层都能独立单元测试

3. trim_messages 的两种调用形态（关键知识点）
   - 立即执行：trim_messages(messages, max_tokens=..., ...) -> list[BaseMessage]
     第一个位置参数传了消息列表，直接返回裁剪后的结果
   - 延迟执行：trim_messages(max_tokens=..., ...) -> RunnableLambda
     不传消息列表，返回一个 Runnable，可以用 | 接进 LCEL 管道
   本示例用的是第二种，得到的 trimmer 是 RunnableLambda 实例

4. trim_messages 的核心参数详解
   - max_tokens: int        裁剪后允许的最大 token 数（必填）
   - token_counter          token 计数方式，可传：
                              BaseLanguageModel（用模型自带的分词器，最准确）
                              Callable（如 len，按消息条数或字符数近似）
                              "approximate"（内置的快速近似算法）
   - strategy: "first"|"last"
                              "last"  保留最近的消息（缓冲窗口语义，默认）
                              "first" 保留最早的消息（很少用）
   - include_system: bool   是否把 SystemMessage 计入裁剪范围。
                              False 表示 system 消息不参与裁剪，永远保留，
                              这是正确做法（角色设定不能被裁掉）
   - allow_partial: bool    是否允许截断单条消息的内容。
                              False 表示消息要么完整保留、要么整条丢弃
   - start_on: str          裁剪后的第一条消息必须是什么角色。
                              "human" 保证对话以人类消息开头，
                              避免出现以 AIMessage 开头的非法消息序列

5. 为什么 start_on="human" 很重要
   - 大多数模型要求对话序列符合 human/ai 交替出现的格式
   - 若裁剪后恰好以 AIMessage 开头，部分模型（如 Anthropic）会直接报错
   - start_on="human" 会继续往后丢弃，直到第一条是 HumanMessage 为止

6. RunnableWithMessageHistory 的工作机制
   - 把一个普通链包装成「自动带历史」的链，替代旧版的手动 save_context
   - 执行前：按 session_id 调用 get_session_history 取出历史，
             注入到 history_messages_key 指定的键
   - 执行后：自动把本轮的用户输入与 AI 输出 append 回该 session 的历史
   - 三个 key 参数的作用：
       input_messages_key   链输入 dict 中哪个键是「本轮用户输入」
       history_messages_key 链输入 dict 中哪个键用于接收「历史消息列表」
       output_messages_key  链输出 dict 中哪个键是「AI 回复」（输出为 str 时可省略）

7. session_id 与 config 的作用
   - config={"configurable": {"session_id": "xxx"}} 是 LangChain 的运行时配置通道
   - RunnableWithMessageHistory 从中读取 session_id，传给 get_session_history
   - 多用户场景下靠 session_id 隔离记忆，防止串号
   - 忘记传 session_id 会直接抛异常，这是框架的强制约束

8. 本示例的数据流转全过程
   用户输入 "你好"
     ↓ RunnableWithMessageHistory（读历史）
   {"query": "你好", "history": [历史消息列表]}
     ↓ RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
   {"query": "你好", "history": [裁剪后的消息列表]}   ← 注意 history 被覆盖了
     ↓ prompt（MessagesPlaceholder 展开 history）
   [SystemMessage, ...裁剪后的历史..., HumanMessage("你好")]
     ↓ llm → AIMessageChunk 流 → StrOutputParser → str 流
     ↓ RunnableWithMessageHistory（写历史）
   历史中追加 HumanMessage("你好") 与 AIMessage(完整回复)

9. 注意事项
   - assign 中 history 的键名与输入键名相同，会「覆盖」原值，这正是裁剪生效的方式
   - 写回历史的是「未裁剪的完整消息」，裁剪只影响本次发给模型的内容，
     因此 store 中的历史会持续增长，裁剪只是发送时的过滤
   - token_counter=llm 会调用模型的分词器，对部分模型可能产生额外开销
   - InMemoryChatMessageHistory 进程退出即丢失，生产环境请换 Redis 实现

===================================================================================
"""
from operator import itemgetter

import dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建提示模板
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：用「消息列表」的方式构建聊天提示模板，比 from_template 更灵活
#   参数：messages - 列表元素可以是：
#           ("system"|"human"|"ai", "模板字符串") 元组形式
#           MessagesPlaceholder 占位符对象
#           BaseMessage 实例
#   返回：ChatPromptTemplate 实例，invoke 接收 dict，返回 ChatPromptValue
#
#   MessagesPlaceholder(variable_name: str) -> MessagesPlaceholder
#     作用：在消息序列中预留一个「消息列表」的插槽
#     参数：variable_name - 对应链输入 dict 中的键名，这里是 "history"
#     关键点：它接收的是 List[BaseMessage]（消息对象列表），不是字符串。
#             渲染时会把列表中的每条消息「展开」成独立的一条消息，
#             而不是拼成一个字符串塞进某条消息里
#     可选参数：optional=True 表示该变量缺失时用空列表兜底，不报错
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是DeepSeek开发的聊天机器人，请根据对应的上下文回复用户问题"),
    MessagesPlaceholder("history"),  # 需要的history其实是一个列表
    ("human", "{query}"),
])

# 2.创建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建 OpenAI 兼容协议的聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，实现 Runnable 协议
#   双重用途：这里的 llm 既作为对话模型，又作为下面 trimmer 的 token_counter，
#             用模型自带的分词器计数，比字符数近似准确得多
llm = ChatOpenAI(model="deepseek-v4-pro")

# 3.LangChain 1.x 写法：历史存储交给 ChatMessageHistory，裁剪交给 trim_messages
# strategy="last" 保留最近的消息，等价于缓冲窗口/令牌缓冲记忆
#
# trim_messages(*, max_tokens, token_counter, strategy, ...) -> RunnableLambda
#   作用：创建一个「消息裁剪器」，把过长的历史裁剪到 token 上限以内
#   调用形态：这里没有传第一个位置参数 messages，因此返回的是 RunnableLambda，
#             可以直接用管道操作符 | 接入 LCEL 链中延迟执行
#   参数逐一说明：
#     max_tokens=200      裁剪后历史允许的最大 token 数，超出部分会被丢弃
#     strategy="last"     保留「最近」的消息（从后往前保留），实现缓冲窗口语义。
#                         若为 "first" 则保留最早的消息
#     token_counter=llm   使用 llm 的分词器计算 token 数，
#                         内部调用 llm.get_num_tokens_from_messages(messages)。
#                         也可传 len（按条数近似）或 "approximate"（快速近似）
#     include_system=False  SystemMessage 不参与裁剪，永远完整保留。
#                           角色设定必须保留，否则模型行为会漂移
#     allow_partial=False   不允许截断单条消息内容，消息要么整条保留要么整条丢弃。
#                           设为 True 时会切断最边界那条消息的文本
#     start_on="human"      保证裁剪结果的第一条是 HumanMessage，
#                           维持 human/ai 交替的合法对话格式，
#                           避免以 AIMessage 开头导致某些模型报错
#   返回：RunnableLambda 实例，invoke 接收消息列表，返回裁剪后的消息列表
trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=llm,
    include_system=False,
    allow_partial=False,
    start_on="human",
)

# 构建核心链
# RunnablePassthrough.assign(**kwargs) -> RunnableAssign
#   作用：保留输入 dict 的全部原有字段，同时计算并追加/覆盖指定字段
#   参数：history=itemgetter("history") | trimmer
#     - itemgetter("history")：从输入 dict 中取出 history 字段（原始完整历史）
#     - | trimmer：把取出的消息列表交给裁剪器处理
#     - 组合后是一个 Runnable：dict -> list[BaseMessage]（已裁剪）
#   关键点：新字段名 "history" 与输入中已有的 "history" 同名，
#           因此结果是「用裁剪后的历史覆盖原始历史」，这正是裁剪生效的方式
#   返回：RunnableAssign 实例，输出 {"query": ..., "history": 裁剪后的列表}
#
# 完整链的组成与数据流转：
#   {"query": str, "history": [完整历史]}
#     → RunnableAssign  → {"query": str, "history": [裁剪后历史]}
#     → prompt          → ChatPromptValue（含展开后的历史消息）
#     → llm             → AIMessage / AIMessageChunk 流
#     → StrOutputParser → str / str 流
chain = RunnablePassthrough.assign(
    history=itemgetter("history") | trimmer
) | prompt | llm | StrOutputParser()

# 会话历史仓库：以 session_id 为键，隔离不同用户/会话的消息历史
# 类型注解 dict[str, BaseChatMessageHistory] 说明值是历史存储接口的任意实现，
# 因此把 InMemoryChatMessageHistory 换成 RedisChatMessageHistory 无需改动其他代码
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    get_session_history(session_id: str) -> BaseChatMessageHistory
      作用：会话历史工厂函数，根据 session_id 获取（或懒创建）对应的历史存储

      参数：
        session_id: str - 会话唯一标识，由 config["configurable"]["session_id"] 传入

      返回：
        BaseChatMessageHistory - 该会话的消息历史存储实例

      调用时机：
        由 RunnableWithMessageHistory 在每次 invoke/stream 时自动调用，
        业务代码无需手动调用（本示例末尾的调用仅用于打印观察）

      设计要点：
        1. 懒加载：首次访问某个 session_id 时才创建实例，避免预分配浪费
        2. 幂等：同一个 session_id 必须始终返回「同一个实例」，
                 否则历史无法累积（每次都是新的空历史）
        3. 可替换：只需改这里的构造函数即可切换存储后端，例如
                   return RedisChatMessageHistory(session_id, url="redis://...")
    """
    # 懒创建：只在该会话第一次出现时初始化历史存储
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 4.包装成带历史的链，输入输出会自动写回消息历史，无需手动 save_context
# RunnableWithMessageHistory(runnable, get_session_history, *, input_messages_key,
#                            history_messages_key, output_messages_key=None) -> 实例
#   作用：为任意链自动装配「读历史 → 执行 → 写历史」的能力
#   参数：
#     runnable: Runnable - 被包装的业务链，这里是 chain
#     get_session_history: Callable[[str], BaseChatMessageHistory]
#                          - 历史工厂函数，框架会用 session_id 调用它
#     input_messages_key="query"
#                          - 指明链输入 dict 中哪个键承载「本轮用户输入」，
#                            框架据此把用户输入包装成 HumanMessage 写回历史
#     history_messages_key="history"
#                          - 指明把读取到的历史消息列表注入到输入 dict 的哪个键，
#                            必须与 Prompt 中 MessagesPlaceholder 的名字一致
#     output_messages_key  - 指明链输出 dict 中哪个键是 AI 回复。
#                            本链输出是纯 str（StrOutputParser），
#                            框架能自动识别，因此可以省略
#   返回：RunnableWithMessageHistory 实例，本身仍是 Runnable，
#         支持 invoke / stream / batch 等全部方法
#
#   自动化的两个动作（替代旧版手动 memory.save_context）：
#     执行前：history = get_session_history(session_id).messages，注入到 "history" 键
#     执行后：history.add_user_message(query) + history.add_ai_message(输出)
#
#   重要：写回历史的是「未裁剪的原始输入输出」，
#         trimmer 只影响本次发送给模型的内容，store 中的历史仍会持续增长
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# 运行时配置：通过 configurable 通道传递 session_id
#   config 结构：{"configurable": {"session_id": "会话标识"}}
#   RunnableWithMessageHistory 会从这里读取 session_id 并传给 get_session_history
#   注意：这是框架的强制要求，缺少 session_id 会抛出 ValueError
#   多用户场景下应传入真实的用户/会话标识，如 f"user_{user_id}_{conversation_id}"
config = {"configurable": {"session_id": "muxiaoke"}}

# 5.死循环构建对话命令行
while True:
    # 读取用户输入，阻塞等待
    query = input("Human: ")

    # 退出条件：输入 q 直接终止进程
    # 注意：InMemoryChatMessageHistory 存在内存中，退出后历史全部丢失
    if query == "q":
        exit(0)

    # chain_with_history.stream(input: dict, config: dict) -> Iterator[str]
    #   作用：以流式方式执行带历史的链
    #   参数：
    #     input: dict - 只需提供 {"query": ...}，
    #                   history 字段由 RunnableWithMessageHistory 自动注入
    #     config: dict - 运行时配置，必须包含 configurable.session_id
    #   返回：Iterator[str] - 生成器，逐块产出解析后的字符串增量
    #   说明：stream 是惰性的，不遍历就不会真正发起请求，
    #         历史写回也发生在流被完全消费之后
    response = chain_with_history.stream({"query": query}, config=config)

    print("AI: ", flush=True, end="")
    # 消费生成器：逐块打印增量内容
    #   flush=True 立即刷新缓冲区，实现打字机效果
    #   end="" 取消默认换行，让增量内容连续拼接显示
    for chunk in response:
        print(chunk, flush=True, end="")
    print("")

    # 打印当前会话的完整历史，用于观察记忆的累积过程
    # get_session_history("muxiaoke").messages -> List[BaseMessage]
    #   注意：这里打印的是「未裁剪的完整历史」，会随轮次持续增长。
    #         而实际发给模型的是经 trimmer 裁剪后的子集，
    #         对比两者可以直观理解「存储全量、发送裁剪」的设计
    print("history: ", get_session_history("muxiaoke").messages)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# 由 Memory 组件同时持有历史并负责裁剪，需要手动 load_memory_variables / save_context
#
# 对照阅读要点（理解 1.x 的演进动机）：
#   - 旧版 ConversationTokenBufferMemory 一个组件干三件事：存历史、算 token、裁剪
#     新版拆成 ChatMessageHistory（存）+ trim_messages（裁）+ RunnableWithMessageHistory（注入）
#   - 旧版必须手动调 memory.save_context 写回，忘记调用就没有记忆，是常见 bug
#     新版由 RunnableWithMessageHistory 自动写回，不会遗漏
#   - 旧版通过 RunnableLambda(memory.load_memory_variables) 把 Memory 硬接进链，
#     返回 dict 还需再用 itemgetter("history") 取值，写法冗长
#   - 旧版没有 session_id 概念，一个 Memory 实例只服务一个会话，
#     多用户场景需要自己维护 dict[user_id, Memory]，容易串号
#
# from langchain_classic.memory import ConversationTokenBufferMemory
# from langchain_core.runnables import RunnableLambda
#
# # ConversationTokenBufferMemory(max_token_limit, return_messages, input_key, llm)
# #   return_messages=True - 返回消息对象列表（配合 MessagesPlaceholder），
# #                          False 则返回拼接好的字符串
# #   input_key="query"    - save_context 时从 inputs 的哪个键取用户输入
# #   llm=...              - 用于计算 token 数的模型，等价于新版的 token_counter
# memory = ConversationTokenBufferMemory(
#     return_messages=True,
#     input_key="query",
#     llm=ChatOpenAI(model="deepseek-v4-pro"),
# )
#
# # RunnableLambda(memory.load_memory_variables) | itemgetter("history")
# #   load_memory_variables({}) 返回 {"history": [...]}，
# #   因此需要再用 itemgetter("history") 从 dict 中取出列表
# chain = RunnablePassthrough.assign(
#     history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
# ) | prompt | llm | StrOutputParser()
#
# while True:
#     query = input("Human: ")
#
#     if query == "q":
#         exit(0)
#
#     chain_input = {"query": query, "language": "中文"}
#
#     response = chain.stream(chain_input)
#     print("AI: ", flush=True, end="")
#     # 旧版必须自己累积完整输出，才能写回 Memory
#     output = ""
#     for chunk in response:
#         output += chunk
#         print(chunk, flush=True, end="")
#     # memory.save_context(inputs: dict, outputs: dict) -> None
#     #   手动写回本轮问答，忘记调用则完全没有记忆效果
#     memory.save_context(chain_input, {"output": output})
#     print("")
#     print("history: ", memory.load_memory_variables({}))

# ===================================================================================
# 最佳实践与其他用法
# ===================================================================================
#
# 1. 按「消息条数」而非 token 数做窗口（等价于旧版 ConversationBufferWindowMemory）：
#    trimmer = trim_messages(
#        max_tokens=6,              # 这里的单位由 token_counter 决定
#        strategy="last",
#        token_counter=len,         # len 按「消息条数」计数
#        include_system=False,
#        start_on="human",
#    )
#    说明：token_counter=len 时 max_tokens 实际表示「保留最近 6 条消息」
#
# 2. 用快速近似计数替代模型分词器（省去一次分词开销，适合高并发）：
#    trimmer = trim_messages(max_tokens=200, strategy="last",
#                            token_counter="approximate", start_on="human")
#
# 3. 切换成持久化存储（仅改工厂函数一行，其余代码不变）：
#    from langchain_community.chat_message_histories import RedisChatMessageHistory
#    def get_session_history(session_id: str) -> BaseChatMessageHistory:
#        return RedisChatMessageHistory(session_id=session_id,
#                                       url="redis://localhost:6379", ttl=3600)
#
# 4. 非流式调用（一次性拿到完整结果）：
#    content = chain_with_history.invoke({"query": "你好"}, config=config)
#
# 5. 多会话切换（同一个链服务多个用户，靠 session_id 隔离）：
#    chain_with_history.invoke({"query": "我是谁"},
#                              config={"configurable": {"session_id": "user_a"}})
#    chain_with_history.invoke({"query": "我是谁"},
#                              config={"configurable": {"session_id": "user_b"}})
#    两个会话的历史互不可见
#
# 6. 同时裁剪与保留系统消息（推荐配置，本示例已采用）：
#    include_system=False 保证 SystemMessage 永不被裁掉，
#    否则长对话后角色设定丢失，模型行为会明显漂移
#
# 7. 常见错误与解决方案：
#    - 错误：ValueError: Missing keys ['session_id'] in config['configurable']
#      原因：invoke/stream 时忘记传 config 或 config 中缺少 session_id
#      解决：始终传 config={"configurable": {"session_id": "..."}}
#
#    - 错误：Input to ChatPromptTemplate is missing variables {'history'}
#      原因：history_messages_key 与 MessagesPlaceholder 的变量名不一致
#      解决：确保两处名字完全相同，或给 MessagesPlaceholder 加 optional=True
#
#    - 问题：历史裁剪似乎没有生效，token 仍在增长
#      说明：这是预期行为。store 中始终保存全量历史，
#            trimmer 只裁剪「本次发给模型」的内容。
#            若想让存储本身也不增长，需在写回后主动裁剪 history.messages
#
#    - 问题：模型报错「messages must start with a user message」
#      原因：裁剪后第一条恰好是 AIMessage
#      解决：加上 start_on="human"（本示例已配置）
#
#    - 问题：多个用户的对话内容互相串了
#      原因：所有请求用了同一个硬编码的 session_id
#      解决：用真实用户标识动态生成 session_id
#
#    - 错误：TypeError: 'RunnableLambda' object is not iterable
#      原因：误以为 trim_messages 总是返回列表
#      说明：不传 messages 位置参数时返回的是 RunnableLambda（延迟执行），
#            要立即得到列表请写 trim_messages(messages, max_tokens=..., ...)
