#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/23 1:55
@Author  : thezehui@gmail.com
@File    : 1.摘要缓冲混合记忆示例.py

===================================================================================
知识点讲解：摘要缓冲混合记忆（SummarizationMiddleware + Agent + Checkpointer）
===================================================================================

1. 什么是摘要缓冲混合记忆（Summary Buffer Memory）
   - 把对话历史切成两部分：
       摘要区：早期对话被 LLM 压缩成一段摘要文本（长期记忆，token 恒定）
       缓冲区：近期对话保留完整原文（短期记忆，细节无损）
   - 触发机制：当历史超过阈值时，把最早的部分交给 LLM 压缩进摘要，并从缓冲区移除
   - 定位：在「信息保真度」与「token 成本」之间取得平衡，长对话场景的最佳方案

2. 三种记忆策略的对比（贯穿第 7、10、11 章）
   策略             早期信息      token 成本      额外 LLM 调用
   全量缓冲         完整保留      线性增长        无
   窗口裁剪（第10章）直接丢弃      恒定            无
   摘要缓冲（本章）  压缩保留      恒定            有（每次触发摘要都要调模型）
   - 窗口裁剪省钱但会失忆，摘要缓冲不失忆但每次压缩要额外花一次模型调用
   - 第 7 章用原生 SDK 手写过这套逻辑，本章展示 LangChain 1.x 的内置实现

3. LangChain 1.x 的实现范式（本示例核心）
   旧版由 ConversationSummaryBufferMemory 一个组件隐式完成全部工作。
   1.x 改为 Agent + Middleware + Checkpointer 三层协作：
   - create_agent    构建 Agent 执行图（基于 LangGraph）
   - Middleware      在 Agent 每步执行前后插入横切逻辑，摘要就是一种 middleware
   - Checkpointer    负责状态（含完整消息历史）的持久化，线程即会话
   好处：摘要逻辑与业务链解耦，可与其他 middleware（限流、审计、工具过滤）自由组合

4. AgentMiddleware 中间件机制
   - SummarizationMiddleware 继承自 AgentMiddleware，是官方内置中间件之一
   - 工作位置：在把消息发给模型之前拦截，检查是否触发摘要条件
   - 触发后动作：调用摘要模型压缩早期消息，用摘要消息替换原始消息序列
   - 该替换会写入 Agent 状态，因此摘要是「持久化的」，不是每次重新计算

5. SummarizationMiddleware 的核心参数
   - model              执行摘要任务的模型，可与对话模型不同
                        （常见优化：对话用强模型，摘要用便宜的小模型）
   - trigger            触发摘要的条件，支持三种单位的元组：
                          ("tokens", 300)     token 数超过 300 时触发
                          ("messages", 20)    消息条数超过 20 条时触发
                          ("fraction", 0.8)   占用上下文窗口 80% 时触发
                        也可传条件列表，任一满足即触发
   - keep               摘要后保留多少原文（缓冲区大小），同样支持三种单位：
                          ("messages", 4)     保留最近 4 条消息原文（默认 20 条）
                          ("tokens", 500)     保留最近约 500 token 的原文
   - token_counter      token 计数函数，默认 count_tokens_approximately（快速近似）
   - summary_prompt     摘要提示词模板，内置模板会引导模型输出结构化摘要
                        （含 SESSION INTENT / SUMMARY / ARTIFACTS / NEXT STEPS 分节）
   - trim_tokens_to_summarize
                        送入摘要模型的消息本身也要限长，默认 4000 token，
                        防止待摘要内容过长再次超出上下文窗口

6. trigger 与 keep 的配合关系（理解摘要节奏的关键）
   - trigger 决定「什么时候压缩」，keep 决定「压缩后留多少原文」
   - 本例 trigger=("tokens", 300)、keep=("messages", 4) 的含义：
       历史一旦超过 300 token，就把除最近 4 条之外的消息全部压缩成摘要
   - 两者差距越大，摘要触发越频繁（压缩掉的内容多，但很快又会超阈值）
   - 建议 keep 对应的 token 量明显小于 trigger，否则会频繁触发摘要浪费调用

7. Checkpointer 与 thread_id 的作用
   - Checkpointer 是 LangGraph 的状态持久化机制，保存 Agent 的完整 state
   - state 中的 messages 就是对话历史，因此「持久化状态」== 「持久化记忆」
   - thread_id 是会话标识，作用等价于第 10 章的 session_id：
       同一个 thread_id 的多次调用共享同一份状态（同一个会话）
       不同 thread_id 的状态完全隔离（不同会话）
   - InMemorySaver 存在内存中，进程退出即丢失；
     生产环境可换 SqliteSaver / PostgresSaver / RedisSaver

8. Checkpointer 与 RunnableWithMessageHistory 的区别
   - RunnableWithMessageHistory（第 10 章）：只持久化「消息历史」，适用于简单链
   - Checkpointer（本章）：持久化 Agent 的「全部状态」，
     包括消息、工具调用中间结果、自定义 state 字段，适用于 Agent 与多步流程
   - Checkpointer 还支持时间旅行（回放到任意历史检查点）与断点续跑

9. 本示例的数据流转全过程
   用户输入 "你好"
     ↓ agent.stream({"messages": [{"role":"user","content":"你好"}]}, config)
     ↓ Checkpointer 按 thread_id 读取已有 state.messages
     ↓ SummarizationMiddleware 检查 token 数是否超过 300
     │    未超过 → 原样放行
     │    已超过 → 调用 llm 把早期消息压缩成摘要，只保留最近 4 条原文
     ↓ 组装 [system_prompt, 摘要消息, ...最近4条..., HumanMessage("你好")]
     ↓ llm 流式生成 AIMessageChunk
     ↓ Checkpointer 把本轮的用户消息与 AI 回复写回 state

10. stream_mode 参数的含义
   - "messages"  逐 token 产出 (消息块, 元数据) 二元组，适合做打字机效果
   - "values"    每步产出完整的 state 快照
   - "updates"   每步只产出 state 的增量变更
   - 本例用 "messages"，因此循环中要解包成 chunk, _ 两个变量

11. 注意事项
   - 摘要会额外消耗一次 LLM 调用，trigger 设置过小会显著增加成本
   - 摘要是有损压缩，关键信息（人名、偏好、结论）可能在多次压缩后丢失
   - 摘要替换是写入 state 的，一旦压缩就不可逆，原文无法恢复
   - InMemorySaver 进程退出即丢失，仅适合 demo 与测试
   - agent.get_state(config) 返回的 messages 是「摘要后」的结果，
     能直接观察到早期消息被替换成摘要消息的现象

12. 对照 0.x ConversationSummaryBufferMemory 的额外坑（历史参考）
   - 0.x 把摘要默认设为 system 角色，消息列表会变成 [system, system, human, ai, ...] 连续多条 system 消息；部分聊天模型（如百度文心）不支持多条 system，且要求历史必须是 1 条 Human + 1 条 AI 的配对格式，直接套用可能报错
   - 0.x 在极端场景（首轮短、次轮长）会执行两次 token 长度计算，若不异步处理对话会变慢
   - 0.x 压缩的源码逻辑（langchain/memory/summary_buffer.py 的 prune）：超过 max_token_limit 时不断 pop 最早的消息，再调用 predict_new_summary 更新 moving_summary_buffer；1.x 的 SummarizationMiddleware 把这套逻辑下沉为 middleware，行为等价但更解耦

===================================================================================
"""
import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建 OpenAI 兼容协议的聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，实现 Runnable 协议
#   双重用途：本例中同一个 llm 既作为对话模型（create_agent 的 model），
#             又作为摘要模型（SummarizationMiddleware 的 model）。
#             生产环境常拆开：对话用强模型保证质量，摘要用小模型降低成本
llm = ChatOpenAI(model="deepseek-v4-pro")

# 2.LangChain 1.x 写法：摘要能力由 SummarizationMiddleware 提供，
# 历史持久化由 LangGraph 的 checkpointer 负责(线程即会话)
#
# create_agent(model, tools, system_prompt, middleware, checkpointer, ...) -> CompiledGraph
#   作用：创建一个基于 LangGraph 的 Agent 执行图
#   参数：
#     model: BaseChatModel | str - 驱动 Agent 决策与回复的主模型
#     tools: list - Agent 可调用的工具列表。
#                   本例传空列表 []，表示不使用工具，Agent 退化为纯聊天机器人
#     system_prompt: str - 系统提示词，设定 Agent 的角色与行为准则。
#                          会被自动放在消息序列最前面，且不参与摘要压缩
#     middleware: Sequence[AgentMiddleware] - 中间件列表，
#                          在 Agent 执行的各个节点前后插入横切逻辑，
#                          按列表顺序依次生效
#     checkpointer: BaseCheckpointSaver - 状态持久化器，
#                          负责按 thread_id 保存与恢复 Agent 的完整状态
#   返回：CompiledGraph 实例（LangGraph 编译后的图），
#         支持 invoke / stream / get_state / update_state 等方法
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="你是DeepSeek开发的聊天机器人，请根据对应的上下文回复用户问题",
    middleware=[
        # SummarizationMiddleware(model, *, trigger, keep, token_counter,
        #                         summary_prompt, trim_tokens_to_summarize) -> 中间件实例
        #   作用：在消息发给模型之前拦截，超过阈值时把早期消息压缩成摘要
        #   继承关系：AgentMiddleware 的子类，是官方内置中间件
        #   等价于旧版：ConversationSummaryBufferMemory
        SummarizationMiddleware(
            # model: BaseChatModel | str
            #   执行摘要任务的模型。摘要质量直接决定长期记忆的可靠性，
            #   信息密度高的场景建议用能力较强的模型
            model=llm,
            # trigger: tuple[str, int|float] | list
            #   触发摘要的条件，("tokens", 300) 表示历史 token 数超过 300 时触发。
            #   其他可选形式：
            #     ("messages", 20)  按消息条数触发
            #     ("fraction", 0.8) 按占用上下文窗口的比例触发（更自适应）
            #   可传列表表示「任一条件满足即触发」
            trigger=("tokens", 300),  # 超过 300 个 token 触发摘要
            # keep: tuple[str, int|float]
            #   摘要后保留多少「原文」，即缓冲区大小，默认 ("messages", 20)。
            #   ("messages", 4) 表示保留最近 4 条消息的完整原文，
            #   更早的消息全部被压缩进摘要。
            #   调参建议：keep 对应的 token 量应明显小于 trigger 阈值，
            #             否则摘要后很快又超阈值，导致频繁触发浪费调用
            keep=("messages", 4),  # 保留最近 4 条消息的原文(缓冲部分)
            # 其他可选参数（本例使用默认值）：
            #   token_counter=count_tokens_approximately
            #     token 计数函数，默认使用快速近似算法，无需调用分词器
            #   summary_prompt=<内置模板>
            #     摘要提示词，内置模板会引导模型输出带
            #     SESSION INTENT / SUMMARY / ARTIFACTS / NEXT STEPS 分节的结构化摘要，
            #     可传自定义字符串覆盖（模板中需包含 {messages} 占位符）
            #   trim_tokens_to_summarize=4000
            #     送入摘要模型的待压缩内容本身的 token 上限，
            #     防止待摘要内容过长导致摘要请求本身超出上下文窗口
        ),
    ],
    # InMemorySaver() -> InMemorySaver
    #   作用：创建基于内存的检查点存储器，按 thread_id 保存 Agent 完整状态
    #   参数：无必填参数（可选 serde 自定义序列化器）
    #   返回：InMemorySaver 实例，实现 BaseCheckpointSaver 接口
    #   持久化内容：不只是消息历史，还包括工具调用中间结果、自定义 state 字段
    #   局限：存在进程内存中，程序退出即丢失。
    #         生产环境请换 SqliteSaver / PostgresSaver / RedisSaver
    checkpointer=InMemorySaver(),
)

# 运行时配置：通过 configurable 通道传递 thread_id
#   config 结构：{"configurable": {"thread_id": "会话标识"}}
#   thread_id 的作用等价于第 10 章的 session_id：
#     相同 thread_id 的多次调用共享同一份状态（延续同一会话）
#     不同 thread_id 的状态完全隔离（互不可见）
#   注意：使用 checkpointer 时必须传 thread_id，否则 LangGraph 会抛异常
config = {"configurable": {"thread_id": "muxiaoke"}}

# 3.死循环构建对话命令行
while True:
    # 读取用户输入，阻塞等待
    query = input("Human: ")

    # 退出条件：输入 q 直接终止进程
    # 注意：InMemorySaver 存在内存中，退出后全部状态（含摘要与历史）丢失
    if query == "q":
        exit(0)

    print("AI: ", flush=True, end="")

    # agent.stream(input: dict, config: dict, stream_mode: str) -> Iterator[tuple]
    #   作用：以流式方式执行 Agent
    #   参数：
    #     input: dict - Agent 的输入状态增量，标准格式为
    #                   {"messages": [{"role": "user", "content": "..."}]}。
    #                   只需传本轮新消息，历史由 checkpointer 自动从状态中恢复
    #     config: dict - 运行时配置，必须包含 configurable.thread_id
    #     stream_mode: str - 流式输出模式：
    #                   "messages" 逐 token 产出 (消息块, 元数据) 二元组
    #                   "values"   每步产出完整 state 快照
    #                   "updates"  每步只产出 state 增量
    #   返回：Iterator[tuple[BaseMessageChunk, dict]]
    #         由于 stream_mode="messages"，每次产出二元组，
    #         因此 for 循环中解包为 chunk, _（元数据用不到，用下划线忽略）
    #
    #   执行期间中间件的介入时机：
    #     读取历史 → SummarizationMiddleware 检查阈值 → 必要时压缩 → 调用模型
    for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="messages",
    ):
        # chunk.text -> str
        #   作用：取出消息块的纯文本内容
        #   说明：相比 chunk.content，text 属性能正确处理多模态内容块（list 形式），
        #         始终返回拼接好的字符串，是更安全的取值方式
        #   flush=True 立即刷新缓冲区实现打字机效果；end="" 取消默认换行
        print(chunk.text, flush=True, end="")
    print("")

    # agent.get_state(config) -> StateSnapshot
    #   作用：读取指定 thread_id 当前的 Agent 状态快照
    #   参数：config - 含 thread_id 的运行时配置
    #   返回：StateSnapshot 对象，主要字段：
    #           values      当前状态字典，values["messages"] 即消息历史
    #           next        下一步将执行的节点
    #           config      本次快照对应的配置（含 checkpoint_id）
    #   观察要点：对话轮次变多后，可以看到早期的 Human/AI 消息被替换成
    #             一条摘要消息，而最近 4 条仍是原文，
    #             这就是「摘要 + 缓冲」混合结构的直观体现
    print("history: ", agent.get_state(config).values["messages"])

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# ConversationSummaryBufferMemory 在超过 max_token_limit 时把早期对话压缩成摘要，
# 近期对话保持原文，摘要与缓冲都由 Memory 组件隐式完成
#
# 对照阅读要点（理解 1.x 的演进动机）：
#   - 旧版把「存储 + 计数 + 摘要 + 注入」全塞进一个 Memory 对象，行为不透明，
#     摘要何时触发、用什么提示词压缩，都难以观察与定制
#   - 新版拆成 Middleware（摘要）+ Checkpointer（存储）+ Agent（编排），
#     每层可独立替换：换摘要模型、换触发策略、换存储后端都互不影响
#   - 旧版必须手动 memory.save_context 写回，忘记调用就没有记忆
#     新版由 checkpointer 自动持久化状态，不会遗漏
#   - 旧版一个 Memory 实例只服务一个会话，多用户需自己维护映射；
#     新版靠 thread_id 天然支持多会话隔离
#   - 旧版 max_token_limit 一个参数同时决定触发与保留，粒度粗；
#     新版 trigger / keep 分离，可精细控制摘要节奏
#
# from operator import itemgetter
#
# from langchain_classic.memory import ConversationSummaryBufferMemory
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#
# # ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
# #   MessagesPlaceholder("history") 用于接收 Memory 返回的消息列表，
# #   在新版中该角色由 Agent 内部的消息状态承担，无需显式占位
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是DeepSeek开发的聊天机器人，请根据对应的上下文回复用户问题"),
#     MessagesPlaceholder("history"),
#     ("human", "{query}"),
# ])
# # ConversationSummaryBufferMemory(max_token_limit, return_messages, input_key, llm)
# #   max_token_limit=300 - 缓冲区 token 上限，超过即把最早的对话压缩进摘要，
# #                         对应新版的 trigger=("tokens", 300)
# #   return_messages=True - 返回消息对象列表（配合 MessagesPlaceholder）
# #   input_key="query"    - save_context 时从 inputs 的哪个键取用户输入
# #   llm=llm              - 用于生成摘要与计算 token 的模型，
# #                          对应新版 SummarizationMiddleware 的 model 参数
# memory = ConversationSummaryBufferMemory(
#     max_token_limit=300,
#     return_messages=True,
#     input_key="query",
#     llm=llm,
# )
#
# # RunnableLambda(memory.load_memory_variables) | itemgetter("history")
# #   load_memory_variables({}) 返回 {"history": [摘要消息, ...近期原文...]}，
# #   因此需再用 itemgetter("history") 从 dict 中取出列表
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
#     #   手动写回本轮问答，内部会判断是否超过 max_token_limit 并触发摘要
#     memory.save_context(chain_input, {"output": output})
#     print("")
#     print("history: ", memory.load_memory_variables({}))

# ===================================================================================
# 最佳实践与其他用法
# ===================================================================================
#
# 1. 对话与摘要使用不同模型（显著降低成本，推荐）：
#    chat_llm = ChatOpenAI(model="deepseek-v4-pro")      # 强模型负责对话
#    summary_llm = ChatOpenAI(model="deepseek-flash")    # 小模型负责摘要
#    agent = create_agent(model=chat_llm, tools=[], middleware=[
#        SummarizationMiddleware(model=summary_llm, trigger=("tokens", 2000),
#                                keep=("messages", 6)),
#    ], checkpointer=InMemorySaver())
#
# 2. 按上下文窗口占比自适应触发（无需硬编码 token 数，更通用）：
#    SummarizationMiddleware(model=llm, trigger=("fraction", 0.7),
#                            keep=("fraction", 0.3))
#
# 3. 多条触发条件（任一满足即压缩）：
#    SummarizationMiddleware(model=llm,
#                            trigger=[("tokens", 4000), ("messages", 40)],
#                            keep=("messages", 8))
#
# 4. 自定义摘要提示词（需保留 {messages} 占位符）：
#    SummarizationMiddleware(
#        model=llm,
#        trigger=("tokens", 300),
#        keep=("messages", 4),
#        summary_prompt="请把下面的对话压缩成摘要，必须保留用户的姓名、"
#                       "偏好、所在城市等关键信息：\n\n{messages}",
#    )
#
# 5. 切换成持久化 checkpointer（跨进程保留记忆）：
#    from langgraph.checkpoint.sqlite import SqliteSaver
#    with SqliteSaver.from_conn_string("checkpoints.db") as saver:
#        agent = create_agent(model=llm, tools=[], checkpointer=saver, ...)
#
# 6. 多会话隔离（同一个 agent 服务多个用户）：
#    agent.invoke({"messages": [{"role": "user", "content": "我叫张三"}]},
#                 config={"configurable": {"thread_id": "user_a"}})
#    agent.invoke({"messages": [{"role": "user", "content": "我是谁"}]},
#                 config={"configurable": {"thread_id": "user_b"}})
#    user_b 不会知道 user_a 叫张三
#
# 7. 非流式调用（一次性拿到完整结果）：
#    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]},
#                          config=config)
#    print(result["messages"][-1].content)   # 取最后一条消息即 AI 回复
#
# 8. 与窗口裁剪中间件组合（先摘要再裁剪，双重保险）：
#    from langchain.agents.middleware import SummarizationMiddleware
#    middleware=[SummarizationMiddleware(...), 其他中间件...]
#    middleware 按列表顺序依次生效，可自由叠加
#
# 9. 常见错误与解决方案：
#    - 错误：ValueError: Checkpointer requires configurable thread_id
#      原因：配置了 checkpointer 但调用时没传 thread_id
#      解决：始终传 config={"configurable": {"thread_id": "..."}}
#
#    - 错误：ValueError: too many values to unpack
#      原因：stream_mode="messages" 产出的是二元组，却只用一个变量接收
#      解决：写成 for chunk, metadata in agent.stream(...)，
#            或改用 stream_mode="values" 配合单变量
#
#    - 问题：摘要一直没有触发
#      原因：trigger 阈值设得过高，历史还没达到该 token 数
#      解决：调低 trigger，或用 agent.get_state(config) 观察实际 token 规模
#
#    - 问题：摘要触发过于频繁，成本飙升
#      原因：keep 保留的内容量接近或超过 trigger 阈值，
#            压缩后很快又超阈值，形成反复压缩
#      解决：让 keep 对应的 token 量明显小于 trigger（如 trigger 的 1/3）
#
#    - 问题：摘要后模型忘记了用户的姓名/偏好
#      原因：默认摘要提示词偏向任务型总结，可能丢弃闲聊型关键信息
#      解决：用 summary_prompt 自定义提示词，显式要求保留人物画像类信息
#
#    - 问题：重启程序后记忆全部丢失
#      原因：InMemorySaver 只存在于进程内存
#      解决：改用 SqliteSaver / PostgresSaver 等持久化实现
#
#    - 注意：摘要是有损且不可逆的。压缩后原文已从 state 中移除，
#      无法恢复。对合规审计场景，应另外把完整对话落库存档
