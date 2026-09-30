#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 23:33
@Author  : thezehui@gmail.com
@File    : 2.文件对话消息历史组件实现记忆.py

===================================================================================
知识点讲解：FileChatMessageHistory 文件持久化记忆
===================================================================================

1. FileChatMessageHistory 的定位
   - 位于 langchain_community.chat_message_histories，是 BaseChatMessageHistory 的文件系统实现
   - 把消息序列化成 JSON 存到本地文件，最大优势是「进程重启后记忆仍在」
   - 定位：单机、单用户、轻量级持久化，是理解持久化机制的最佳入门实现

2. 底层工作原理
   - 构造时接收一个文件路径，若文件不存在会自动创建
   - messages 属性：每次读取都从磁盘加载并反序列化 JSON → List[BaseMessage]
   - add_message()：先读全量、追加、再全量写回（覆盖式写入）
   - 序列化依赖 langchain_core.messages 的 messages_to_dict / messages_from_dict

3. 性能特征与适用边界
   - 每次读写都是全量 IO，消息越多越慢，时间复杂度 O(n)
   - 无文件锁，多进程并发写入会互相覆盖，存在数据丢失风险
   - 适用：本地 CLI 工具、个人助手、demo 演示
   - 不适用：Web 服务、多用户、高并发场景（应换 Redis / 数据库实现）

4. 本示例的「存储 + 处理」组合（课程标准写法）
   - 记忆的存储：FileChatMessageHistory("./memory.txt") 负责把对话落盘
   - 记忆的处理逻辑：在每次循环里构造 system_prompt，用 f-string 把历史注入 <context> 标签：
       system_prompt = "...\n<context>\n{chat_history}\n</context>\n"
   - f"{chat_history}" 会调用对象的 __str__，得到消息的文本表示
   - 这是「原生 OpenAI SDK + LangChain 存储组件」混用的过渡写法；纯 LangChain 方案应使用 MessagesPlaceholder + RunnableWithMessageHistory

5. 记忆写入的时机（非常关键）
   - 必须在流式响应「完全消费完毕」之后才调用 add_user_message / add_ai_message
   - 原因：流式返回是增量的，中途 ai_content 还不完整
   - 课程标准写法在循环结束后统一调用 add_messages([HumanMessage(query), AIMessage(ai_content)])
   - 顺序也有讲究：先 add_user_message 再 add_ai_message，保证消息顺序正确

6. 推理模型的流式处理要点
   - extra_body={"thinking": {"type": "disabled"}} 关闭深度思考，降低延迟与成本
   - 即便关闭思考，仍保留 reasoning_content 的兼容处理，属于防御性编程
   - 判断条件必须用 if delta.content: 而非 content is None: break，否则思考阶段的第一个 chunk 就会中断整个流

7. 与其他持久化实现的切换
   - FileChatMessageHistory("./memory.txt")
   - RedisChatMessageHistory(session_id="u1", url="redis://...", ttl=3600)
   - SQLChatMessageHistory(session_id="u1", connection_string="sqlite:///chat.db")
   - 三者 API 完全一致，切换存储后端只需改一行构造代码

8. 注意事项
   - 文件名虽为 .txt，内容实际是 JSON 格式
   - 历史会无限增长，长期运行需配合 trim_messages 或定期归档
   - 文件中会明文保存用户对话，涉及隐私数据时需加密或脱敏
   - 多用户场景下应按 user_id 拆分文件路径，避免记忆串号

===================================================================================
"""
import os

import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建客户端&记忆
# OpenAI(base_url: str) -> OpenAI
#   作用：创建 OpenAI 兼容协议的客户端
#   参数：base_url - API 服务地址，从环境变量读取，支持指向代理或自建网关
#   返回：OpenAI 客户端实例
#   说明：api_key 由 SDK 自动从环境变量 OPENAI_API_KEY 读取
client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"))

# FileChatMessageHistory(file_path: str) -> FileChatMessageHistory
#   作用：创建基于本地文件的对话消息历史存储
#   参数：file_path - 存储文件路径（相对路径基于当前工作目录）
#                    文件不存在时会自动创建，存在时会自动加载已有历史
#   返回：FileChatMessageHistory 实例，实现 BaseChatMessageHistory 接口
#   存储格式：JSON 数组，每个元素形如 {"type": "human", "data": {"content": "..."}}
#   重要特性：程序重启后历史依然存在，这是与 InMemoryChatMessageHistory 的本质区别
chat_history = FileChatMessageHistory("./memory.txt")

# 2.循环对话
while True:
    # 3.获取用户的输入
    query = input("Human: ")

    # 4.检测用户是否退出对话
    # exit(0) 直接终止进程；由于历史已写入文件，退出不会丢失记忆
    if query == "q":
        exit(0)

    # 5.发起聊天对话
    print("AI: ", flush=True, end="")

    # 构造 system 提示词，把历史记忆通过 <context> 标签注入
    # f"{chat_history}" 隐式调用 chat_history.__str__()，得到历史消息的文本表示
    # 提示词工程要点：用 XML 风格标签包裹上下文，帮助模型明确区分「历史资料」与「当前指令」
    system_prompt = (
        "你是DeepSeek开发的DeepSeek聊天机器人，可以根据相应的上下文回复用户信息，上下文里存放的是人类与你对话的信息列表。\n\n"
        f"<context>{chat_history}</context>\n\n"
    )

    # messages: list[ChatCompletionMessageParam]
    #   作用：构造符合 OpenAI 规范的消息列表
    #   结构：system 消息承载角色设定与历史上下文，user 消息承载本轮提问
    #   类型注解 ChatCompletionMessageParam 来自 openai.types.chat，
    #   提供 IDE 补全与静态类型检查，避免拼错 role/content 字段
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # client.chat.completions.create(...) -> Stream[ChatCompletionChunk]
    #   作用：发起流式对话补全请求
    #   参数：
    #     model: str - 模型名称
    #     extra_body: dict - 透传给服务端的扩展参数（非 OpenAI 标准字段）
    #                        {"thinking": {"type": "disabled"}} 表示关闭深度思考模式，
    #                        可显著降低响应延迟与 token 消耗
    #     messages: list - 消息列表
    #     stream: bool - True 表示流式返回
    #   返回：Stream 对象，可迭代，每次产出一个 ChatCompletionChunk
    response = client.chat.completions.create(
        model='deepseek-v4-pro',
        extra_body={"thinking": {"type": "disabled"}},
        messages=messages,
        stream=True,
    )

    # 累积变量：ai_content 收集完整回答（仅这部分写入记忆）
    ai_content = ""
    # 状态标志：标记当前是否处于思考阶段，用于控制 [思考]/[回答] 前缀只打印一次
    in_reasoning = False

    for chunk in response:
        # 防御性判断：部分网关会下发 choices 为空的心跳包，需跳过避免 IndexError
        if not chunk.choices:
            continue
        # delta 是本次增量数据，含 content / reasoning_content / role 等字段
        delta = chunk.choices[0].delta

        # 6.1 输出思考过程
        # getattr(delta, "reasoning_content", None)
        #   作用：安全读取推理模型特有字段，普通模型无此属性时返回 None 而不报错
        #   返回：str | None - 思维链增量文本
        #   说明：本示例已通过 extra_body 关闭思考，此分支通常不会命中，
        #         保留是为了在开启思考时代码依然可用
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_reasoning:
                print("[思考] ", flush=True, end="")
                in_reasoning = True
            print(reasoning, flush=True, end="")

        # 6.2 输出正式回答，并累积到ai_content中用于记录记忆
        # 关键点：使用 `if delta.content:` 而非 `is None` 判断，
        #        既能跳过 None，也能跳过空串，且不会中断整个流
        if delta.content:
            if in_reasoning:
                print("\n[回答] ", flush=True, end="")
                in_reasoning = False
            # 只累积正式回答，思考内容不入库，避免污染上下文并浪费 token
            ai_content += delta.content
            print(delta.content, flush=True, end="")

    # add_user_message(message: str) -> None
    #   作用：把本轮用户输入写入历史
    #   参数：query - 用户输入文本，自动包装成 HumanMessage
    #   返回：None
    #   副作用：触发一次文件写入（读全量 → 追加 → 写全量）
    chat_history.add_user_message(query)

    # add_ai_message(message: str) -> None
    #   作用：把本轮 AI 完整回答写入历史
    #   参数：ai_content - 流式累积得到的完整回答，自动包装成 AIMessage
    #   返回：None
    #   副作用：再次触发文件写入
    #   时机要求：必须在 for 循环结束（流消费完毕）之后调用，
    #            否则 ai_content 内容不完整，记忆会缺失
    chat_history.add_ai_message(ai_content)
    print("")

# ===================================================================================
# 最佳实践与其他用法
# ===================================================================================
#
# 1. 用批量写入减少文件 IO（一轮问答只写一次盘，性能翻倍）：
#    from langchain_core.messages import HumanMessage, AIMessage
#    chat_history.add_messages([HumanMessage(content=query), AIMessage(content=ai_content)])
#
# 2. 按用户隔离记忆文件，避免多用户记忆串号：
#    chat_history = FileChatMessageHistory(f"./memory/{user_id}.json")
#
# 3. 切换成生产级存储（API 完全一致，仅改构造函数）：
#    from langchain_community.chat_message_histories import RedisChatMessageHistory
#    chat_history = RedisChatMessageHistory(
#        session_id="user_1001", url="redis://localhost:6379", ttl=3600
#    )
#
# 4. 用 get_buffer_string 得到更规范的历史文本（替代 f"{chat_history}"）：
#    from langchain_core.messages import get_buffer_string
#    history_text = get_buffer_string(chat_history.messages)
#
# 5. 控制历史长度，避免超出上下文窗口：
#    from langchain_core.messages import trim_messages
#    trimmed = trim_messages(chat_history.messages, max_tokens=2000,
#                            strategy="last", token_counter=len)
#
# 6. 推荐的纯 LangChain 写法（自动读写历史，无需手动 add_*，见第 10 章）：
#    from langchain_core.runnables.history import RunnableWithMessageHistory
#    chain_with_history = RunnableWithMessageHistory(
#        chain,
#        lambda session_id: FileChatMessageHistory(f"./memory/{session_id}.json"),
#        input_messages_key="query",
#        history_messages_key="history",
#    )
#    chain_with_history.invoke({"query": q}, config={"configurable": {"session_id": "u1"}})
#
# 7. 常见错误与解决方案：
#    - 问题：AI 回复没有被记住，历史里只有用户消息
#      原因：add_ai_message 写在了 for 循环内部，或 ai_content 累积逻辑有误
#      解决：确认写入语句在循环体外，且只在 `if delta.content:` 分支内累积
#
#    - 问题：多开两个终端后记忆互相覆盖
#      原因：FileChatMessageHistory 无文件锁，全量覆盖式写入
#      解决：改用 Redis/数据库实现，或为每个进程使用独立文件
#
#    - 问题：memory.txt 内容看起来是乱码/JSON 而非可读对话
#      说明：这是正常的，文件存储的是消息的 JSON 序列化结果，
#            要看可读文本请用 get_buffer_string(chat_history.messages)
#
#    - 问题：对话轮次多了以后响应变慢并报上下文超限
#      原因：历史无限增长且每次全量注入 Prompt
#      解决：配合 trim_messages 裁剪或摘要压缩（见第 10、11 章）
