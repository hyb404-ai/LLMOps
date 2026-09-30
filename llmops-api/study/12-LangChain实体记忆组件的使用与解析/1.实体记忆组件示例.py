#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/23 15:32
@Author  : thezehui@gmail.com
@File    : 1.实体记忆组件示例.py

===================================================================================
知识点讲解：实体记忆（结构化输出 + 显式实体存储）
===================================================================================

1. 什么是实体记忆（Entity Memory）
   - 实体记忆跟踪对话中提到的实体，并记住关于特定实体的既定事实：它用 LLM 从对话中提取实体信息，再随时间建立对该实体的知识
   - 一般用来存储和查询对话中引用的各类信息，例如人物、地点、事件等
   - 它不保存「完整对话流」，而是从对话中抽取出「实体 → 描述」的知识条目，例如从
     「我是慕小课，正在学习 LangChain」抽取出：
       {"慕小课": "用户本人，正在学习 LangChain", "LangChain": "用户正在学习的框架"}
   - 本质：把非结构化的对话，转化为结构化的「用户画像 / 知识库」

2. 实体记忆与其他记忆策略的本质区别
   策略                 存储形式            token 特征        信息组织
   缓冲记忆             消息列表原文        随轮次线性增长    时间序
   窗口记忆（第10章）   最近 N 条原文        恒定              时间序（丢弃旧的）
   摘要记忆（第11章）   摘要文本 + 近期原文  恒定              时间序（压缩旧的）
   实体记忆（本章）     实体字典            随实体数增长      按实体聚合
   - 前三者都是「按时间」组织，实体记忆是「按主题/对象」组织
   - 实体记忆能长期稳定记住「用户是谁、喜欢什么」，不会因对话变长而遗忘
   - 缺点：丢失对话的时序与上下文细节，无法回答「刚才我说了什么」

3. LangChain 实现范式的演进（1.x 显式 vs 0.x 隐式）
   - 0.x 封装了 ConversationEntityMemory，用内置固定提示词隐式抽取实体，
     但预设 Prompt 过于笨重、极度消耗 Token、对大模型要求极高，实用度并不高
   - 1.x 改为显式三步：
     a. 用 Pydantic 定义实体的数据结构（Entities）
     b. 用 llm.with_structured_output(Entities) 做类型安全的抽取
     c. 实体存储由应用层自己持有（本例是一个普通 dict）
   - 好处：结构可控、抽取可测、存储可换（dict → Redis → 图数据库）

4. with_structured_output 的工作原理（关键知识点）
   - 签名：llm.with_structured_output(schema) -> Runnable[..., schema 实例]
   - 底层机制：把 Pydantic 模型转换成 JSON Schema，
               通过模型的 function calling / tool calling 能力约束输出格式
   - 返回：一个新的 Runnable，其 invoke 直接返回 Pydantic 对象实例，
           而非 AIMessage，因此后面不需要接 OutputParser
   - 优势：输出结构由模型侧强制保证，比「提示词要求返回 JSON + 手动解析」可靠得多
   - 注意：要求模型支持 function calling，不支持的模型需退化为 JSON 模式

5. Pydantic 模型在结构化输出中的三个作用
   - 类型约束：dict[str, str] 明确要求「字符串到字符串的映射」
   - 语义提示：类的 docstring 与 Field(description=...) 会被写入 JSON Schema，
               直接影响模型的抽取行为，相当于「字段级提示词」
   - 运行时校验：模型返回的数据会被 Pydantic 校验，不符合类型会抛异常

6. temperature=0 的必要性
   - 实体抽取是「确定性任务」，同样的输入应得到同样的输出
   - temperature=0 关闭采样随机性，让结果尽可能稳定可复现
   - 对比：闲聊、创作类任务需要 temperature > 0 保证多样性
   - 本例中 temperature=0 同时作用于抽取与对话两条链（共用同一个 llm 实例）

7. 本示例的双链架构（重要设计模式）
   - 一轮对话需要两次 LLM 调用，分别由两条链承担：
     * entity_extractor：抽取链，负责「从输入中提取结构化实体」
         extract_prompt | llm.with_structured_output(Entities) → 输出 Entities 对象
     * chat_chain：对话链，负责「结合实体与历史生成回复」
         chat_prompt | llm | StrOutputParser() → 输出 str
   - 两条链共用同一个 llm 实例，但输出类型完全不同

8. dict.update 实现的实体累积与覆盖语义
   - entity_store.update(new_entities) 的行为：
       新实体名 → 新增条目
       已存在的实体名 → 用新描述「覆盖」旧描述
   - 覆盖是双刃剑：
       优点：实体信息可以随对话演进而更新（如用户换了城市）
       缺点：旧描述被直接丢弃，可能丢失历史信息
   - 若需累积而非覆盖，应改为手动合并描述文本

9. 典型抽取结果与多轮累积（来自课程示例）
   - 经过「我是慕小课…」「最喜欢的编程语言是 Python」「我住在广州」三轮对话后，
     entity_store 会累积出四条实体：
       {'慕小课': '慕小课最近正在学习LangChain。',
        'LangChain': 'LangChain 是一个专注于构建和连接语言模型的项目。',
        'Python': 'Python 是一门非常受欢迎且功能强大的编程语言…',
        '广州': '广州是中国的第三大城市…'}
   - 这正是实体记忆「按主题聚合、长期不遗忘」特征的直观体现

10. 实体记忆 + 消息历史的混合使用
   - 本示例同时维护了两份记忆：entity_store（实体）与 chat_history（对话流）
   - 这是生产环境的常见做法：
       entity_store 提供「长期稳定的用户画像」，永不遗忘
       chat_history 提供「近期对话的上下文」，可配合 trim_messages 裁剪
   - 两者互补：前者答「我是谁」，后者答「刚才聊了什么」

11. 注意事项
   - 每轮对话要调用两次 LLM（抽取 + 回复），延迟与成本约为普通对话的两倍
   - 抽取链可以异步化或降级到小模型，以降低延迟与成本
   - `entity_store or "无"` 利用了空 dict 的 falsy 特性，避免在提示词里渲染出 "{}"
   - entity_store 是普通 dict，进程退出即丢失，生产环境需持久化
   - 抽取质量高度依赖 system_prompt 与 Field description，需要针对业务调优

===================================================================================
"""
import dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# ChatOpenAI(model: str, temperature: float) -> ChatOpenAI
#   作用：创建 OpenAI 兼容协议的聊天模型实例
#   参数：
#     model - 模型名称
#     temperature: float - 采样温度，取值 0 ~ 2。
#                  0 表示贪心解码（几乎确定性输出），值越大随机性越强。
#                  这里设为 0 是因为「实体抽取」属于确定性任务，
#                  需要同样的输入稳定产出同样的实体，便于测试与复现
#   返回：ChatOpenAI 实例，实现 Runnable 协议
#   复用说明：同一个实例同时驱动「抽取链」与「对话链」，
#             生产环境可拆分为抽取用小模型、对话用强模型
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)


# LangChain 1.x 写法：实体抽取用结构化输出显式完成，实体存储由应用自己掌控
class Entities(BaseModel):
    """从用户输入中抽取出的实体及其描述"""
    # class Entities(BaseModel)
    #   作用：用 Pydantic 模型定义「实体抽取结果」的数据结构
    #   继承：BaseModel 是 Pydantic 的基类，提供校验、序列化与 JSON Schema 生成
    #   关键点：类的 docstring 会被写入生成的 JSON Schema 的 description 字段，
    #           直接参与引导模型理解「要产出什么」，因此 docstring 不是纯注释，
    #           而是提示词的一部分，务必写清楚
    #
    # entities: dict[str, str]
    #   类型约束：要求模型输出「字符串 → 字符串」的映射，
    #             键是实体名称（如 "慕小课"），值是实体描述（如 "用户本人"）
    #
    # Field(description: str) -> FieldInfo
    #   作用：为字段补充元信息，description 会被写入 JSON Schema
    #   参数：description - 字段的语义说明，相当于「字段级提示词」，
    #                       模型会据此决定抽取粒度与输出格式
    #   本例要点："没有实体时返回空字典" 显式告诉模型如何处理无实体的情况，
    #             避免模型编造实体或返回 null 导致校验失败
    #   其他常用参数：default（默认值）、alias（别名）、
    #                 ge/le（数值范围）、min_length/max_length（长度约束）
    entities: dict[str, str] = Field(
        description="实体名称到实体描述的映射，没有实体时返回空字典",
    )


# 抽取链的提示模板
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：用消息列表构建聊天提示模板
#   参数：messages - ("角色", "模板字符串") 元组列表
#     ("system", ...) 设定抽取任务的角色与抽取范围（人物、地点、技术）
#     ("human", "{input}") 承载待抽取的用户原始输入
#   返回：ChatPromptTemplate 实例
#   设计要点：抽取链只看「本轮输入」，不带历史消息。
#             这样每轮只抽取增量实体，避免对已抽取过的内容重复消耗 token
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是实体抽取助手，请从用户输入中抽取人物、地点、技术等实体，并给出简短描述。"),
    ("human", "{input}"),
])

# 构建抽取链
# llm.with_structured_output(schema) -> Runnable
#   作用：把普通聊天模型包装成「输出符合指定结构」的 Runnable
#   参数：schema - Pydantic 模型类（也支持 TypedDict 或 JSON Schema dict）
#   返回：Runnable 实例，其 invoke 直接返回 Entities 对象，而非 AIMessage
#   底层机制：
#     1. 把 Entities 转换成 JSON Schema
#     2. 以 tool/function 的形式注册给模型，强制模型按该结构输出
#     3. 解析模型返回的结构化数据，用 Pydantic 校验并实例化
#   关键优势：结构正确性由模型侧保证，无需再接 OutputParser 手动解析 JSON，
#             也不会出现「模型返回了一段解释文字导致解析失败」的问题
#   前置要求：模型必须支持 function calling / tool calling
#
#   数据流转：{"input": str} → extract_prompt → messages
#             → llm(结构化约束) → Entities 对象
entity_extractor = extract_prompt | llm.with_structured_output(Entities)

# 对话链的提示模板
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   参数：messages - 三个部分组成：
#     ("system", ...{entities}) 角色设定 + 已知实体信息注入。
#                 {entities} 是变量占位符，渲染时会被 entity_store 的字符串表示替换，
#                 这是「让模型看到长期记忆」的方式
#     MessagesPlaceholder("history") 消息列表插槽，接收 List[BaseMessage]，
#                 渲染时把历史消息逐条展开，提供近期对话上下文
#     ("human", "{input}") 本轮用户输入
#   返回：ChatPromptTemplate 实例
#   设计要点：实体（长期画像）走 system 消息，历史（近期上下文）走 placeholder，
#             两种记忆各自注入不同位置，职责清晰
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是DeepSeek开发的聊天机器人，请结合已知实体信息回复用户。\n\n已知实体信息：\n{entities}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# 构建对话链
# 组成：chat_prompt | llm | StrOutputParser()
#   StrOutputParser() -> StrOutputParser
#     作用：从 AIMessage 中提取纯文本内容，等价于取 message.content
#     返回：StrOutputParser 实例，invoke 接收 AIMessage，返回 str
#   与抽取链的对比：
#     抽取链末端是 with_structured_output，输出 Pydantic 对象
#     对话链末端是 StrOutputParser，输出纯字符串
#
# 数据流转：{"input":..., "entities":..., "history":[...]}
#           → chat_prompt → messages → llm → AIMessage → StrOutputParser → str
chat_chain = chat_prompt | llm | StrOutputParser()

# 实体存储 + 消息历史，均由应用层显式持有
#
# entity_store: dict[str, str]
#   作用：实体记忆的存储容器，键是实体名，值是实体描述
#   特点：一个普通 Python dict，没有任何框架封装，完全由应用层掌控。
#         这正是 1.x 的设计理念——存储结构透明可控，
#         想换成 Redis / MySQL / 图数据库只需替换读写逻辑
#   局限：进程内存储，程序退出即丢失
entity_store: dict[str, str] = {}

# InMemoryChatMessageHistory() -> InMemoryChatMessageHistory
#   作用：创建基于内存列表的对话消息历史存储
#   参数：无必填参数
#   返回：InMemoryChatMessageHistory 实例，实现 BaseChatMessageHistory 接口
#   与 entity_store 的分工：
#     entity_store  存「结构化的实体知识」，按主题聚合，回答「用户是谁」
#     chat_history  存「原始对话消息流」，按时间排列，回答「刚才聊了什么」
#   两份记忆互补并存，是生产环境的常见做法
chat_history = InMemoryChatMessageHistory()


def chat(user_input: str) -> str:
    """一轮对话：先抽取实体写入存储，再带着实体信息与历史生成回复

    chat(user_input: str) -> str
      作用：完成一轮完整的「实体记忆对话」

      参数：
        user_input: str - 用户本轮的输入文本

      返回：
        str - AI 生成的回复文本

      副作用（三处状态变更）：
        1. entity_store 被更新（新增或覆盖实体条目）
        2. chat_history 追加一条 HumanMessage
        3. chat_history 追加一条 AIMessage

      LLM 调用次数：2 次
        第 1 次：抽取实体（entity_extractor）
        第 2 次：生成回复（chat_chain）
        因此延迟与成本约为普通单链对话的两倍

      执行顺序要点：
        必须「先抽取、再回复」，这样本轮刚提到的实体就能立即被用于本轮回复，
        实现「我是慕小课」→ 下一句就能称呼用户为慕小课的效果
    """
    # 步骤 1：抽取本轮输入中的实体，并合并进实体存储
    #
    # entity_extractor.invoke(input: dict) -> Entities
    #   参数：{"input": user_input} - 对应 extract_prompt 中的 {input} 占位符
    #   返回：Entities 对象，通过 .entities 属性取出 dict[str, str]
    #
    # dict.update(other: dict) -> None
    #   作用：把新抽取的实体合并进已有存储
    #   合并语义：
    #     新实体名 → 新增条目
    #     已存在的实体名 → 用新描述「覆盖」旧描述（旧描述被丢弃）
    #   覆盖的取舍：便于实体信息随对话演进而更新（如用户搬家换城市），
    #               但也可能丢失历史描述。若需累积应手动拼接描述文本
    entity_store.update(entity_extractor.invoke({"input": user_input}).entities)

    # 步骤 2：带着实体信息与对话历史生成回复
    #
    # chat_chain.invoke(input: dict) -> str
    #   参数：必须提供 chat_prompt 需要的全部三个变量：
    #     "input"    - 本轮用户输入，对应 ("human", "{input}")
    #     "entities" - 实体信息，对应 system 消息中的 {entities}。
    #                  `entity_store or "无"` 是短路技巧：
    #                  空 dict 在布尔上下文中为 False，此时取 "无"，
    #                  避免在提示词里渲染出无意义的 "{}" 干扰模型
    #     "history"  - 历史消息列表，对应 MessagesPlaceholder("history")。
    #                  传入的是 List[BaseMessage] 消息对象列表，不是字符串
    #   返回：str - 经 StrOutputParser 提取后的纯文本回复
    #
    #   注意：此处传入的 history 是「本轮输入写入之前」的历史，
    #         本轮输入由 ("human", "{input}") 单独承载，因此不会重复
    content = chat_chain.invoke({
        "input": user_input,
        "entities": entity_store or "无",
        "history": chat_history.messages,
    })

    # 步骤 3：把本轮问答写回消息历史
    #
    # add_user_message(message: str) -> None
    #   作用：追加一条人类消息，str 会被自动包装成 HumanMessage
    #   返回：None
    chat_history.add_user_message(user_input)

    # add_ai_message(message: str) -> None
    #   作用：追加一条 AI 消息，str 会被自动包装成 AIMessage
    #   返回：None
    #   顺序要求：必须在 add_user_message 之后，
    #             保证历史中 human/ai 交替的正确时序
    chat_history.add_ai_message(content)
    return content


# 连续三轮对话，逐步积累实体
# 第 1 轮：抽取出人物「慕小课」与技术「LangChain」
print(chat("你好，我是慕小课。我最近正在学习LangChain。"))
# 第 2 轮：新增技术实体「Python」；
#          由于实体中已有「慕小课」，模型能在回复中直接称呼用户
print(chat("我最喜欢的编程语言是 Python。"))
# 第 3 轮：新增地点实体「广州」
print(chat("我住在广州"))

# 查询已经记住的实体
# 输出示例：{"慕小课": "用户本人，正在学习 LangChain",
#            "LangChain": "一个 LLM 应用开发框架",
#            "Python": "用户最喜欢的编程语言",
#            "广州": "用户居住的城市"}
# 观察要点：实体是「按对象聚合」的知识条目，而非按时间排列的对话流，
#           这正是实体记忆与其他记忆策略的本质差异
print(entity_store)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# ConversationEntityMemory 内部用固定提示词抽取实体并存到 entity_store，
# 配合 ConversationChain 与内置的 ENTITY_MEMORY_CONVERSATION_TEMPLATE 使用
#
# 对照阅读要点（理解 1.x 的演进动机）：
#   - 旧版抽取提示词内置且不可见，抽取什么、抽取粒度如何，都无法定制
#     新版用 Pydantic + 自定义 prompt 显式声明，结构与语义完全可控
#   - 旧版抽取结果无类型约束，全靠正则/字符串解析，脆弱且易出错
#     新版靠 with_structured_output + Pydantic 校验，类型安全
#   - 旧版实体存储藏在 memory.entity_store.store 内部，替换后端需继承改写
#     新版存储就是应用层的普通 dict，换 Redis/图数据库只改读写逻辑
#   - 旧版 ConversationChain 与 ENTITY_MEMORY_CONVERSATION_TEMPLATE 强绑定，
#     提示词结构无法调整；新版 prompt 完全开放
#   - 旧版实体抽取与对话生成的调用被封装在一次 chain.invoke 中，不透明；
#     新版两条链显式分离，可分别做监控、降级、换模型
#
# from langchain_classic.chains.conversation.base import ConversationChain
# from langchain_classic.memory import ConversationEntityMemory
# from langchain_classic.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
# from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
#
# llm = QianfanChatEndpoint()
#
# # ConversationChain(llm, prompt, memory) -> ConversationChain
# #   llm    - 对话模型
# #   prompt - 必须使用内置的 ENTITY_MEMORY_CONVERSATION_TEMPLATE，
# #            该模板预留了 entities 与 history 变量，结构固定不可调
# #   memory - ConversationEntityMemory 实例，内部同时负责
# #            抽取实体、存储实体、加载实体、存储对话历史
# chain = ConversationChain(
#     llm=llm,
#     prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
#     memory=ConversationEntityMemory(llm=llm),
# )
#
# # chain.invoke(input: dict) -> dict
# #   内部隐式完成：抽取实体 → 加载实体与历史 → 生成回复 → 写回实体与历史
# #   返回：dict，回复文本在 ["response"] 键中
# print(chain.invoke({"input": "你好，我是慕小课。我最近正在学习LangChain。"}))
# print(chain.invoke({"input": "我最喜欢的编程语言是 Python。"}))
# print(chain.invoke({"input": "我住在广州"}))
#
# # 查询实体中的对话
# # memory.entity_store.store -> dict[str, str]
# #   实体存储藏在 Memory 内部两层属性下，路径不直观且难以替换后端
# res = chain.memory.entity_store.store
# print(res)

# ===================================================================================
# 最佳实践与其他用法
# ===================================================================================
#
# 1. 用更丰富的实体结构替代 dict[str, str]（支持类型与置信度）：
#    class Entity(BaseModel):
#        name: str = Field(description="实体名称")
#        type: str = Field(description="实体类型：person/location/technology/other")
#        description: str = Field(description="实体的简短描述")
#
#    class Entities(BaseModel):
#        """从用户输入中抽取出的实体列表"""
#        entities: list[Entity] = Field(description="抽取到的实体列表，没有则返回空列表")
#
# 2. 抽取链降级到小模型（抽取是简单任务，可显著省钱降延迟）：
#    extract_llm = ChatOpenAI(model="deepseek-flash", temperature=0)
#    entity_extractor = extract_prompt | extract_llm.with_structured_output(Entities)
#
# 3. 实体描述累积而非覆盖（保留演进历史）：
#    new_entities = entity_extractor.invoke({"input": user_input}).entities
#    for name, desc in new_entities.items():
#        if name in entity_store and desc not in entity_store[name]:
#            entity_store[name] = f"{entity_store[name]}；{desc}"
#        else:
#            entity_store[name] = desc
#
# 4. 实体存储持久化（跨进程保留用户画像）：
#    import json, redis
#    r = redis.Redis()
#    def load_entities(user_id: str) -> dict[str, str]:
#        raw = r.get(f"entities:{user_id}")
#        return json.loads(raw) if raw else {}
#    def save_entities(user_id: str, store: dict[str, str]) -> None:
#        r.set(f"entities:{user_id}", json.dumps(store, ensure_ascii=False))
#
# 5. 结合窗口裁剪控制历史长度（实体不裁剪，历史裁剪）：
#    from langchain_core.messages import trim_messages
#    trimmed = trim_messages(chat_history.messages, max_tokens=500,
#                            strategy="last", token_counter=llm, start_on="human")
#    content = chat_chain.invoke({"input": user_input,
#                                 "entities": entity_store or "无",
#                                 "history": trimmed})
#    优势：实体提供永不遗忘的长期画像，历史只保留近期上下文，token 可控
#
# 6. 并发执行抽取与回复（用 RunnableParallel 把两次调用并行化，降低延迟）：
#    注意：并行后本轮抽取的实体无法用于本轮回复，只能用于下一轮，
#          需要权衡「延迟」与「实体即时生效」
#
# 7. 只在必要时抽取（降低成本）：
#    先用轻量规则（如输入长度、是否包含「我是/我叫/我住」等模式）判断，
#    只对可能含实体的输入调用抽取链，避免每轮都额外调一次模型
#
# 8. 常见错误与解决方案：
#    - 错误：NotImplementedError / 模型不支持 with_structured_output
#      原因：模型不支持 function calling / tool calling
#      解决：改用 JSON 模式 + PydanticOutputParser，
#            或换成支持函数调用的模型
#
#    - 错误：ValidationError: entities field required
#      原因：模型返回的结构缺少必填字段
#      解决：在 Field description 中更明确地说明输出要求，
#            或给字段加默认值 Field(default_factory=dict, description=...)
#
#    - 错误：Input to ChatPromptTemplate is missing variables {'entities'}
#      原因：chat_chain.invoke 时漏传了 entities 或 history
#      解决：三个变量 input / entities / history 必须全部提供
#
#    - 问题：抽取出大量无意义实体（如「你好」「最近」）
#      原因：system_prompt 未限定抽取范围与粒度
#      解决：在提示词中明确实体类型白名单，并要求「不确定时不要抽取」
#
#    - 问题：实体描述被反复覆盖，信息越来越简略
#      原因：dict.update 是覆盖语义
#      解决：改用上面第 3 条的累积合并逻辑
#
#    - 问题：模型没有使用已知实体信息回复
#      原因：entity_store 渲染成 Python dict 的 repr 形式，可读性差
#      解决：格式化成自然语言再注入，例如
#            "\n".join(f"- {k}: {v}" for k, v in entity_store.items()) or "无"
#
#    - 问题：每轮延迟明显偏高
#      原因：一轮对话需要两次串行的 LLM 调用
#      解决：抽取链换小模型、或用异步并发、或按需触发抽取
