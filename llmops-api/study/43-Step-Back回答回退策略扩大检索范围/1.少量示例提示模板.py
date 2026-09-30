#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 18:15
@Author  : thezehui@gmail.com
@File    : 1.少量示例提示模板.py

===================================================================================
知识点讲解：FewShotChatMessagePromptTemplate 少量示例提示模板
===================================================================================

1. 什么是 Few-Shot Prompting（少样本提示）
   - Zero-Shot：只给指令，不给例子，靠模型自身理解
   - Few-Shot：在 Prompt 中提供若干"输入-输出"示范，让模型通过类比学会任务格式
   - 本质是"上下文学习"（In-Context Learning）：不更新模型参数，
     仅通过 Prompt 中的示例引导模型的输出分布
   - 这是成本最低的"模型定制"手段，效果往往优于冗长的自然语言描述

2. 为什么 Few-Shot 对回答格式的约束特别有效
   - 自然语言指令容易被模型"部分忽略"，而示例是直接可模仿的模式
   - 本例中 3 个示例的答案都是纯数字（"4"、"5"、"300"），
     模型会强烈倾向于也只输出数字，而不是"2+2 等于 4，计算过程是……"
   - 这对需要严格格式输出（JSON、纯数字、固定枚举值）的场景尤为关键

3. 两类少量示例模板的区别
   - FewShotPromptTemplate：面向补全式 LLM，生成的是一段纯文本
   - FewShotChatMessagePromptTemplate（本例）：面向 ChatModel，
     生成的是结构化的消息列表（HumanMessage / AIMessage 交替）
   - 后者更符合对话模型的训练格式，模型遵循度更高，是现代 ChatModel 的首选

4. FewShotChatMessagePromptTemplate 的构造参数
   - example_prompt：单条示例的渲染模板，本例用 human/ai 一问一答的结构
   - examples：示例数据列表，每个元素是 dict，key 必须与 example_prompt 的变量名一致
   - example_selector：示例选择器（与 examples 互斥），
     可根据用户输入动态挑选最相关的示例（如 SemanticSimilarityExampleSelector），
     适用于示例库很大、不能全部塞进 Prompt 的场景

5. 嵌套模板组合的能力（关键特性）
   - ChatPromptTemplate.from_messages() 的列表元素不仅可以是 (role, template) 元组，
     还可以直接嵌入另一个 PromptTemplate 对象
   - 本例的结构是：system 指令 + few_shot_prompt（展开为 6 条消息）+ human 真实问题
   - 渲染时 few_shot_prompt 会被"就地展开"成
     [Human, AI, Human, AI, Human, AI] 共 6 条消息
   - 这种组合能力让 Prompt 可以模块化拼装与复用

6. few_shot_prompt.format() 的调试价值
   - format() 无参数即可调用，因为示例数据已全部固化在 examples 中，
     模板内部没有待填充的外部变量
   - 输出的是渲染后的文本，便于直观检查示例是否按预期展开
   - 这是排查 Few-Shot 不生效问题的第一手段

7. 本例与 Step-Back 策略的关系
   - 这是第 43 节的铺垫示例：Step-Back（回答回退）策略的核心实现，
     就是用 Few-Shot 教会 LLM"如何把具体问题改写为更宽泛的前置问题"
   - 因为"问题回退"是一个抽象的、难以用文字描述清楚的任务，
     用几个示例来示范远比长篇解释有效——这正是 Few-Shot 的最佳应用场景

8. 最佳实践建议
   - 示例数量建议 3~5 个：太少学不到模式，太多浪费 token 且可能引入冲突
   - 示例质量远比数量重要，要覆盖典型场景与边界情况，且格式必须严格统一
   - 示例的答案格式必须与期望输出完全一致（本例全是纯数字，绝不能混入解释文字）
   - 示例库很大时改用 example_selector（如语义相似度选择器）动态挑选，控制 Prompt 长度
   - 示例中不要出现事实错误，模型会照抄错误模式
   - 开发阶段务必用 few_shot_prompt.format() 打印检查渲染结果，确认消息结构正确
   - 注意本例是"格式示范"而非"能力增强"：LLM 的算术能力不会因示例提升，
     复杂计算仍应交给计算器工具（Tool Calling），Few-Shot 只负责规范输出形态

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()

# 1.构建示例模板与示例
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：定义"单条示例"的渲染格式
#   参数：messages - (角色, 模板) 元组列表
#         ("human", "{question}") - 模拟用户提问
#         ("ai", "{answer}")      - 模拟 AI 的标准回答
#   返回：ChatPromptTemplate 实例，输入变量为 question 与 answer
#   关键点：这个模板会对 examples 中的"每一条"数据各渲染一次，
#           从而形成 Human/AI 交替的对话式示范
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])

# 示例数据列表
#   结构要求：每个元素是 dict，其 key 必须与 example_prompt 的变量名完全一致
#             （即必须有 question 和 answer 两个 key，缺失会抛 KeyError）
#   示例设计意图：
#     ① 三个示例覆盖加法与乘法，示范"任意算术问题"这一任务范围
#     ② 答案统一为纯数字，不含任何解释或单位，
#        强约束模型也只输出数字，这是 Few-Shot 最核心的格式引导作用
examples = [
    {"question": "帮我计算下2+2等于多少？", "answer": "4"},
    {"question": "帮我计算下2+3等于多少？", "answer": "5"},
    {"question": "帮我计算下20*15等于多少？", "answer": "300"},
]

# 2.构建少量示例提示模板
# FewShotChatMessagePromptTemplate(
#     example_prompt: BaseChatPromptTemplate,
#     examples: list[dict] = None,
#     example_selector: BaseExampleSelector = None,
#     input_variables: list[str] = None
# ) -> FewShotChatMessagePromptTemplate
#   作用：把示例数据批量渲染为 Human/AI 交替的消息列表
#   参数：example_prompt - 单条示例的渲染模板
#         examples       - 固定的示例数据列表（与 example_selector 二选一）
#   返回：FewShotChatMessagePromptTemplate 实例，可嵌入到其他 ChatPromptTemplate 中
#   渲染结果（3 个示例 → 6 条消息）：
#     [HumanMessage("帮我计算下2+2等于多少？"), AIMessage("4"),
#      HumanMessage("帮我计算下2+3等于多少？"), AIMessage("5"),
#      HumanMessage("帮我计算下20*15等于多少？"), AIMessage("300")]
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# few_shot_prompt.format(**kwargs) -> str
#   作用：渲染模板并返回字符串形式，用于开发调试
#   参数：无需传参，因为示例数据已固化在 examples 中，没有外部待填变量
#   返回：渲染后的文本，可直观看到 Human/AI 交替的 6 条消息
#   调试价值：Few-Shot 不生效时，第一步就该打印这里检查示例是否正确展开
print("少量示例模板:", few_shot_prompt.format())

# 3.构建最终提示模板
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：组装完整的提示模板，体现"模板嵌套组合"的能力
#   参数：messages 列表包含三种不同类型的元素：
#     ("system", "...")   - 系统消息，设定模型角色与任务边界
#     few_shot_prompt     - 直接嵌入的模板对象（非元组），
#                           渲染时会就地展开为 6 条示例消息
#     ("human", "{question}") - 真实用户问题的占位，留待 invoke 时填充
#   返回：ChatPromptTemplate 实例，输入变量为 question
#   最终渲染的消息序列（共 8 条）：
#     [System, Human(例1), AI(例1), Human(例2), AI(例2), Human(例3), AI(例3), Human(真实问题)]
#   顺序设计要点：示例必须放在真实问题之前，
#                 让模型先"看懂模式"再面对真实任务
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个可以计算复杂数学问题的聊天机器人"),
    few_shot_prompt,
    ("human", "{question}"),
])

# 4.创建大语言模型与链
# ChatOpenAI(model: str, temperature: float) -> ChatOpenAI
#   作用：创建聊天模型客户端
#   参数：model       - 模型名称
#         temperature - 0 表示关闭随机性；
#                       计算类任务必须用 0，保证答案确定且可复现
#   返回：ChatOpenAI 实例
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# LCEL 管道组装：prompt | llm | StrOutputParser()
#   数据流转：
#     dict/str → prompt → PromptValue(8 条消息) → llm → AIMessage → parser → str
#   StrOutputParser() 的作用：从 AIMessage 中抽取 content 字段，得到纯字符串
chain = prompt | llm | StrOutputParser()

# 5.调用链获取结果
# chain.invoke(input) -> str
#   作用：执行完整链路
#   参数：input - 这里直接传字符串 "帮我计算下14*15等于多少"
#         说明：模板只有一个输入变量 question 时，LangChain 支持传裸字符串，
#               会自动包装为 {"question": "..."}；
#               变量多于一个时必须显式传 dict
#   返回：模型生成的答案字符串
#   预期输出："210"（纯数字，无任何解释文字）
#   效果对比：
#     若不使用 Few-Shot，模型很可能输出
#       "14 × 15 = 210。计算过程：14 × 15 = 14 × 10 + 14 × 5 = 140 + 70 = 210"
#     加了 Few-Shot 后，模型模仿示例的简洁风格，只输出 "210"
#   这就是少量示例对输出格式的强约束作用
print(chain.invoke("帮我计算下14*15等于多少"))
