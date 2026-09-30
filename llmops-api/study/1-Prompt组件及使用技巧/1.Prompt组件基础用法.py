#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/8 21:36
@Author  : thezehui@gmail.com
@File    : 1.Prompt组件基础用法.py

===================================================================================
知识点讲解：Prompt 组件基础与核心 API

1. 设计动机：为什么需要 Prompt 组件
   - 大多数 LLM 应用都不会把用户输入直接丢给 LLM，而是把用户输入嵌入到一段更大的文本片段中，即「提示模板」，由模板提供特定任务所需的附加上下文
   - 一个最基础的聊天机器人流程即：用户输入 -> Prompt 组装 -> LLM 推理 -> 输出解析 -> 返回结果，Prompt 是所有 AI 应用交互的起点
   - 朴素写法（手工 f-string 拼接）的缺陷：提示词与某个具体模型的输入格式强绑定，切换模型（文本补全模型 <-> 聊天模型）时要重写拼接逻辑
   - 所以 LangChain 封装了 Prompt 组件：它是「高可移植性」的，同一个 Prompt 可以支持各种 LLM，切换 LLM 时无需修改 Prompt

2. 组件定位与全景：两大类
   - Prompt Template：将 Prompt 按照 template 做格式化，负责变量处理与提示词组合，实战中的主用组件
   - Selectors（示例选择器）：根据不同条件去选择不同提示词，或在 few-shot 场景选择不同示例来进一步提高 Prompt 支持能力
   - 二者关系：本质上 Selectors 只是 Prompt Template 的二次封装；由于 Selectors 使用范围较窄、应用场景较小，实战中重点使用 Prompt Template

3. Prompt Template 子组件职责表
   - PromptTemplate               : 创建「文本」消息提示模板，用于与大语言模型 / 文本生成模型（补全类模型）交互
   - ChatPromptTemplate           : 创建「聊天」消息提示模板，一般用于与聊天模型交互
   - MessagesPlaceholder          : 消息占位符，在聊天模型中对「不确定是否需要」的消息进行占位（典型用途：chat_history）
   - SystemMessagePromptTemplate  : 创建系统消息提示模板，角色为 system
   - HumanMessagePromptTemplate   : 创建人类消息提示模板，角色为 human
   - AIMessagePromptTemplate      : 创建 AI 消息提示模板，角色为 ai
   - PipelinePromptTemplate       : 创建管道消息，可把提示模板当作变量快速复用（注意：该类属于 0.x 旧版 API，LangChain 1.x 已移除，替代方案见 4.复用提示模板.py）
   - 说明：ChatPromptTemplate.from_messages 中写 ("system", "...") 这种元组，LangChain 内部会自动把它转换成 SystemMessagePromptTemplate，因此「元组简写」与「显式使用 XxxMessagePromptTemplate」是等价的两种写法

4. PromptTemplate 基础用法
   - 用于构造单一字符串格式的提示词，支持 {variable} 变量占位符
   - 提供 format() 与 invoke() 生成最终提示词：format 直接返回 str，invoke 返回 PromptValue
   - 本例还演示了 + 拼接 str 后变量自动合并（详见 2.字符串提示拼接.py）

5. ChatPromptTemplate 基础用法
   - 用于构造多轮对话格式的提示词（适配 ChatModel），支持 system、human、ai 等不同角色消息
   - MessagesPlaceholder 用于插入动态数量的历史消息（本例 chat_history 即此用法）
   - HumanMessagePromptTemplate 用于构造带变量的用户消息
   - 元组简写 ("system", "...") / ("human", "...") 会被自动转换为对应角色的消息模板

6. partial() 方法：提前绑定部分变量
   - 用于「预格式化」提示模板中的部分变量，返回一个新的模板对象，剩余变量留到调用时再传
   - 典型用途：提前绑定 now（当前时间）、固定的配置参数等，本例用它提前绑定 now
   - partial 绑定后，该变量从 input_variables 中移出，调用时不必再传

7. PromptValue 与四个方法的差异
   - partial     : 格式化提示模板中的「部分」变量，返回新模板对象，剩余变量留待调用时再传
   - format      : 传递变量数据，格式化提示模板为「文本消息」（直接返回 str）
   - invoke      : 传递变量数据，格式化提示模板为「提示」（返回 PromptValue 对象）
   - to_string   : 将提示 / 消息提示列表转换成「字符串」
   - to_messages : 将提示转换成「消息列表」
   - 记忆要点：format 的产物是 str，invoke 的产物是 PromptValue；PromptValue 是中间态，再通过 to_string / to_messages 落到具体形态
   - PromptValue 这层抽象正是「同一个 Prompt 适配不同模型」的关键：交给文本模型时走 to_string，交给聊天模型时走 to_messages

8. + 运算符重载：模板组装拼接
   - Prompt 组件对 + 运算符使用 __add__ 魔术方法进行了重写，所以几乎所有 Prompt 组件都可以用 + 进行组装拼接
   - = 是左操作数的方法，左侧必须是 Prompt 组件对象，右侧可以是 Prompt 组件或普通 str（框架内部做类型归一）
   - 对比 Runnable 组件重写的是 __or__ / __ror__ 来支撑 | 管道，两者是不同层面的运算符重载，不要混淆（| 的用法见 4-LCEL表达式与Runnable可运行协议）
   - 具体拼接示例见 2.字符串提示拼接.py 与 3.消息提示模板拼接.py

9. 模板格式化引擎：template_format 的可选值
   - 默认使用 f-string 方式格式化变量：用 {} 花括号包裹变量或表达式，语法简洁、可执行简单运算、性能较好，但只限用在 Python 中（模板不便于跨语言 / 跨平台复用）
   - 也可显式指定 template_format="jinja2"：除变量替换外，还支持循环 / 条件等控制结构以及自定义过滤器和宏，可用性更广；代价是需要额外安装 jinja2 库，且变量语法为双花括号 {{subject}}
   - 写法对比：
       PromptTemplate.from_template("请讲一个关于{subject}的笑话")
       PromptTemplate.from_template("请讲一个关于{{subject}}的笑话", template_format="jinja2")
   - 取舍建议：模板里只做变量替换就用默认 f-string；需要循环 / 条件等逻辑（如动态渲染检索到的多条文档）再考虑 jinja2

10. 变量数量约定
   - 在模板中定义了多少个变量，调用时就需要传递多少个变量对应的值
   - 已用 partial 预绑定的变量不必再传（如本例的 now）
   - 拼接 / 合并多个子模板时，输入变量会自动合并到同一个 input_variables 集合，调用时需一次性补齐所有变量

===================================================================================
"""
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# 使用 from_template 快速创建 PromptTemplate 实例
prompt = PromptTemplate.from_template("请讲一个关于{subject}的冷笑话")

# invoke() 方法接收字典参数，返回 PromptValue 对象
prompt_value = prompt.invoke({"subject": "程序员"})

# format() 方法直接返回格式化后的字符串
print(prompt.format(subject="喜剧演员"))

# to_string() 将 PromptValue 转换为字符串
print(prompt_value.to_string())

# to_messages() 将 PromptValue 转换为消息列表（单个 HumanMessage）
print(prompt_value.to_messages())

print("==================")

# 创建 ChatPromptTemplate，适用于多轮对话场景
chat_prompt = ChatPromptTemplate.from_messages([
    # system 消息：定义 AI 的角色和行为规则
    ("system", "你是DeepSeek开发的聊天机器人，请根据用户的提问进行回复，当前的时间为:{now}"),
    
    # MessagesPlaceholder：占位符，用于插入动态数量的历史消息
    # 在 invoke 时需要传入 chat_history 参数（列表类型）
    MessagesPlaceholder("chat_history"),
    
    # HumanMessagePromptTemplate：构造带变量的用户消息模板
    HumanMessagePromptTemplate.from_template("请讲一个关于{subject}的冷笑话"),
    
# partial() 提前绑定 now 参数为当前时间，后续调用时不需要再传入
]).partial(now=datetime.now())

# 调用时传入 chat_history 和 subject 参数
chat_prompt_value = chat_prompt.invoke({
    # chat_history 是历史消息列表，可以是元组或 Message 对象
    "chat_history": [
        ("human", "我叫慕小课"),
        AIMessage("你好，我是DeepSeek，有什么可以帮到您"),
    ],
    "subject": "程序员",
})

# 输出 ChatPromptValue 对象（包含完整的消息列表）
print(chat_prompt_value)

# 转换为字符串格式，展示所有消息内容
print(chat_prompt_value.to_string())
