#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/8 22:07
@Author  : thezehui@gmail.com
@File    : 3.消息提示模板拼接.py

===================================================================================
知识点讲解：ChatPromptTemplate 消息拼接技巧

1. 拼接的应用场景
   - 在聊天应用中，系统提示词、历史消息、用户输入通常是分离的
   - 通过拼接可以将这些部分模块化管理，提高代码复用性，适合构建多轮对话系统，动态组合不同的提示词片段

2. + 操作符重载与合并对象
   - ChatPromptTemplate 与 PromptTemplate 一样重写了 __add__ 方法，因此所有聊天提示模板都能用 + 拼接
   - 区别在于合并的对象不同：PromptTemplate 的 + 合并的是 template 文本字符串；ChatPromptTemplate 的 + 合并的是 messages 消息模板列表（按顺序 extend）
   - = 是左操作数的方法，左侧必须是 ChatPromptTemplate，右侧可以是 ChatPromptTemplate 或 str

3. 元组简写与显式消息模板类的等价关系
   - 本例写的 ("system", "...") / ("human", "{query}") 是元组简写形式，框架内部会把它们分别转换为 SystemMessagePromptTemplate 与 HumanMessagePromptTemplate，可从打印结果验证
   - 角色 -> 类 全表：
       * system -> SystemMessagePromptTemplate
       * human  -> HumanMessagePromptTemplate
       * ai     -> AIMessagePromptTemplate
       * 不定长历史消息 -> MessagesPlaceholder
   - 也可以显式构造对应的 XxxMessagePromptTemplate，与元组简写完全等价

4. 消息拼接的顺序与角色组织
   - 通常按照 system -> assistant -> human 的顺序组织
   - system 消息定义 AI 的角色和约束，human 消息表示用户输入
   - 可以在中间插入历史消息（使用 MessagesPlaceholder），实现多轮上下文注入

5. 典型输出示例与观察点
   - 打印拼接后的模板对象，可见 messages 列表已按 [system, human] 顺序合并，且变量被合并进 input_variables：
       input_variables=['query', 'username']
       messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(
                     input_variables=['username'], template='你是...我叫{username}')),
                 HumanMessagePromptTemplate(prompt=PromptTemplate(
                     input_variables=['query'], template='{query}'))]
   - 观察点：每个 MessagePromptTemplate 内部都包装了一个 PromptTemplate，即「消息模板 = 角色 + 文本模板」，这解释了为什么聊天模板也支持 f-string 变量
   - 调用 format(...)（或 invoke 后 to_string()）得到带角色前缀的文本：
       System: 你是...开发的聊天机器人，请根据用户的提问进行回复，我叫慕小课
       Human: 你好,你是?
   - 本例直接打印 invoke 的结果，因此看到的是 ChatPromptValue 对象，其中 messages 已是填充完毕的 SystemMessage / HumanMessage 实例

6. 模块化设计的优势与工程实践
   - 将系统提示词和用户消息分离，便于维护和复用，可根据不同场景动态组合，提高代码可读性和可测试性
   - 真实项目里常演化为：通用 system 人设库 + 各业务场景的 human 模板，运行时按需 + 组合，避免为每个场景重复维护一份完整 system 提示词

===================================================================================
"""
from langchain_core.prompts import ChatPromptTemplate

# 定义系统提示词部分（包含 username 变量）
# system 消息用于定义 AI 的角色、背景和行为规则
system_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是DeepSeek开发的聊天机器人，请根据用户的提问进行回复，我叫{username}"),
])

# 定义用户输入部分（包含 query 变量）
# human 消息表示用户的实际提问内容
human_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}")
])

# 使用 + 操作符拼接两个 ChatPromptTemplate
# 拼接后的消息顺序：[system, human]
chat_prompt = system_chat_prompt + human_chat_prompt

# 调用时需要同时提供 username 和 query 两个变量
# 返回 ChatPromptValue 对象，包含完整的消息列表
print(chat_prompt.invoke({
    "username": "慕小课",
    "query": "你好，你是?"
}))
