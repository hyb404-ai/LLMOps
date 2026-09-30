#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 18:05
@Author  : thezehui@gmail.com
@File    : 3.RunnablePassthrough简化invoke调用.py

===================================================================================
知识点讲解：RunnablePassthrough 数据透传与字段追加
===================================================================================

1. 什么是 RunnablePassthrough
   - LCEL 两大核心工具类之一（另一个是 RunnableParallel）
   - 最基础的语义：把输入原封不动地传给下游，invoke(x) 直接返回 x
   - 常与 RunnableParallel 配合，用于「获取数据并保持或新增键」
   - 课程文档定位：透传上游参数输入，可以不变或新增额外的键

2. 简化 invoke 调用：裸字符串包装成字典（基础用法）
   - 直接透传用法：{"query": RunnablePassthrough()} 把裸字符串包装成 {"query": 字符串}
   - 这样调用方可以写 chain.invoke("你好，你是") 而不是 chain.invoke({"query": "..."})
   - 课程文档示例输出：你好！是的，我是ChatGPT...
   - 注意：RunnablePassthrough() 获取的是「整个输入」（字符串或字典）

3. RunnablePassthrough 的两种用法
   - 直接透传：RunnablePassthrough()，常用于把裸输入包装成 dict
   - 追加字段：RunnablePassthrough.assign(**kwargs)，保留输入 dict 全部原有键并追加新键
   - 若想从输入字典中取出某一部分（而非整体），用 itemgetter：
       {"query": itemgetter("query")} | prompt | llm | parser

4. assign() 的工作机制（本示例核心）
   - 签名：RunnablePassthrough.assign(**kwargs: Runnable | Callable) -> RunnableAssign
   - 每个 kwarg 的 key 是新增字段名，value 是计算该字段的 Runnable 或函数
   - value 函数接收「完整的原始输入 dict」，返回该字段的值
   - 输出 = 原始输入 dict ∪ 新增字段（即 {**input, **new_fields}）
   - 新增字段之间并行计算，内部同样基于 RunnableParallel 实现

5. assign 与显式并行字典的对比（对照本章第 2 个示例）
   - 显式字典写法：
       {"context": lambda x: retrieval(x["query"]), "query": itemgetter("query")}
     必须手动列出每一个下游需要的字段，query 要写一次透传，字段多时非常冗长
   - assign 写法：
       RunnablePassthrough.assign(context=lambda x: retrieval(x["query"]))
     只声明「新增什么」，原有字段自动保留，代码更短且不易漏字段
   - 二者对 prompt 而言输出完全一致：{"query": ..., "context": ...}

6. 数据流转全过程
   {"query": "你好，我是谁?"}
        ↓ RunnablePassthrough.assign(context=...)
        ↓   （原有 query 保留，并行计算 context = retrieval(query)）
   {"query": "你好，我是谁?", "context": "我是慕小课"}
        ↓ prompt（渲染模板，填充两个变量）
   [HumanMessage(content="请根据用户的问题回答...")]
        ↓ llm
   AIMessage(content="...")
        ↓ parser
   "最终回答字符串"

7. 典型应用场景
   - RAG：在原输入上追加 context 字段，是 LangChain 官方 RAG 模板的标准写法
   - 记忆注入：追加 history 字段，如 assign(history=lambda x: memory.messages)
   - 多字段派生：assign(context=..., timestamp=..., user_profile=...) 一次追加多个
   - 链路中途保留原始输入：便于后续步骤引用最初的 query 做引用溯源

8. 注意事项
   - assign 的输入必须是 dict，传入字符串会报错（需先用 {"key": RunnablePassthrough()} 包装）
   - 若 assign 的字段名与输入已有字段重名，新值会「覆盖」旧值
   - assign 的 value 函数只接受一个参数（完整输入 dict），多参数请用 partial / bind
   - RunnablePassthrough.assign 返回的是 RunnableAssign 实例，而非 RunnablePassthrough

===================================================================================
"""

import dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()


def retrieval(query: str) -> str:
    """
    一个模拟的检索器函数

    作用：模拟 RAG 中的检索环节，依据用户问题返回参考上下文

    参数：
        query: str - 用户原始提问文本

    返回：
        str - 检索到的上下文文本（示例中固定返回模拟资料）

    副作用：
        打印检索日志，便于观察 assign 分支的实际执行时机
    """
    print("正在检索:", query)
    return "我是慕小课"


# 1.编排prompt
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建含 context 与 query 两个变量的提示词模板
#   参数：template - 模板字符串
#   返回：ChatPromptTemplate 实例，invoke 时需要同时提供 context 与 query
prompt = ChatPromptTemplate.from_template("""请根据用户的问题回答，可以参考对应的上下文进行生成。

<context>
{context}
</context>

用户的提问是: {query}""")

# 2.构建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，invoke 接收 PromptValue，返回 AIMessage
llm = ChatOpenAI(model="deepseek-flash")

# 3.输出解析器
# StrOutputParser() -> StrOutputParser
#   作用：把 AIMessage 转换成纯字符串
#   返回：StrOutputParser 实例，invoke 接收 AIMessage，返回 str
parser = StrOutputParser()

# 4.构建链
# RunnablePassthrough.assign(**kwargs) -> RunnableAssign
#   作用：在保留输入 dict 全部原有字段的基础上，追加计算出的新字段
#   参数：context=lambda x: retrieval(x["query"])
#         - key "context"：要新增的字段名，对应 prompt 中的 {context} 占位符
#         - value lambda：接收完整输入 dict，取出 query 后调用检索函数
#         - lambda 会被自动包装成 RunnableLambda
#   返回：RunnableAssign 实例，类型为 Runnable[dict, dict]
#   输出规则：{**原输入, "context": 计算结果}
#
#   与第 2 个示例的关键差异：
#     第 2 个示例需要写 "query": itemgetter("query") 来手动透传 query，
#     而 assign 会自动保留 query，因此这里无需再写透传分支
#
#   完整链的数据流转：
#     {"query": ...} → RunnableAssign → {"query":..., "context":...}
#                    → prompt → messages → llm → AIMessage → parser → str
chain = RunnablePassthrough.assign(context=lambda x: retrieval(x["query"])) | prompt | llm | parser

# 5.调用链
# chain.invoke(input: dict) -> str
#   参数：input - 只需 {"query": "..."}，context 由 assign 自动补齐
#   执行流程：
#     1. RunnableAssign 收到 {"query": "你好，我是谁?"}
#     2. 调用 retrieval("你好，我是谁?")，打印检索日志并返回 "我是慕小课"
#     3. 合并输出 {"query": "你好，我是谁?", "context": "我是慕小课"}
#     4. prompt 渲染完整提示词，llm 生成回答，parser 提取字符串
#   返回：str - 大模型结合上下文生成的最终回答
content = chain.invoke({"query": "你好，我是谁?"})

print(content)

# ===================================================================================
# 最佳实践与其他调用方式
# ===================================================================================
#
# 1. 一次追加多个字段（各字段并行计算）：
#    chain = RunnablePassthrough.assign(
#        context=lambda x: retrieval(x["query"]),
#        language=lambda x: "中文",
#    ) | prompt | llm | parser
#
# 2. 纯透传用法：把裸字符串输入包装成 dict，从而简化 invoke 调用，
#    调用方可以写 chain.invoke("你好，我是谁?") 而不是 chain.invoke({"query": "..."})
#    chain = {"query": RunnablePassthrough()} | RunnablePassthrough.assign(
#        context=lambda x: retrieval(x["query"])
#    ) | prompt | llm | parser
#    content = chain.invoke("你好，我是谁?")
#
# 3. 接入真实检索器（retriever 本身就是 Runnable，可直接管道组合）：
#    from operator import itemgetter
#    chain = RunnablePassthrough.assign(
#        context=itemgetter("query") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
#    ) | prompt | llm | parser
#
# 4. 批量调用：
#    results = chain.batch([{"query": "我是谁?"}, {"query": "慕小课是什么?"}])
#
# 5. 流式调用：
#    for chunk in chain.stream({"query": "你好，我是谁?"}):
#        print(chunk, end="", flush=True)
#
# 6. 常见错误与解决方案：
#    - 错误：TypeError: 'str' object is not a mapping
#      原因：给 assign 传入了字符串而不是 dict
#      解决：先用 {"query": RunnablePassthrough()} 包装成 dict
#
#    - 错误：assign 追加的字段没有生效 / 被覆盖
#      原因：字段名与输入中已有字段重名，assign 的结果会覆盖原值
#      解决：换一个字段名，或确认覆盖行为符合预期
#
#    - 混淆点：RunnablePassthrough() 与 RunnablePassthrough.assign()
#      前者原样返回输入（不改结构），后者返回「原输入 + 新增字段」的新 dict
