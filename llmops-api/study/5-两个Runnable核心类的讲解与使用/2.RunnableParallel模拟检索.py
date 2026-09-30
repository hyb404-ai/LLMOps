#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 17:44
@Author  : thezehui@gmail.com
@File    : 2.RunnableParallel模拟检索.py

===================================================================================
知识点讲解：用 RunnableParallel 构建 RAG 输入（隐式字典 + itemgetter）
===================================================================================

1. 本示例解决的核心问题
   - RAG（检索增强生成）的 Prompt 通常需要两个变量：context（检索到的资料）与 query（用户问题）
   - 但调用方只想传一个 {"query": "..."}，不希望手动准备 context
   - 解决方案：在链最前面放一个「并行字典」，一个分支去检索生成 context，
     另一个分支把原始 query 透传出来，两者汇总成 prompt 需要的完整输入

2. RunnableParallel 的职责定位（回顾）
   - 官方定义：用于「操作 Runnable 的输出，以匹配序列中下一个 Runnable 的输入」，
     并行运行多个 Runnable 并格式化输出结构
   - 本示例正是用到它的「格式化输出结构」职责：把上游单一输入整形成下游 prompt 需要的两字段字典
     （「并行执行」职责见 1.RunnableParallel使用技巧.py）

3. LCEL 中 dict 的隐式转换规则（关键知识点）
   - 在管道中直接写 dict，__or__ / __ror__ 会通过 coerce_to_runnable 自动转成 RunnableParallel
   - dict 的 value 若是普通函数 / lambda，会被自动包装成 RunnableLambda；
     value 若是 dict，会递归包装成嵌套的 RunnableParallel
   - 因此 {"context": lambda x: ..., "query": itemgetter("query")} 完全等价于
     RunnableParallel(context=RunnableLambda(...), query=RunnableLambda(itemgetter("query")))

4. 等价写法：显式 RunnableParallel
   - 课程文档给出的同语义写法是显式传参，value 甚至可以直接传普通函数（由 coerce_to_runnable 包装）：
       chain = RunnableParallel(
           context=retrieval,           # retrieval 是普通函数，自动包成 RunnableLambda
           query=RunnablePassthrough(), # 透传整个输入
       ) | prompt | llm | parser
   - 两个分支并行执行：retrieval(x["query"]) 生成 context，RunnablePassthrough() 把原始输入整体作为 query
   - 与隐式 dict 写法输出一致，区别在于显式写法更直观、隐式写法更简洁

5. operator.itemgetter 的作用与优势
   - itemgetter("query") 返回一个可调用对象，效果等同 lambda x: x["query"]
   - 优势：C 实现，性能更好；语义明确，专用于「从 dict 取字段」
   - 课程文档提示：若输入是字典且只想取其中一部分，itemgetter 是标准做法
   - 在 LCEL 中极为常见，是从上游字典提取单个字段的惯用写法

6. lambda 在 LCEL 中的自动包装
   - lambda x: retrieval(x["query"]) 会被包装成 RunnableLambda
   - RunnableLambda 让任意 Python 函数获得 Runnable 能力（invoke / batch / stream）
   - 函数只接受「一个参数」，即上游传来的完整输入；多参数需借助 functools.partial 或 bind

7. 数据流转全过程
   {"query": "你好，我是谁?"}
        ├─→ lambda x: retrieval(x["query"]) ──→ "我是慕小课"        → context
        └─→ itemgetter("query")            ──→ "你好，我是谁?"      → query
                                    ↓
        {"context": "我是慕小课", "query": "你好，我是谁?"}
                                    ↓ prompt
        [HumanMessage(填充好 context 与 query 的完整提示词)]
                                    ↓ llm
        AIMessage(content="...")
                                    ↓ parser
        "最终回答字符串"

8. 与 RunnablePassthrough.assign 的对比（见本章第 3 个示例）
   - 本写法：必须「显式列出」prompt 需要的每一个字段，漏写会报 KeyError
   - assign 写法：自动保留原输入的所有字段，只需追加新增的 context 字段，更简洁
   - 字段多时推荐 assign；需要完全重塑输入结构时用显式 dict 更清晰

9. 注意事项
   - 字典分支是并行执行的，retrieval 与 itemgetter 同时跑（itemgetter 极快，感知不到）
   - 检索函数若有副作用（打印日志、写库），并行下的执行顺序不保证
   - 真实 RAG 中 retrieval 通常返回 List[Document]，
     需再拼接成字符串（"\\n\\n".join(doc.page_content for doc in docs)）后填入 context

===================================================================================
"""
from operator import itemgetter

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()


def retrieval(query: str) -> str:
    """
    一个模拟的检索器函数

    作用：模拟真实 RAG 中的向量检索环节，根据用户问题返回相关上下文

    参数：
        query: str - 用户的原始提问文本

    返回：
        str - 检索到的上下文文本（这里固定返回一段模拟资料）

    真实场景对照：
        真实实现通常是 vector_store.similarity_search(query, k=4)，
        返回 List[Document] 后再用 "\\n\\n".join(doc.page_content for doc in docs) 拼接
    """
    print("正在检索:", query)
    return "我是慕小课"


# 1.编排prompt
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建包含两个变量占位符的提示词模板
#   参数：template - 模板字符串，含 {context}（检索上下文）与 {query}（用户问题）
#   返回：ChatPromptTemplate 实例，invoke 时必须同时提供 context 与 query 两个键
#   说明：<context></context> 标签是常见的提示词工程技巧，帮助模型明确区分资料边界
prompt = ChatPromptTemplate.from_template("""请根据用户的问题回答，可以参考对应的上下文进行生成。

<context>
{context}
</context>

用户的提问是: {query}""")

# 2.构建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，invoke 接收 PromptValue/消息列表，返回 AIMessage
llm = ChatOpenAI(model="deepseek-flash")

# 3.输出解析器
# StrOutputParser() -> StrOutputParser
#   作用：从 AIMessage 中提取纯文本内容
#   返回：StrOutputParser 实例，invoke 接收 AIMessage，返回 str
parser = StrOutputParser()

# 4.构建链
# 这里的字典会被 LCEL 隐式转换为 RunnableParallel，是本示例的核心。
#
#   "context": lambda x: retrieval(x["query"])
#     - lambda 被自动包装成 RunnableLambda
#     - 入参 x 是链的原始输入 dict，取出 x["query"] 交给 retrieval 执行检索
#     - 返回值写入输出字典的 "context" 键
#
#   "query": itemgetter("query")
#     - itemgetter("query") 等价于 lambda x: x["query"]，实现原样透传
#     - 必须显式写出来，否则 prompt 渲染时会因缺少 query 变量而报错
#
# 等价的显式写法：
#   from langchain_core.runnables import RunnableParallel, RunnableLambda
#   RunnableParallel(
#       context=RunnableLambda(lambda x: retrieval(x["query"])),
#       query=RunnableLambda(itemgetter("query")),
#   ) | prompt | llm | parser
#
# 数据流转：
#   {"query": ...} → RunnableParallel → {"context": ..., "query": ...}
#                  → prompt → messages → llm → AIMessage → parser → str
chain = {
            "context": lambda x: retrieval(x["query"]),
            "query": itemgetter("query"),
        } | prompt | llm | parser

# 5.调用链
# chain.invoke(input: dict) -> str
#   参数：input - 只需提供 {"query": "..."}，context 由链内部自动补齐
#   执行流程：
#     1. 并行字典拿到 {"query": "你好，我是谁?"}
#     2. context 分支调用 retrieval("你好，我是谁?")，打印检索日志并返回 "我是慕小课"
#     3. query 分支通过 itemgetter 原样取出 "你好，我是谁?"
#     4. 汇总成 {"context": "我是慕小课", "query": "你好，我是谁?"} 传给 prompt
#     5. prompt 渲染成完整提示词 → llm 生成 AIMessage → parser 提取字符串
#   返回：str - 大模型基于检索上下文给出的最终回答
content = chain.invoke({"query": "你好，我是谁?"})

print(content)

# ===================================================================================
# 最佳实践与其他调用方式
# ===================================================================================
#
# 1. 更推荐的写法（见本章第 3 个示例）：使用 RunnablePassthrough.assign 只追加 context，
#    无需手动透传 query，输入字段多时优势明显：
#    from langchain_core.runnables import RunnablePassthrough
#    chain = RunnablePassthrough.assign(
#        context=lambda x: retrieval(x["query"])
#    ) | prompt | llm | parser
#
# 2. 真实 RAG 中把检索器接入链的标准写法（retriever 本身就是 Runnable）：
#    chain = {
#        "context": itemgetter("query") | retriever | format_docs,
#        "query": itemgetter("query"),
#    } | prompt | llm | parser
#    其中 format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
#
# 3. 批量调用：
#    results = chain.batch([{"query": "我是谁?"}, {"query": "慕小课是什么?"}])
#
# 4. 流式调用（检索部分先完成，随后逐 token 输出回答）：
#    for chunk in chain.stream({"query": "你好，我是谁?"}):
#        print(chunk, end="", flush=True)
#
# 5. 常见错误与解决方案：
#    - 错误：Input to ChatPromptTemplate is missing variables {'query'}
#      原因：并行字典中忘记写 "query": itemgetter("query") 这一分支
#      解决：补齐所有模板变量，或改用 RunnablePassthrough.assign
#
#    - 错误：TypeError: 'str' object is not subscriptable
#      原因：链的输入传了字符串而非字典，导致 x["query"] 取值失败
#      解决：invoke 时传 dict；若确实想传字符串，可在最前面加
#            {"query": RunnablePassthrough()} 做一次包装
#
#    - 错误：lambda 需要多个参数时报 TypeError
#      解决：RunnableLambda 的函数只接受单个入参，多参数请用
#            functools.partial 或 .bind() 预先绑定
