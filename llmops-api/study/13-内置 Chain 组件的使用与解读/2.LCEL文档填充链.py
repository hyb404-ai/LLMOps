#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/21 9:30
@Author  : thezehui@gmail.com
@File    : 2.LCEL文档填充链.py

===================================================================================
知识点讲解：Stuff 文档填充链（create_stuff_documents_chain）
===================================================================================

1. 什么是文档填充链（Stuff Documents Chain）
   - 将多个 Document 对象的内容拼接后，"塞入"（stuff）提示词的上下文变量中
   - 是 RAG 检索增强生成流程中最简单直接的文档处理策略
   - 适用于文档总量较小、能完整放入模型上下文窗口的场景

2. Document 对象的结构
   - page_content: str - 文档正文内容，是拼接的主要数据源
   - metadata: dict - 文档元数据（来源、页码、时间等），默认不参与拼接
   - 通常由 DocumentLoader 加载或由 Retriever 检索产出

3. create_stuff_documents_chain 的工作原理
   - 接收一个 llm 和一个 prompt，返回一个 Runnable 链
   - 内部用 format_document 将每个 Document 格式化为字符串
   - 用 document_separator（默认 "\n\n"）连接所有文档内容
   - 将拼接结果注入 prompt 的 context 变量（由 document_variable_name 指定，默认 "context"）

4. 四种常见的文档处理策略对比
   - Stuff：全部塞入、一次调用，最简单但受上下文窗口限制
   - MapReduce：先对每篇文档单独处理再汇总，可处理大量文档
   - Refine：迭代式逐个文档精炼答案，质量高但耗时长
   - MapRerank：对每篇文档独立打分，选取最佳答案

5. 提示词中 context 变量的约定
   - prompt 必须包含 {context} 占位符，否则会抛出异常
   - 推荐用 XML 标签（如 <context>...</context>）包裹，帮助模型区分上下文边界
   - 其他变量（如 {query}）由调用方在 invoke 时传入

6. 典型输出示例与观察点（课程示例）
   - 对「大家都喜欢什么颜色」提问，传入三篇文档，模型会跨文档归纳：
       根据提供的信息：
       - 小明喜欢绿色，不喜欢黄色。
       - 小王喜欢粉色，也有一点喜欢红色。
       - 小泽喜欢蓝色，但更喜欢青色。
       所以，小明喜欢绿色，小王喜欢粉色和红色（少许），小泽喜欢青色和蓝色。
   - 观察点：Stuff 策略把全部上下文一次性喂给模型，由其自行跨文档汇总

7. 注意事项与最佳实践
   - 文档过多会超出上下文窗口，导致报错或内容截断
   - 生产环境建议先用 Retriever 筛选 Top-K 相关文档再填充
   - 可通过 document_prompt 自定义单个文档的格式（如包含来源信息）

8. 数据流转过程
   - {"query": str, "context": list[Document]}
   - → 文档拼接：list[Document] → str（以 document_separator 连接）
   - → prompt 填充：{"query": str, "context": str} → Messages
   - → llm 推理：Messages → AIMessage → 输出解析 → str

===================================================================================
"""
import dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量（API Key 等）
dotenv.load_dotenv()

# 1.创建提示模板
# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：根据消息列表创建多角色聊天提示词模板
#   参数：messages - 元组列表，每个元组为 (role, template)
#         role 可取值：system / human / ai / placeholder
#   返回：ChatPromptTemplate 实例
#   关键约定：模板中必须包含 {context} 占位符，供文档内容注入
#   最佳实践：使用 <context> XML 标签包裹上下文，帮助模型明确边界
prompt = ChatPromptTemplate.from_messages([
    # system 消息：定义角色与行为规则，并声明 context 上下文变量
    ("system", "你是一个强大的聊天机器人，能根据用户提供的上下文来回复用户的问题。\n\n<context>{context}</context>"),
    # human 消息：用户的实际提问，通过 query 变量传入
    ("human", "{query}")
])

# 2.创建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例，作为链的推理引擎
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，实现 Runnable 协议
llm = ChatOpenAI(model="deepseek-v4-pro")

# 3.创建链应用
# create_stuff_documents_chain(
#     llm: LanguageModelLike,
#     prompt: BasePromptTemplate,
#     output_parser: BaseOutputParser = None,
#     document_prompt: BasePromptTemplate = None,
#     document_separator: str = "\n\n",
#     document_variable_name: str = "context",
# ) -> Runnable[dict, Any]
#   作用：构建文档填充链，将文档列表拼接后注入提示词并调用 LLM
#   参数：
#     llm - 用于推理的语言模型（必填）
#     prompt - 提示词模板，必须包含 document_variable_name 指定的变量（必填）
#     output_parser - 输出解析器，默认 StrOutputParser（输出为 str）
#     document_prompt - 单个文档的格式化模板，默认只取 page_content
#     document_separator - 多个文档之间的连接符，默认两个换行
#     document_variable_name - 提示词中接收文档内容的变量名，默认 "context"
#   返回：Runnable 链，invoke 时接收 dict（含 context 为 Document 列表）
#   内部流程：
#     1. 从输入 dict 中取出 context（Document 列表）
#     2. 用 document_prompt 格式化每个 Document
#     3. 用 document_separator 拼接为单个字符串
#     4. 替换输入 dict 中的 context 为拼接后的字符串
#     5. 依次执行 prompt → llm → output_parser
chain = create_stuff_documents_chain(prompt=prompt, llm=llm)

# 4.文档列表
# Document(page_content: str, metadata: dict = {}) -> Document
#   作用：创建 LangChain 标准文档对象
#   参数：
#     page_content - 文档正文内容，是被拼接进 context 的部分
#     metadata - 文档元数据（可选），如 {"source": "file.txt", "page": 1}
#   返回：Document 实例
#   说明：生产环境中这些 Document 通常由 Retriever 从向量库检索得到
documents = [
    Document(page_content="小明喜欢绿色，但不喜欢黄色"),
    Document(page_content="小王喜欢粉色，也有一点喜欢红色"),
    Document(page_content="小泽喜欢蓝色，但更喜欢青色"),
]

# 5.调用链
# chain.invoke(input: dict) -> str
#   作用：执行文档填充链，基于文档内容回答问题
#   参数：input - 必须包含两个键
#     query: str - 用户的提问，对应 prompt 中的 {query}
#     context: list[Document] - 文档列表，会被拼接后注入 {context}
#   返回：LLM 生成的字符串答案
#   数据流转：
#     {"query": "...", "context": [Doc1, Doc2, Doc3]}
#     → context 拼接为 "小明喜欢绿色...\n\n小王喜欢粉色...\n\n小泽喜欢蓝色..."
#     → prompt 填充 → [SystemMessage(含拼接内容), HumanMessage(query)]
#     → llm 推理 → AIMessage
#     → StrOutputParser → str
content = chain.invoke({"query": "请帮我统计一下大家都喜欢什么颜色", "context": documents})

print(content)

# ==================== 最佳实践与扩展写法 ====================
# 1. 自定义文档分隔符与文档格式（携带来源信息）
# from langchain_core.prompts import PromptTemplate
# document_prompt = PromptTemplate.from_template("来源：{source}\n内容：{page_content}")
# chain = create_stuff_documents_chain(
#     llm=llm,
#     prompt=prompt,
#     document_prompt=document_prompt,
#     document_separator="\n---\n",
# )
# # 此时 Document 必须包含 metadata={"source": "..."}
#
# 2. 自定义上下文变量名（当 prompt 中不使用 context 时）
# chain = create_stuff_documents_chain(
#     llm=llm,
#     prompt=prompt,
#     document_variable_name="docs",   # prompt 中需写 {docs}
# )
#
# 3. 与 Retriever 组合构成完整 RAG 链（推荐生产用法）
# from langchain_classic.chains.retrieval import create_retrieval_chain
# retrieval_chain = create_retrieval_chain(retriever, chain)
# result = retrieval_chain.invoke({"input": "大家喜欢什么颜色"})
# # retriever 自动检索相关文档并填入 context，无需手动传递
#
# 4. 流式输出（提升交互体验）
# for chunk in chain.stream({"query": "...", "context": documents}):
#     print(chunk, end="", flush=True)
