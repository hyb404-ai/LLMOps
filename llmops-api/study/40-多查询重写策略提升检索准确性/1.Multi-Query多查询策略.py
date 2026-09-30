#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/3 9:25
@Author  : thezehui@gmail.com
@File    : 1.Multi-Query多查询策略.py

===================================================================================
知识点讲解：Multi-Query 多查询重写策略
===================================================================================

1. 单查询检索的核心痛点
   - 向量检索的效果高度依赖「用户提问的措辞」与「文档原文措辞」的向量距离
   - 用户表述随意、口语化、术语不统一时，即使知识库里有答案也可能召回失败
     例如用户问「怎么调参数」，文档里写的是「应用配置项说明」，字面和向量都不够近
   - 一次查询只能从「一个角度」探测向量空间，覆盖面天然有限

2. Multi-Query 的核心思想（查询扩展 / Query Expansion）
   - 让大语言模型把用户的原始问题改写成 N 个语义等价但措辞不同的查询
   - 用这 N 个查询分别去检索，然后把结果合并去重
   - 相当于从多个角度多次探测向量空间，显著提升召回率（Recall）
   - 这是「用 LLM 的语言能力补偿 Embedding 的语义盲区」

3. MultiQueryRetriever 的完整执行流程
   - 从 LangSmith 记录看，检索器先调用 LLM 生成 3 条与原始问题相关的子问题，再逐个用底层检索器检索，最后合并去重
   - 内部步骤：
     Step 1  用内置 DEFAULT_QUERY_PROMPT 让 LLM 生成多个（默认 3 个）改写查询
     Step 2  LineListOutputParser 按换行把 LLM 输出切分成查询列表
     Step 3  generate_queries()：得到查询列表（include_original=True 时把原问题也加进去）
     Step 4  retrieve_documents()：对每个查询调用底层 retriever.invoke()
     Step 5  unique_union()：基于 page_content 去重合并，返回最终文档列表

4. from_llm() 工厂方法的关键参数
   - retriever：底层基础检索器，真正执行向量检索（必填）
   - llm：用于改写查询的大语言模型（必填），温度应设为 0 保证稳定可复现
   - prompt：自定义查询生成提示模板，默认使用英文的 DEFAULT_QUERY_PROMPT
     （You are an AI language model assistant... Provide these alternative questions separated by newlines. Original question: {question}）
     中文场景建议自定义中文 Prompt（如「你是一个AI语言模型助手…请用换行符分隔这些替代问题。原始问题：{question}」）以提升改写质量
   - include_original：是否把用户原始问题也纳入检索，默认 False；设为 True 更安全，防止 LLM 改写跑偏导致原本能召回的内容丢失
   - parser_key：已废弃参数，新版本忽略

5. unique_union 去重逻辑的注意点
   - 底层 _unique_documents 源码极朴素，仅循环遍历并记录唯一文档：
       def _unique_documents(documents):
           return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]
   - 去重基于 Document 的完整序列化内容（内容完全相同才算重复）
   - 去重后顺序不保证与相关性一致——因为本质是遍历去重，所以「只保证召回，不保证排序」
   - 这正是第 41 节 RAG-Fusion（用 RRF 算法重新排序）要解决的问题

6. 代价与权衡
   - 每次检索多出 1 次 LLM 调用（生成查询）+ N 倍的向量检索次数
   - 延迟从「1 次向量检索」变成「1 次 LLM + N 次向量检索」，通常上升到秒级
   - Token 成本增加，且返回文档数量膨胀（3 个查询 × k 条 ≈ 3k 条），可能超出 LLM 上下文窗口

7. 核心注意事项与最佳实践
   - 不同模型生成的 query 格式可能不按 \n 分割，效果可能不如原问题，务必多次测试 prompt 或设 include_original=True 兜底
   - 生成查询的 LLM 一定要 temperature=0，否则每次改写结果不同、检索不可复现
   - 中文知识库强烈建议自定义中文 prompt，默认英文 Prompt 对中文改写质量一般
   - 底层 retriever 配 mmr 可进一步提升多样性，与 Multi-Query 的多角度思想互补
   - 返回文档数量会成倍膨胀，务必在下游加 ReRank（第 52 节）或截断，控制 Prompt 长度
   - 延迟敏感场景（实时对话）要谨慎使用，可考虑缓存改写结果或降级为单查询
   - 想要「既提召回又保排序」，请使用第 41 节的 RAG-Fusion（RRF）方案
   - 多查询策略是最基础、最简单的 RAG 优化，不涉复杂算法，仅稍影响单次对话耗时

===================================================================================
"""
import dotenv
import weaviate
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()

# 1.构建向量数据库与检索器
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点地址
#         auth_credentials - AuthApiKey 包装的 API Key
#   返回：Weaviate v4 客户端
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：把已有 Weaviate Collection 包装为 LangChain 向量库
#   参数：index_name - Collection 名称，指向已写入 LLMOps 文档的 "DatasetDemo"
#         text_key   - 正文字段名
#         embedding  - 嵌入模型，必须与入库时保持一致
#   返回：向量库实例
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://eftofnujtxqcsa0sn272jw.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("21pzYy0orl2dxH9xCoZG1O2b0euDeKJNEbB0"),
    ),
    index_name="DatasetDemo",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# db.as_retriever(search_type: str) -> VectorStoreRetriever
#   作用：把向量库转换为基础检索器，作为 MultiQueryRetriever 的底层执行者
#   参数：search_type="mmr" - 使用最大边际相关性搜索
#   为什么选 mmr：Multi-Query 会产生多个相似的改写查询，
#                 它们的检索结果本来就容易重叠；
#                 用 mmr 让每个查询内部先保证多样性，合并后覆盖面更广
#   返回：VectorStoreRetriever 实例（默认 k=4）
retriever = db.as_retriever(search_type="mmr")

# 2.创建多查询检索器
# MultiQueryRetriever.from_llm(
#     retriever: BaseRetriever,
#     llm: BaseLanguageModel,
#     prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
#     include_original: bool = False,
#     ...
# ) -> MultiQueryRetriever
#   作用：工厂方法，自动装配"查询生成链 + 底层检索器"，构造多查询检索器
#   参数详解：
#     retriever        - 底层基础检索器，每个改写查询都会调用它执行一次真实检索
#     llm              - 负责改写查询的大语言模型；
#                        temperature=0 保证改写结果确定、可复现
#     include_original - True 表示把用户的原始问题也加入查询列表；
#                        这样即使 LLM 的 3 个改写全部跑偏，原始查询仍能召回内容，
#                        是一条重要的保底策略
#   返回：MultiQueryRetriever 实例（BaseRetriever 子类，具备完整 Runnable 能力）
#   内部装配细节：
#     from_llm 会构建 llm_chain = prompt | llm | LineListOutputParser()
#     其中 LineListOutputParser 负责把 LLM 的多行文本输出解析为 list[str]
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    include_original=True,
)

# 3.执行检索
# multi_query_retriever.invoke(input: str) -> list[Document]
#   作用：执行完整的多查询检索流程
#   参数：input - 用户的原始问题
#   返回：去重合并后的 Document 列表
#   完整数据流转（关键执行流程）：
#     1) invoke("关于LLMOps应用配置的文档有哪些")
#     2) generate_queries() → llm_chain 调用 LLM，生成类似：
#          "LLMOps 平台的应用配置项有哪些？"
#          "如何配置 LLMOps 应用？"
#          "LLMOps 应用设置相关文档在哪里？"
#        LineListOutputParser 按换行切分为 list[str]
#     3) 因 include_original=True，原始问题被追加进列表 → 共 4 个查询
#     4) retrieve_documents() 对 4 个查询分别调用 retriever.invoke()，
#        每次 mmr 返回 4 条 → 累计约 16 条（带重复）
#     5) unique_union() 基于文档内容去重 → 最终返回若干条唯一文档
#   耗时分布：1 次 LLM 调用（约 1~3 秒）+ 4 次向量检索（每次含 1 次 embed_query）
docs = multi_query_retriever.invoke("关于LLMOps应用配置的文档有哪些")

# 打印去重合并后的完整文档列表
print(docs)

# 打印最终文档条数
# 预期明显大于单查询的 4 条（去重后通常在 6~16 之间），
# 这个增量就是 Multi-Query 带来的召回率提升；
# 但注意此时的顺序不代表相关性排序（见知识点 5）
print(len(docs))
