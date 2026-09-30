#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 22:36
@Author  : thezehui@gmail.com
@File    : 1.weaviate嵌入向量数据库示例.py

===================================================================================
知识点讲解：Weaviate 向量数据库
===================================================================================

1. 什么是 Weaviate
   - Weaviate 是完全用 Go 语言构建的开源云原生向量数据库，支持自托管与云服务（WCS）两种部署
   - 核心特点：GraphQL 原生查询、混合检索（向量+BM25）、强大的 schema 管理、内置多种向量化模块（OpenAI、Cohere、HuggingFace 等）
   - 与 Pinecone/TCVectorDB 一样有「集合」概念，集合类似关系型数据库的表，管理一类数据对象
   - 依赖安装：pip install langchain-weaviate weaviate-client

2. 部署方式与适用场景
   - Weaviate 云：官方托管，支持数据复制、零停机更新、无缝扩容；免费账号每实例最多 14 天，付费不限
   - Docker 部署：docker run 一条命令即可，暴露 8080(REST) 与 50051(gRPC) 端口，适合评估/开发
   - K8s 部署：适用于开发和生产
   - 嵌入式 Weaviate：基于本地文件，仅 Linux/macOS 支持，适合评估
   - 常用运维命令：docker start/stop/rm、docker ps / docker ps -a、docker images / docker rmi

3. 使用 Weaviate 的标准流程
   - 创建部署（云/Docker/K8s）→ 安装客户端与 LangChain 集成包 → 连接 Weaviate → 创建数据集/集合 → 添加数据/向量 → 相似性搜索（含过滤器）
   - 关键约束：集合名字必须以大写字母开头，且只能包含字母、数字、下划线，否则创建报错（与 Python 类名规范几乎一致）
   - 集合可在代码创建，也可在可视化管理界面（console.weaviate.cloud）创建；LangChain 会在使用时自动检测集合是否存在，不存在则直接创建

4. 连接方式
   - 本地连接 connect_to_local("host", "port")：用于 Docker 自托管实例，常用于开发测试
   - 云服务连接 connect_to_wcs(cluster_url=..., auth_credentials=AuthApiKey(...))：连接 WCS 托管实例，后台面板提供 REST 与 gRPC 两种地址及 API 密钥
   - 客户端对象 weaviate.WeaviateClient 封装了 REST/gRPC 连接，LangChain 组件底层复用该连接

5. Weaviate 核心概念
   - Class（类/集合）：相当于关系数据库的表，本例为 "Dataset"（注意大写开头命名）
   - Object（对象）：集合中的一条记录，含 id、properties、vector
   - Property（属性）：结构化数据字段，如 text、page、account_id
   - Schema（模式）：定义 Class 结构，包括属性类型、向量化配置、索引策略

6. 过滤查询的特点
   - 使用面向对象的 Filter API，而非字符串表达式或 JSON 字典
   - 链式调用：Filter.by_property("page").greater_or_equal(5)；.equal("pdf")；.contains_any(["ai","ml"])
   - 支持逻辑组合：& (and)、| (or)、~ (not)
     示例：(Filter.by_property("page") >= 5) & (Filter.by_property("type") == "pdf")

7. WeaviateVectorStore 关键参数
   - client：WeaviateClient 实例，封装连接信息
   - index_name：集合（Class）名称，相当于表名
   - text_key：原文在 properties 中的字段名，默认 "text"
   - embedding：嵌入模型，用于客户端向量化（也可配置 Weaviate 内置向量化模块）

8. Retriever 模式
   - as_retriever() 把 VectorStore 转换为 LangChain Retriever 接口
   - 统一返回 List[Document]（不含分数），适合直接接入 RAG 链
   - 默认调用 similarity_search，可通过 search_type 切换为 MMR 等策略

9. 适用场景与局限
   - 适用：需要混合检索（向量+关键词）的知识库、灵活 schema 与复杂元数据查询、需自托管又要完整数据库特性、多租户 SaaS（原生租户级物理隔离）
   - 局限：自托管运维成本高于 Serverless（Pinecone）；社区生态/文档完善度低于 Faiss、Pinecone；gRPC 在部分网络环境有兼容性问题

10. 最佳实践
    - 生产环境使用 WCS 或 K8s 部署，避免单机 Docker 的可靠性风险
    - 善用 Weaviate 内置向量化模块（如 text2vec-openai），可省去客户端嵌入调用
    - 复杂过滤条件应在 Weaviate 层完成，不要应用层后置过滤以保证性能
    - 定期备份 schema 定义，Class 删除后无法恢复
    - 更换嵌入模型需重建整个 Class

===================================================================================
"""
import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.原始文本数据与元数据
# 主题分散（猫、音乐、学习、食物、梦境、手机、阅读、野餐、狗），
# 便于观察语义检索的排序与过滤效果
texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]
# page 字段：用于演示数值范围过滤
# account_id 字段：第 6 条独有，用于演示字段存在性过滤
metadatas = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6, "account_id": 1},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]

# 2.创建 Weaviate 连接客户端
# weaviate.connect_to_local(host, port) 作用：连接本地自托管的 Weaviate 实例
#   参数：host -> str，Weaviate 服务的 IP 地址或域名（本例为 "192.168.2.120"）
#         port -> str，Weaviate 服务端口，默认 8080（gRPC 端口为 50051）
#   返回：weaviate.WeaviateClient，封装了 gRPC/HTTP 连接的客户端对象
#   用途：用于本地开发、测试环境或私有化部署场景
# 备注：生产环境可改用 connect_to_wcs 连接云服务，需提供 cluster_url 与 auth_credentials
client = weaviate.connect_to_local("192.168.2.120", "8080")

# 备选：连接 Weaviate Cloud Services（WCS）托管实例（已注释）
# weaviate.connect_to_wcs(cluster_url, auth_credentials) 作用：连接 WCS 云端实例
#   cluster_url      : WCS 控制台提供的集群 URL
#   auth_credentials : AuthApiKey 对象，封装 API Key 用于鉴权
#   优势：免运维、自动扩容、全球多区域部署、企业级 SLA
# client = weaviate.connect_to_wcs(
#     cluster_url="https://eftofnujtxqcsa0sn272jw.c0.us-west3.gcp.weaviate.cloud",
#     auth_credentials=AuthApiKey("21pzYy0orl2dxH9xCoZG1O2b0euDeKJNEbB0"),
# )

# 创建嵌入模型
# text-embedding-3-small 输出 1536 维向量，
# Weaviate 首次写入时会根据向量维度自动配置 Class 的向量索引
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 3.创建 LangChain 向量数据库实例
# WeaviateVectorStore 参数说明：
#   client=client
#       上面创建的 weaviate.WeaviateClient 连接对象，封装了服务端地址与鉴权信息
#   index_name="Dataset"
#       Weaviate 的 Class 名称（相当于表名）。若不存在，LangChain 会自动创建；
#       若已存在，会复用现有 Class（需注意 schema 兼容性）
#   text_key="text"
#       原文在 Weaviate Object properties 中的字段名。
#       LangChain 会把 Document.page_content 写入该字段，检索时也从该字段回取原文
#   embedding=embedding
#       嵌入模型，用于客户端向量化。写入时调用 embed_documents，检索时调用 embed_query
# 执行副作用：实例化时会检查 Class "Dataset" 是否存在，不存在则创建，
#            并根据首次写入的向量维度自动配置 vectorIndexConfig
db = WeaviateVectorStore(
    client=client,
    index_name="Dataset",
    text_key="text",
    embedding=embedding,
)

# 4.添加数据
# add_texts(texts, metadatas) 参数与返回：
#   texts     : Iterable[str]，待写入的文本列表
#   metadatas : Optional[List[dict]]，元数据列表，长度需与 texts 严格相等且顺序对应
#   返回      : List[str]，写入成功的 Weaviate Object UUID 列表
# 执行流程：
#   a) 调用 embedding.embed_documents(texts) 批量把文本转为 1536 维向量
#   b) 把 text（原文）、metadata 字段、vector 组装为 Weaviate Object
#   c) 通过 client.batch 批量写入 "Dataset" Class
#   d) 返回 Weaviate 自动生成的 UUID 列表（用于后续删除或更新）
ids = db.add_texts(texts, metadatas)
print(ids)

# 5.执行带过滤条件的相似性搜索
# Filter.by_property(name).greater_or_equal(value) 作用：构造数值范围过滤条件
#   by_property("page")   : 指定过滤字段为 page
#   greater_or_equal(5)   : 大于等于 5（page >= 5）
#   返回：weaviate.classes.query.Filter 对象，可用 & | ~ 组合
# 常用过滤方法：
#   equal(value)                  : 等于
#   not_equal(value)              : 不等于
#   greater_than(value)           : 大于
#   less_than(value)              : 小于
#   less_or_equal(value)          : 小于等于
#   contains_any(values)          : 数组包含任意值
#   contains_all(values)          : 数组包含所有值
filters = Filter.by_property("page").greater_or_equal(5)

# similarity_search_with_score(query, filters) 参数与返回：
#   query   : str，查询文本，会被向量化后做语义匹配
#   k       : int，返回条数，默认 4（未显式传入）
#   filters : Optional[Filter]，Weaviate 的面向对象过滤条件
#   返回    : List[Tuple[Document, float]]，float 为 cosine 相似度（Weaviate 默认度量）
# 执行流程：
#   a) 调用 embedding.embed_query("笨笨") 把查询转为向量
#   b) 向 Weaviate 发起 nearVector 查询，附带 where 过滤条件（filters）
#   c) Weaviate 先应用 pre-filtering（只在 page >= 5 的记录中检索），再做 ANN 检索
#   d) 返回 top-k 结果与 cosine 距离，LangChain 转换为 (Document, score) 元组
# 预期结果：虽然 query 问的是「笨笨」（语义上应匹配 page=1 的猫咪），
#          但由于过滤条件 page >= 5，候选集只剩 page 5~10 的 6 条记录，
#          其中「我的狗喜欢追逐球」（page=10）因宠物语义域相关性可能排在前列
print(db.similarity_search_with_score("笨笨", filters=filters))

# 6.转换为 Retriever 并执行检索
# as_retriever() 作用：把 VectorStore 转换为 LangChain Retriever 接口
#   参数：search_type -> str，可选 "similarity"（默认）、"mmr"、"similarity_score_threshold"
#         search_kwargs -> dict，传给底层 similarity_search 的参数，如 {"k": 10, "filters": ...}
#   返回：VectorStoreRetriever 实例，实现 Retriever 接口（invoke / ainvoke）
#   用途：Retriever 统一返回 List[Document] 而不含分数，适合直接接入 RAG 链的 retriever 参数
retriever = db.as_retriever()

# retriever.invoke(query) 作用：执行检索并返回文档列表
#   参数：query -> str，查询文本
#   返回：List[Document]，不含相似度分数（Retriever 接口标准）
#   底层：实际调用 db.similarity_search(query)，参数由 as_retriever 的 search_kwargs 控制
# 注意：此处未传 filters，因此检索全量数据（10 条），预期「笨笨是一只很喜欢睡觉的猫咪」排第一
print(retriever.invoke("笨笨"))
