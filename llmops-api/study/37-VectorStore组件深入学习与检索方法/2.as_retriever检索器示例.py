#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 8:06
@Author  : thezehui@gmail.com
@File    : 3.最大边际相关性示例.py

===================================================================================
知识点讲解：as_retriever() 把向量数据库转换为检索器
===================================================================================
1. VectorStore 与 Retriever 的本质区别
   - VectorStore 是"存储层"，负责增删改查向量，方法名各异（similarity_search、
     max_marginal_relevance_search 等），不同实现签名可能不一致
   - Retriever 是"接口层"，统一暴露 invoke(query) -> list[Document]，并实现 Runnable 协议，
     可直接参与 LCEL 管道编排
   - as_retriever() 就是把存储层包装为标准 Runnable 的适配器（VectorStoreRetriever）

2. as_retriever() 的三种 search_type
   - "similarity"（默认）：纯相似度排序，取 Top-K，不做阈值过滤
   - "similarity_score_threshold"：相似度 + 阈值过滤，底层调用
     similarity_search_with_relevance_scores，search_kwargs 必须包含 score_threshold
   - "mmr"：最大边际相关性，在相关性与多样性之间权衡，避免返回一堆重复内容

3. search_kwargs 常用参数与强制约束
   - k：返回文档条数
   - score_threshold：相关性阈值（仅 similarity_score_threshold 生效）
   - fetch_k：MMR 候选池大小（仅 mmr 生效，默认 20）
   - lambda_mult：MMR 相关性/多样性权重（仅 mmr 生效，默认 0.5）
   - filter：元数据过滤条件，不同向量库语法不同
   - 强制约束：search_type 配置为 similarity_score_threshold 后，search_kwargs 必须
     添加 score_threshold，否则会直接报错；各参数确切含义要结合对应底层函数一起看

4. 检索模式全景与选择
   - similarity（默认） → similarity_search：通用，约 80% 场合够用
   - similarity_score_threshold → similarity_search_with_relevance_scores：小库/要求准确、
     必须排除噪声时使用，必须传 score_threshold
   - mmr → max_marginal_relevance_search：追求创新/创意/多样性、"有哪些"类列举式提问
   - 经验结论：约 80% 场合用相似性搜索即可得到不错效果；只有追求多样性时才考虑 MMR
   - 使用相似性搜索时，尽可能选 similarity_search_with_relevance_scores 并传阈值，
     确保向量库数据较少时也不把不相关数据检索出来

5. 为什么 Retriever 是 RAG 的标准接口
   - 因为实现了 Runnable，可以写成 {"context": retriever, "question": ...} | prompt | llm
   - 换底层存储（FAISS → Weaviate → Pinecone）时，上层链代码完全不用改
   - 天然支持回调追踪（callbacks）、异步（ainvoke）、批量（batch）、运行时配置
   - 它的真正价值在于把检索器变成 Runnable，从而让查询转换阶段的多种策略
     （多查询重写、RAG-Fusion、问题分解、Step-Back、HyDE、集成检索）
     都能以"包装一个基础 retriever"的方式实现
   - 可作为优化对象的 RAG 组件共 5 处：query、TextSplitter、VectorStore、Retriever、Prompt，
     本节同时涉及 TextSplitter、VectorStore、Retriever 三处

6. WeaviateVectorStore 关键参数
   - client：weaviate 连接客户端，通过 connect_to_wcs 连接云端集群
   - index_name：Weaviate 中的 Collection 名称（相当于表名），不存在会自动创建
   - text_key：正文字段名，Weaviate 用该字段存储 Document.page_content
   - embedding：嵌入模型，入库和检索都会用它

7. 本例完整数据流水线
   - Markdown 文件 → UnstructuredMarkdownLoader.load() → 原始 Document
   - → RecursiveCharacterTextSplitter.split_documents() → 若干 chunk
   - → db.add_documents() → 向量化并写入 Weaviate
   - → db.as_retriever() → 标准检索器
   - → retriever.invoke(query) → 相关 Document 列表

8. 典型输出示例与观察点
   - 查询 "关于配置接口的信息有哪些"，配置 k=10 + score_threshold=0.5，
     得到 10 条文档的前 50 字符，例如：
       '接口说明：用于更新对应应用的调试长记忆内容，如果应用没有开启长记忆功能…'
       '如果接口需要授权，需要在 headers 中添加 Authorization ，并附加 access…'
       '1.2 [todo]更新应用草稿配置信息…'
       …（共 10 条）
     len(documents) == 10
   - 观察要点：这份 API 文档语料规模足够、且与查询高度同主题，
     所以 10 条都跨过了 0.5 阈值；换成小语料或跨主题查询时，
     返回条数会明显小于 k，这才是阈值真正在起作用的信号

9. 版本对照说明（示例基于 LangChain 1.x）
   - as_retriever() 的 search_type / search_kwargs 两个参数在 0.x → 1.x 之间未变
   - BaseRetriever.get_relevant_documents() 从 0.3.0 起被弃用，统一改用 Runnable 的 invoke()；
     本示例已按 1.x 写法使用 invoke()
   - 文档中 weaviate.connect_to_wcs() 在较新 weaviate 客户端里已更名为
     connect_to_weaviate_cloud()，老名字仍可用但会告警；迁移时需留意

10. 最佳实践建议
   - 业务代码始终面向 Retriever 编程，不要直接调用 VectorStore 的检索方法，便于后续替换实现
   - 使用 similarity_score_threshold 时务必同时传 k 与 score_threshold，缺一会告警或不生效
   - add_documents 幂等性较弱，重复执行脚本会导致库中出现重复数据；
     正式项目应先按业务主键去重，或用 delete + upsert 逻辑
   - API Key 硬编码在源码中仅适用于教学示例，生产环境必须通过 .env / 密钥管理服务注入
   - Weaviate 客户端是长连接资源，脚本型任务建议用 with 语句或显式 client.close() 释放
   - k 值过大会让 Prompt 超长并引入噪声，通常 3~10 条足够；阈值与 k 要配合调优

===================================================================================
"""
import dotenv
import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OPENAI_API_KEY 等凭证
dotenv.load_dotenv()

# 1.构建加载器与分割器
# UnstructuredMarkdownLoader(file_path: str, mode: str = "single") -> UnstructuredMarkdownLoader
#   作用：基于 unstructured 库解析 Markdown 文件，剥离标记语法并提取纯文本
#   参数：file_path - Markdown 文件路径（相对路径基于脚本运行目录）
#   返回：文档加载器实例，调用 load() 才真正读取文件
loader = UnstructuredMarkdownLoader("./项目API文档.md")

# RecursiveCharacterTextSplitter(
#     separators: list[str],
#     is_separator_regex: bool,
#     chunk_size: int,
#     chunk_overlap: int,
#     add_start_index: bool
# ) -> RecursiveCharacterTextSplitter
#   作用：递归字符分割器，按分隔符优先级逐级降级切分，尽量保持语义完整
#   参数详解：
#     separators         - 分隔符优先级列表，从"段落"到"单字符"逐级降级：
#                          "\n\n"（空行/段落）→ "\n"（换行）→ 中文句末标点
#                          → 英文句末标点 → 分号 → 逗号 → 空格 → ""（强制按字符切）
#     is_separator_regex - True 表示 separators 中的元素按正则表达式解析，
#                          因此 "。|！|？" 这种写法才能同时匹配三种中文标点
#     chunk_size         - 单个 chunk 的最大长度（此处按字符数计算，约 500 字）
#     chunk_overlap      - 相邻 chunk 的重叠字符数，50 字重叠可避免答案正好被切断
#     add_start_index    - True 时在 metadata 中写入 start_index（chunk 在原文的起始偏移），
#                          便于后续做引用定位和高亮
#   返回：文本分割器实例
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", r"；|;\s", r"，|,\s", " ", "", ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# 2.加载文档并分割
# loader.load() -> list[Document]
#   作用：读取并解析 Markdown 文件，返回 Document 列表
#   返回：mode="single"（默认）时整个文件合并为 1 个 Document，
#         metadata 中包含 source（文件路径）等信息
documents = loader.load()

# text_splitter.split_documents(documents: list[Document]) -> list[Document]
#   作用：把长 Document 切分为多个小 chunk，并自动继承原 Document 的 metadata
#   参数：documents - 待分割的文档列表
#   返回：分割后的 Document 列表，每条长度不超过 chunk_size
#   为什么必须分割：
#     1) Embedding 模型有最大输入长度限制（8191 token），超长会被截断
#     2) 向量是整段语义的"平均"，段落越长语义越模糊，检索精度越差
#     3) 送入 LLM 的上下文长度有限，需要按 chunk 粒度精确投喂
chunks = text_splitter.split_documents(documents)

# 3.将数据存储到向量数据库
# weaviate.connect_to_wcs(cluster_url: str, auth_credentials: AuthCredentials) -> WeaviateClient
#   作用：连接 Weaviate Cloud Service（WCS）托管集群
#   参数：cluster_url      - 集群 REST 端点地址
#         auth_credentials - 鉴权凭证，AuthApiKey(api_key) 表示使用 API Key 鉴权
#   返回：Weaviate v4 客户端实例（内部维护 HTTP + gRPC 长连接）
#
# WeaviateVectorStore(
#     client: WeaviateClient,
#     index_name: str,
#     text_key: str,
#     embedding: Embeddings
# ) -> WeaviateVectorStore
#   作用：把 Weaviate 集群包装为 LangChain VectorStore
#   参数：index_name - Collection（类）名称，首字母需大写，不存在时首次写入自动创建
#         text_key   - 存放正文的属性名，检索时从该字段还原 page_content
#         embedding  - 嵌入模型，LangChain 侧生成向量后再写入 Weaviate
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

# db.add_documents(documents: list[Document]) -> list[str]
#   作用：批量向量化并写入向量数据库
#   参数：documents - 待写入的 chunk 列表
#   返回：写入成功的文档 ID 列表（Weaviate 中为 UUID）
#   内部执行流程：
#     1) 提取所有 page_content
#     2) 调用 embedding.embed_documents() 批量生成向量（网络请求，耗时主要在这里）
#     3) 通过 gRPC 批量写入 Weaviate，向量 + 正文 + metadata 一起落库
#   注意：重复运行本脚本会重复写入相同内容，导致检索出现大量重复文档
db.add_documents(chunks)

# 4.转换检索器（带阈值的相似性搜索，数据为10条，得分阈值为0.5）
# db.as_retriever(search_type: str = "similarity", search_kwargs: dict = None) -> VectorStoreRetriever
#   作用：把 VectorStore 包装为实现 Runnable 协议的标准检索器
#   参数：search_type   - 检索策略，可选 "similarity" / "similarity_score_threshold" / "mmr"
#         search_kwargs - 传给底层检索方法的参数字典
#   本例配置说明：
#     search_type="similarity_score_threshold"
#       → 内部调用 similarity_search_with_relevance_scores()
#     k=10               → 最多返回 10 条
#     score_threshold=0.5 → 只保留相关性得分 >= 0.5 的文档
#   返回：VectorStoreRetriever 实例，可直接 invoke / 参与 LCEL 管道
#   效果：实际返回条数 <= 10，取决于有多少文档跨过 0.5 阈值，甚至可能为 0 条
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5},
)

# 5.检索结果
# retriever.invoke(input: str, config: RunnableConfig = None) -> list[Document]
#   作用：执行检索，这是 Runnable 协议的统一入口
#   参数：input - 查询字符串（注意 Retriever 的输入是 str，不是 dict）
#   返回：相关 Document 列表，按相关性降序
#   内部执行流程：
#     invoke() → BaseRetriever.invoke() → _get_relevant_documents()
#     → VectorStoreRetriever 根据 search_type 分发到对应的 VectorStore 方法
#     → 相似度阈值过滤 → 返回 Document 列表
#   数据流转："关于配置接口的信息有哪些"
#              → embed_query 生成查询向量
#              → Weaviate 向量近邻检索（HNSW 索引）
#              → 归一化得分并过滤 < 0.5 的结果
#              → Document 列表
documents = retriever.invoke("关于配置接口的信息有哪些")

# 打印每条命中文档的前 50 个字符，快速肉眼检查检索质量
print(list(document.page_content[:50] for document in documents))

# 打印实际命中条数
# 若明显小于 k=10，说明阈值 0.5 过滤掉了大量弱相关文档，这正是阈值的预期作用
print(len(documents))
