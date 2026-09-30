#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/28 17:13
@Author  : thezehui@gmail.com
@File    : 1.faiss向量数据库使用示例.py

===================================================================================
知识点讲解：Faiss 本地向量数据库
===================================================================================

1. 什么是 Faiss
   - Faiss（Facebook AI Similarity Search）是 Meta 开源的高效向量相似性检索库
   - 本质是一个「向量索引库」而非完整数据库：没有服务进程、没有网络协议、
     没有权限体系，就是一个可以加载到内存中做近邻搜索的数据结构
   - 支持百万级到十亿级向量的快速 ANN（Approximate Nearest Neighbor）检索
   - 依赖安装：pip install faiss-cpu（CPU 版）；若已装 CUDA 可用 GPU 版 pip install faiss-gpu
   - 官网 https://faiss.ai/ 、仓库 https://github.com/facebookresearch/faiss
   - Faiss 使用 C++ 开发并提供了 Python 接口；在百万级向量的相似性检索中可实现 < 10ms 响应
     （需牺牲搜索准确度）；被广泛用于推荐系统、图片和视频搜索等业务

2. 向量数据库的部署方式分类
   - 本地文件向量数据库：向量数据存本地文件系统，通过查询接口检索，例如 Faiss
   - 本地部署 API 向量数据库：可本地部署且提供 API 接口，通过网络请求访问，例如 Milvus、Annoy、Weaviate
   - 云端 API 向量数据库：向量存云端，通过 API 管理，例如 TCVectorDB、Pinecone

3. LangChain VectorStore 抽象
   - 所有向量数据库（Faiss、Pinecone、Weaviate、TCVectorDB 等）统一实现
     langchain_core.vectorstores.VectorStore 接口
   - 一条向量数据库记录 = 向量(vector) + 元数据(metadata) + id；metadata 可承载
     文本原文、页码、归属文档 id、作者、创建时间等任意自定义信息
   - 构建时常用 from_texts / from_documents 两个通用方法，从文本或文档快捷导入，
     内部自动调用 embedding 把文本转成向量
   - 核心方法：
     · add_texts / add_documents            : 写入数据（内部自动调用 embedding）
     · similarity_search(query, k)          : 相似性检索，返回 List[Document]
     · similarity_search_with_score(...)    : 检索并返回 (Document, score) 元组
     · similarity_search_with_relevance_scores(...) : 返回 0~1 相关性得分
     · similarity_search_by_vector(vector)  : 直接用向量检索，跳过嵌入
     · max_marginal_relevance_search(...)   : MMR 检索，在相关性与多样性间平衡
     · as_retriever()                       : 转换为 Retriever 接入 LCEL 链
     · delete(ids)                          : 删除数据
   - 得益于这层抽象，切换向量数据库时上层 RAG 代码几乎无需修改；
     向量数据库没有复杂查询功能也没有事务，比传统数据库更易上手

4. Faiss 的持久化机制
   - 持久化后生成两个文件（必须成对存在）：
     · index.faiss : Faiss 原生二进制索引文件，存放全部向量数据
     · index.pkl   : Python pickle 文件，存放 docstore（原文与元数据）及 index_to_docstore_id 映射
   - 保存：db.save_local("./vector-store/")；加载：FAISS.load_local("./vector-store/", embedding, ...)
   - 仅有向量无法还原原文，仅有原文无法做检索，二者缺一不可
   - 持久化可避免每次使用都重建索引，极大提升效率

5. allow_dangerous_deserialization 参数（安全要点）
   - index.pkl 由 pickle 序列化，反序列化过程可执行任意代码
   - 加载来源不可信的 pkl 等同于执行陌生脚本，存在 RCE 风险
   - LangChain 从 0.1.x 起强制要求显式传入 allow_dangerous_deserialization=True，
     让开发者主动确认「我信任这个文件的来源」
   - 只有加载自己生成的索引时才应设为 True，绝不要用于外部下载的索引文件

6. similarity_search_with_score 的分数含义
   - Faiss 默认使用 L2 距离（欧几里得距离），返回的 score 是「距离」
   - 因此分数越小越相似，与「相似度越大越相似」的直觉相反，极易踩坑
   - 想获得 0~1 的归一化相似度应使用 similarity_search_with_relevance_scores()
   - 但 LangChain 封装的 Faiss 计算相关性得分有 bug：核心公式默认
       _euclidean_relevance_score_fn(distance) = 1.0 - distance / math.sqrt(2)
     该公式正确前提是向量只有 2 维且每个值范围 [0,1]；N 维下两点最大距离 √N，
     极易算出负数（如 -0.098）。可修正为 1.0 / (1.0 + distance)
   - 构造时还可用 distance_strategy 切到 MAX_INNER_PRODUCT 或 COSINE
   - 务必测试并校验嵌入模型生成向量的数值范围，避免明显错误

7. 带过滤的相似性搜索（Faiss 特性）
   - Faiss 原生并不支持元数据过滤，LangChain 封装做了处理：
     先取比 k 更多的 fetch_k（默认 20 条）做近邻搜索，再在结果上按 filter 过滤得到 k 条
   - filter 可传元数据字典，也可传函数（参数为 metadata，返回布尔值）
   - 因过滤发生在 Python 层，效率不如原生支持过滤的向量数据库

8. 数据操作能力与局限
   - 主流向量数据库普遍支持：新增、检索、带得分检索、带条件检索、删除，
     但几乎都不支持「修改」——索引结构需在插入后构建优化，允许修改会频繁重建、开销大，
     一般做法是删除后再新增
   - 适用：本地开发调试、单机脚本、离线批处理；数据量中等（百万级内）且变动不频繁；
     无需多进程/多实例共享
   - 局限：数据全驻留内存受单机内存限制；不支持多进程并发写入、无事务与一致性保障；
     元数据过滤能力弱（Python 层后置过滤，效率低）；删除只是标记，不真正回收索引空间

9. 最佳实践建议
   - 开发期用 Faiss 快速验证 RAG 流程，生产环境切换到 Pinecone/Weaviate 等服务化方案
   - 加载与写入必须使用与构建索引时完全相同的嵌入模型，否则检索结果无意义
   - vector-store 目录属运行时数据，建议加入 .gitignore
   - 大规模场景可手动构造 IVF/HNSW 索引（默认暴力检索 IndexFlatL2）提升速度
   - 入门可把向量数据库当成 Excel 表格：安装、写入、查找、删除、更新（先删后加）、保存，
     按办公软件流程学习即可

===================================================================================
"""
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建嵌入模型
# 关键约束：此处的模型必须与当初构建 ./vector-store/ 索引时使用的模型完全一致
# 原因：索引中存储的是 1536 维的 text-embedding-3-small 向量，
#      若换成其他模型，query 向量的维度或语义空间不匹配，
#      要么直接报维度错误，要么静默返回毫无意义的检索结果
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.从本地磁盘加载已持久化的 Faiss 索引
# FAISS.load_local() 参数说明：
#   folder_path="./vector-store/"
#       索引所在目录，需包含 index.faiss（向量数据）与 index.pkl（原文+元数据）
#   embeddings=embedding
#       检索时用于把 query 文本转换为向量的嵌入模型；
#       注意加载过程本身不会调用嵌入模型，它只是被保存在实例中供后续检索使用
#   allow_dangerous_deserialization=True
#       显式允许 pickle 反序列化。index.pkl 由 pickle 序列化而来，
#       反序列化可执行任意代码，故 LangChain 默认禁止，需开发者主动确认信任来源
#   index_name : 可选，默认 "index"，对应文件名前缀，可用于同目录存放多个索引
# 返回值：FAISS 实例（VectorStore 子类），索引数据此时已全量载入内存
db = FAISS.load_local("./vector-store/", embedding, allow_dangerous_deserialization=True)

# 3.执行带分数的相似性搜索
# similarity_search_with_score(query, k=4) 参数与返回：
#   query  : str，查询文本
#   k      : int，返回最相似的前 k 条，默认 4
#   filter : 可选，元数据过滤条件（Faiss 为 Python 层后置过滤，效率较低）
#   返回   : List[Tuple[Document, float]]
#            Document 含 page_content（原文）与 metadata（元数据）
#            float 为 L2 距离，数值越小表示越相似（注意不是相似度！）
# 执行流程：
#   a) 调用 embedding.embed_query(query) 把查询文本转为 1536 维向量
#   b) 在 Faiss 索引中做近邻搜索，得到 top-k 的内部索引下标与距离
#   c) 通过 index_to_docstore_id 映射找到对应的 docstore id
#   d) 从 docstore 取出原始 Document 对象，与距离配对后返回
print(db.similarity_search_with_score("我养了一只猫，叫笨笨"))
