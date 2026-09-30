#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 23:06
@Author  : thezehui@gmail.com
@File    : 1.Pinecone向量数据库使用示例.py

===================================================================================
知识点讲解：Pinecone 云端向量数据库
===================================================================================

1. 什么是 Pinecone
   - Pinecone 是一款完全托管（Serverless）的云端向量数据库
   - 与 Faiss 的本质差别：Faiss 是嵌入进程的索引库，Pinecone 是独立的网络服务
   - 核心优势：免运维、自动扩缩容、支持高并发读写、数据持久化有保障、
     提供强大的 metadata 过滤能力
   - 依赖安装：pip install langchain-pinecone pinecone-client
   - Pinecone 是一个托管的、云原生的向量数据库，具有极简的 API，
     并且无需在本地部署即可快速使用
   - Pinecone 服务提供商还为每个账户设置了足够的免费空间，
     在开发阶段，可以快速基于 Pinecone 快速开发 AI 应用

2. Pinecone 核心概念（架构设计）
   - Organization（组织）：组织是使用相同结算方式的一个或多个项目的集合，
                           例如个人账号、公司账号等都算是一个组织
   - Project（项目）    : 项目是用来管理向量数据库、索引、硬件资源等内容的整合，
                           可以将不同的项目数据进行区分
   - Index（索引）      : 索引是 Pinecone 中数据的最高组织单位，相当于关系数据库中的
                           「数据库」，创建时必须固定 dimension（向量维度）与
                           metric（距离度量）。在索引中需要定义向量的存储维度、
                           查询时使用的相似性指标，支持两种类型的索引：
                           无服务器索引（根据数据大小自动扩容）和 Pod 索引（预设空间/硬件）
   - Namespace（命名空间）: Index 内部的逻辑分区，相当于「表」或「租户隔离单元」，
                           不同 namespace 的数据物理隔离，检索互不干扰，
                           是实现多租户 SaaS 的关键机制（如每个知识库一个 namespace）。
                           命名空间是索引内的分区，用于将索引中的数据区分成不同的组，
                           类似 Excel 中的 Sheet 表
   - Record（记录）     : 记录是数据的基本单位，由 id + values(向量) + metadata(元数据)
                           三部分组成。一条记录涵盖了 ID、向量(values)、元数据(metadata) 等
   - metric 可选值      : cosine（余弦，最常用）、euclidean（欧氏）、dotproduct（内积）

3. 环境配置（.env 文件）
   - PINECONE_API_KEY=你的 API Key
   - 在 https://app.pinecone.io 控制台创建项目后获取
   - 使用前需先在控制台手动创建 Index，并把 dimension 设为 1536
     （与 text-embedding-3-small 的输出维度一致），metric 设为 cosine
   - 可在注册好 Pinecone 后管理页面的 API Key 中设置
   - Pinecone 向量数据库默认只能在内网中链接使用，在生产环境中，也尽可能不将
     数据库暴露到外网中，但是在开发中，则需要配置并开启外网访问功能

4. PineconeVectorStore 组件
   - LangChain 对 Pinecone 的封装，实现标准 VectorStore 接口
   - 构造参数：
     · index_name : 目标 Index 名称，SDK 会自动连接
     · embedding  : 嵌入模型，用于写入与检索时把文本转为向量
     · namespace  : 默认命名空间，后续方法未显式传 namespace 时使用该值
     · text_key   : 原文存放在 metadata 中的键名，默认 "text"
   - 重要理解：Pinecone 本身只存向量与 metadata，不存「文档原文」概念，
     LangChain 是把 page_content 塞进 metadata[text_key] 来实现原文回取的
   - 在 Pinecone 中使用向量数据库，要确保 组织、项目、索引、命名空间、记录 等
     内容均配置好才可以使用

5. 三种相似性检索方法的分数差异（关键易混点）
   - similarity_search(query, k)
       返回 List[Document]，不含分数
   - similarity_search_with_score(query, k)
       返回 List[Tuple[Document, float]]，float 是「数据库原生分数」；
       metric=cosine 时是余弦相似度，越大越相似；
       metric=euclidean 时是距离，越小越相似 —— 含义随 metric 变化，不统一
   - similarity_search_with_relevance_scores(query, k)
       返回 List[Tuple[Document, float]]，float 被归一化到 [0, 1]，
       语义统一为「相关性得分，越大越相关」，跨数据库、跨 metric 可比
       => 业务代码中做阈值过滤（如 score > 0.7）应优先使用此方法

6. Pinecone 的 metadata 过滤能力
   - 和 Faiss 不同，Pinecone 支持原生的带过滤的相似性检索功能（元数据筛选）
   - 使用元数据过滤器会精确检索与过滤器匹配的最临近结果数
   - 过滤器格式为 json，其中键是元数据字段对应字符串，
     值是字符串、数字、布尔值、列表、json 中的一种
   - 支持复杂条件搜索，可以和 $and/$or/$in/$eq/$ne/$gt/$gte/$lt/$lte 等配合
   - 在无服务器索引的情况下，不支持通过元数据筛选器删除对应的数据，
     只在 Pod 索引下支持

7. 最佳实践建议
   - Index 的 dimension 创建后不可修改，更换嵌入模型需新建 Index
   - 善用 namespace 做业务隔离，避免所有数据挤在 default 命名空间
   - 写入（add_texts）是异步生效的，刚写完立即检索可能查不到，需等待几秒
   - Serverless 模式按读写单元与存储量计费，避免在循环中高频小批量写入
   - 生产环境写入建议分批（每批 100 条左右），并对失败批次做重试
   - 如果 LangChain 封装的 VectorStore 方法不能满足需求，可以获取原始实例，
     通过 .get_pinecone_index() 函数来获取对应的 Index，从而进行相应的操作

===================================================================================
"""
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 从 .env 文件加载环境变量（PINECONE_API_KEY、OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建嵌入模型
# text-embedding-3-small 输出 1536 维向量，
# 必须与 Pinecone 控制台创建 Index 时设置的 dimension 严格一致，否则写入会报维度错误
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.准备原始文本数据
# 这批语料故意设计成「主题分散」：包含猫咪、音乐、学习、食物、梦境、手机、
# 阅读、野餐、狗等多个互不相关的话题，便于直观观察语义检索的排序效果
texts: list = [
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

# 3.准备与文本一一对应的元数据
# metadata 的作用：
#   a) 检索时可用作过滤条件（见本课第 2 个示例的 filter 用法）
#   b) 检索结果中返回，供上层业务定位来源（如显示「引用自第 3 页」）
# 注意：metadatas 的长度必须与 texts 严格相等，且顺序一一对应
# 此处第 6 条额外带了 account_id 字段，说明 Pinecone 的 metadata 是「无模式」的，
# 不同记录可以拥有不同的字段集合，这与 TCVectorDB 需预先声明 meta_fields 形成对比
metadatas: list = [
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

# 4.创建 LangChain 向量数据库实例
# PineconeVectorStore 参数说明：
#   index_name="llmops"
#       Pinecone 控制台中已创建的 Index 名称，需提前手动建好（dimension=1536, metric=cosine）
#   embedding=embedding
#       嵌入模型，写入时用它做 embed_documents，检索时用它做 embed_query
#   namespace="dataset"
#       默认命名空间，本实例的读写都限定在这个逻辑分区内，实现数据隔离
# 执行副作用：构造时会用 PINECONE_API_KEY 建立与远端 Index 的连接
db = PineconeVectorStore(index_name="llmops", embedding=embedding, namespace="dataset")

# 5.写入数据（已注释，避免重复运行造成数据重复）
# add_texts(texts, metadatas, namespace) 说明：
#   参数：texts     -> Iterable[str]，待写入的文本列表
#         metadatas -> Optional[List[dict]]，与 texts 等长的元数据列表
#         ids       -> Optional[List[str]]，可指定记录 id；不传则自动生成 uuid4
#         namespace -> 覆盖构造时的默认命名空间
#   返回：List[str]，写入成功的记录 id 列表（后续删除/更新需要用到）
#   执行流程：
#     a) 调用 embedding.embed_documents(texts) 批量生成向量
#     b) 把 page_content 写入 metadata["text"]，与传入的 metadatas 合并
#     c) 组装为 (id, vector, metadata) 三元组，调用 Pinecone upsert 接口批量写入
#   注意：这里被注释掉是因为数据已写入过。Pinecone 的 upsert 以 id 为主键，
#        不传 ids 时每次都生成新 uuid，重复运行会不断追加重复内容
# db.add_texts(texts, metadatas, namespace="dataset")

# 6.执行归一化相关性检索
query = "我养了一只猫，叫笨笨"

# similarity_search_with_relevance_scores(query, k=4) 参数与返回：
#   query  : str，查询文本
#   k      : int，返回条数，默认 4
#   filter : Optional[dict]，metadata 过滤条件（下一个示例详细演示）
#   返回   : List[Tuple[Document, float]]
#            float 已归一化到 [0, 1]，统一语义为「越大越相关」
# 执行流程：
#   a) embed_query(query) 把查询转为 1536 维向量
#   b) 向 Pinecone 发起 query 请求，在 namespace="dataset" 内做 ANN 检索
#   c) 拿到原生 cosine 分数后，通过 _select_relevance_score_fn 转换为 0~1 区间
#   d) 从每条记录的 metadata 中弹出 text 字段作为 page_content，构造 Document
# 预期结果：「笨笨是一只很喜欢睡觉的猫咪」应排在首位（直接命中猫名），
#          「猫咪在窗台上打盹」次之（同主题），其余无关内容分数明显偏低
print(db.similarity_search_with_relevance_scores(query))
