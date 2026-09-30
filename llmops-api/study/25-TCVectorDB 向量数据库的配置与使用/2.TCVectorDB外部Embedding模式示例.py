#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 14:39
@Author  : thezehui@gmail.com
@File    : 2.TCVectorDB外部Embedding模式示例.py

===================================================================================
知识点讲解：TCVectorDB 外部 Embedding 模式
===================================================================================

1. 外部 Embedding 模式的定义
   - 与上一个示例（embedding=None，服务端内置向量化）相对
   - 本模式由「客户端」负责把文本转成向量，数据库只负责存储与检索向量
   - 触发方式：构造 TencentVectorDB 时传入一个真实的 Embeddings 实例
   - 维度决定：Collection 的向量维度由「首次写入的向量」决定并固定（本例 1536 维）

2. 两种模式的完整对比
   ┌──────────────┬────────────────────────┬──────────────────────────┐
   │              │ 内置 Embedding          │ 外部 Embedding            │
   ├──────────────┼────────────────────────┼──────────────────────────┤
   │ embedding 参数│ None                   │ OpenAIEmbeddings 等实例   │
   │ 向量化位置    │ 腾讯云服务端            │ 本地客户端                │
   │ 模型选择      │ 受限于平台提供的模型     │ 完全自由（任意 Embeddings）│
   │ API Key       │ 只需数据库密钥          │ 还需嵌入模型的 API Key    │
   │ 额外费用      │ 含在数据库服务内        │ 按嵌入模型 token 计费     │
   │ 网络往返      │ 1 次（客户端->数据库）   │ 2 次（先模型，再数据库）   │
   │ 跨库迁移      │ 困难（向量不可复现）     │ 容易（可重算相同向量）     │
   │ 可控性        │ 低                     │ 高（可裁剪维度、加前缀等） │
   └──────────────┴────────────────────────┴──────────────────────────┘

3. 何时选择外部 Embedding 模式
   - 需要与其他向量库（Faiss/Pinecone）保持向量一致，便于迁移或多库并行
   - 需要使用特定模型（如 bge-large-zh、自训练模型、私有化部署模型）
   - 需要复用 CacheBackedEmbeddings 做嵌入缓存以降本
   - 需要在向量化前后做自定义处理（如指令前缀、维度裁剪、归一化）

4. 关键约束：维度一致性
   - Collection 创建时的向量维度由「首次写入的向量」决定并固定
   - 本例使用 text-embedding-3-small（1536 维），因此 dataset-external 集合
     被固定为 1536 维；后续若换成 384 维模型写入同一集合，会直接报维度错误
   - 因此 collection_name 命名上区分 builtin / external 非常必要，
     两种模式的向量空间完全不同，混用会导致检索结果毫无意义

5. ⚠ 关键坑：外部 Embedding 必须声明 meta_fields 的 text 字段
   - 课程文档明确指出：LangChain 封装的 TencentVectorDB 存在一个 bug——
     当传递外部 Embedding 模型时，必须配置 meta_fields 属性且字段名为 "text"，
     并在 metadatas 中加上 "text" 字段，数据库才会记录原文信息
   - 若不这样做，写入时只存了 vector 而没存 text，检索时因找不到原文无法构建
     Document 文档，直接报错
   - bug 源码要点（add_texts）：
       if embeddings:
           doc_attrs["vector"] = embeddings[id]   # 外部模式：只存向量
       else:
           doc_attrs["text"] = texts[id]           # 内置模式：存原文
   - 修复写法：
       meta_fields=[MetaField(name="text", data_type=META_FIELD_TYPE_STRING)]
       metadatas = [{"text": text, "page": i} for i, text in enumerate(texts)]
   - 注：本示例未启用 metadatas 故未触发该问题，但生产使用外部模式务必加 text 字段声明

6. meta_fields 元数据字段声明（TCVectorDB 的 metadata 是有模式的）
   - 需要用于过滤的字段必须在构造时通过 meta_fields 预先声明类型，
     声明后服务端才会为该字段建立标量索引，从而支持 expr 表达式过滤
   - 字段类型常量：
     · META_FIELD_TYPE_STRING  -> 字符串
     · META_FIELD_TYPE_UINT64  -> 无符号整数
     · META_FIELD_TYPE_ARRAY   -> 数组
   - 未声明的字段虽可随数据一起存储，但无法参与过滤条件（详见本课第 3 个示例）

7. similarity_search_with_score 与 relevance_scores 的差异
   - 本例使用 similarity_search_with_score，返回「数据库原生分数」
     TCVectorDB 默认使用 cosine 度量，原生分数即余弦相似度，越大越相似
   - 上一个示例使用 similarity_search_with_relevance_scores，
     返回归一化到 [0,1] 的相关性得分，跨数据库语义统一
   - 业务代码做阈值判断时优先用 relevance_scores，调试观察原始数值时用 with_score

8. 最佳实践建议
   - 外部模式下务必用 CacheBackedEmbeddings 包装嵌入模型，显著降低 API 费用
   - 把 database_name 也放进环境变量（本例已这样做），便于多环境切换
   - 反复运行本脚本会重复写入相同文本，调试阶段建议显式传 ids 以实现 upsert 覆盖
   - 生产环境应为写入操作加重试，应对网络抖动与偶发限流

===================================================================================
"""
import os

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams
)
from langchain_openai import OpenAIEmbeddings

# 从 .env 文件加载环境变量
# 本示例需要的变量比内置模式更多：
#   数据库相关：TC_VECTOR_DB_URL / USERNAME / KEY / TIMEOUT / DATABASE
#   嵌入模型相关：OPENAI_API_KEY、OPENAI_API_BASE（内置模式不需要）
dotenv.load_dotenv()

# 1.创建外部嵌入模型
# 这是本示例与内置模式最核心的差异：客户端持有真实的嵌入模型实例
# text-embedding-3-small 输出 1536 维向量，该维度将决定 collection 的向量维度
# 进阶建议：生产环境可用 CacheBackedEmbeddings 包装此对象以缓存嵌入结果
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.创建 TCVectorDB 向量数据库实例（外部 Embedding 模式）
# TencentVectorDB 参数说明：
#   embedding=embedding
#       核心配置！传入真实的 Embeddings 实例，
#       LangChain 会在写入前调用 embed_documents、检索前调用 embed_query，
#       数据库只接收现成的向量，不做任何向量化工作
#   connection_params=ConnectionParams(...)
#       url      -> 实例访问地址
#       username -> 登录用户名
#       key      -> API 密钥
#       timeout  -> 超时秒数（环境变量为字符串，需 int() 转换）
#   database_name=os.environ.get("TC_VECTOR_DB_DATABASE")
#       数据库名从环境变量读取，便于 dev/test/prod 多环境切换
#       （对比内置模式示例中的硬编码 "llmops-test"，这里是更好的实践）
#   collection_name="dataset-external"
#       集合名带 external 后缀，用于与内置模式的 dataset-builtin 集合区分。
#       必须区分的原因：两者向量由不同模型生成，语义空间不兼容，混用即失效
db = TencentVectorDB(
    embedding=embedding,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset-external",
    # 进阶用法：预先声明可过滤的 metadata 字段（TCVectorDB 的 metadata 是有模式的）
    # 只有在此声明过的字段，服务端才会建立标量索引，才能在检索时用 expr 参数过滤
    # 未声明的字段可以存储但无法参与过滤条件
    # meta_fields=[
    #     MetaField(name="text", data_type=META_FIELD_TYPE_STRING),
    # ]
)

# 3.准备原始文本数据（与内置模式示例相同，便于横向对比两种模式的检索效果）
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

# 元数据示例（已注释）：用推导式为每条文本生成 {"text": 原文, "page": 序号}
# 若要启用，必须先在上面的构造参数中通过 meta_fields 声明 text 与 page 字段类型，
# 否则字段虽能写入但无法作为过滤条件使用
# metadatas = [{"text": text, "page": index} for index, text in enumerate(texts)]

# 4.写入数据
# add_texts(texts) 参数与返回：
#   texts     : Iterable[str]，待写入文本列表
#   metadatas : Optional[List[dict]]，元数据（本例未传）
#   ids       : Optional[List[str]]，可指定 id 实现 upsert 覆盖
#   返回      : List[str]，写入成功的记录 id 列表
# 执行流程（外部模式，注意与内置模式的差异）：
#   a) 调用 embedding.embed_documents(texts) —— 发起一次 OpenAI API 请求，
#      把 10 条文本批量转为 10 个 1536 维向量【内置模式没有这一步】
#   b) 把 (id, vector, text, metadata) 组装成 tcvectordb 的 Document 对象
#   c) 调用 SDK 的 upsert 接口写入服务端
#   d) 服务端直接存储收到的向量（不做任何向量化）
#   e) 返回记录 id 列表
ids = db.add_texts(texts)
print("添加文档id列表:", ids)

# 5.执行带原生分数的相似性检索
# similarity_search_with_score(query, k=4) 参数与返回：
#   query  : str，查询文本
#   k      : int，返回条数，默认 4
#   expr   : Optional[str]，TCVectorDB 的过滤表达式（需配合 meta_fields 声明）
#   返回   : List[Tuple[Document, float]]
#            float 为数据库原生分数，TCVectorDB 默认 cosine 度量 => 越大越相似
# 执行流程（外部模式）：
#   a) 调用 embedding.embed_query(query) 在本地把查询转为 1536 维向量
#      【内置模式是把原文发给服务端，由服务端向量化】
#   b) 携带该向量向服务端发起 search 请求
#   c) 服务端做 ANN 检索，返回 top-k 记录与原生相似度分数
#   d) LangChain 把结果转换为 (Document, score) 元组列表
# 预期结果：「笨笨是一只很喜欢睡觉的猫咪」分数最高，「猫咪在窗台上打盹」次之
print(db.similarity_search_with_score("我养了一只猫，叫笨笨"))
