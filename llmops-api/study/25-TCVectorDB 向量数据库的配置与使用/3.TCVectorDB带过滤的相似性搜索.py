#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 16:30
@Author  : thezehui@gmail.com
@File    : 3.TCVectorDB带过滤的相似性搜索.py

===================================================================================
知识点讲解：TCVectorDB 带元数据过滤的相似性搜索
===================================================================================

1. TCVectorDB 与 Pinecone 元数据过滤的本质差异
   - Pinecone：无模式（schema-less）metadata，任意字段随写随用，
     过滤语法为类 MongoDB 的 JSON 字典（如 {"page": {"$gte": 5}}）
   - TCVectorDB：有模式（schema-based）metadata，必须预先声明字段类型，
     过滤语法为类 SQL 的字符串表达式（如 "page>=9"）
   - 这是使用 TCVectorDB 时最容易踩的坑：不声明 meta_fields 就无法过滤，
     而且往往不会报错，只是过滤条件被静默忽略或直接抛出字段不存在的异常

2. meta_fields 字段声明机制（本示例核心）
   - 作用：告诉 TCVectorDB 哪些标量字段需要建立索引以支持过滤
   - 用法：meta_fields=[MetaField(name="字段名", data_type=类型常量)]
   - 支持的类型常量（来自 langchain_community.vectorstores.tencentvectordb）：
     · META_FIELD_TYPE_STRING : 字符串，支持等值匹配与 in
     · META_FIELD_TYPE_UINT64 : 无符号 64 位整数，支持数值比较与范围查询
     · META_FIELD_TYPE_ARRAY  : 数组，支持包含关系判断
   - 课程文档强调：用非向量字段检索时，必须在构建 Collection 时添加对应索引，否则无法检索会报错
   - 重要限制：meta_fields 在 Collection 创建时生效并固化，
     集合建好后再改声明不会自动生效，需要新建集合

3. expr 过滤表达式语法（类 SQL）
   - 表达式格式：<field_name><operator><value>，多个表达式用 and / or / not 连接，可用括号分组
   - string 运算符：=、!=、in、not in（value 必须用英文双引号括起，如 doc_type="pdf"）
   - uint64 运算符：>、>=、=、<、<=、!=
   - 示例："page>=9"、"page in [1,3,5]"、"page=1 or page=10"、"(page>=5 and page<=8) or page=10"
   - 与 Pinecone 的 JSON 字典语法完全不同，迁移时这部分代码必须重写

4. 本示例的一个关键细节：embedding=None
   - 注意代码第 21 行创建了 OpenAIEmbeddings 实例，但第 23 行传的是 embedding=None
   - 这意味着本示例实际运行在「内置 Embedding 模式」下，
     上面创建的 embedding 变量并未被使用（属于示例代码的遗留）
   - 影响：向量化由腾讯云服务端完成，客户端不消耗 OpenAI 配额
   - 这也说明 meta_fields 过滤能力与 Embedding 模式是两个正交的维度，
     内置模式与外部模式都能使用元数据过滤

5. 元数据未声明字段的行为（本示例可观察到）
   - metadatas 中第 6 条额外带了 account_id 字段，但 meta_fields 只声明了 page
   - 结果：account_id 可以随数据写入存储，但无法出现在 expr 过滤条件中，
     若尝试 expr="account_id = 1" 会因字段无索引而失败
   - 对比 Pinecone：同样的数据在 Pinecone 中可以直接用 account_id 过滤

6. 典型输出示例与观察点
   - 执行 similarity_search_with_score("我养了一只猫，叫笨笨", expr="page>=9") 的返回：
       [('我的狗喜欢追逐球，看起来非常开心。', {'page': 10}, 0.757709),
        ('他们一起计划了一次周末的野餐，希望天气能好。', {'page': 9}, 0.675508)]
   - 观察点 1：query 问的是猫（「我养了一只猫，叫笨笨」），但语义最匹配的
     「笨笨是一只很喜欢睡觉的猫咪」（page=1）被过滤条件直接排除
   - 观察点 2：候选集只有 2 条，即使 k 默认为 4，最终也最多只能返回 2 条结果
   - 观察点 3：page=10 的「狗」排在 page=9 的「野餐」之前，因狗与猫同属宠物语义域
   - 结论：元数据过滤优先级高于语义相似度，条件过窄会严重损害召回

7. 最佳实践建议
   - 设计 Collection 时就把所有可能用于过滤的字段一次性声明完整，
     因为后期新增字段需要重建集合并重新灌数据，成本极高
   - 多租户隔离字段（account_id、tenant_id）必须声明为 UINT64 或 STRING 并建索引
   - 过滤表达式是字符串拼接，存在注入风险，
     用户输入必须做类型校验与转义，不要直接拼接到 expr 中
   - 与 Pinecone 一样，过滤条件过窄会导致候选集不足 k 条（本例 expr="page>=9" 仅 2 条）
   - 需要跨向量库通用的过滤逻辑时，考虑在应用层抽象一个过滤条件中间表示，
     再按目标数据库分别翻译为 JSON dict 或 SQL 表达式

===================================================================================
"""
import os

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams,
    MetaField,
    META_FIELD_TYPE_UINT64,
)
from langchain_openai import OpenAIEmbeddings

# 从 .env 文件加载环境变量
# 数据库相关：TC_VECTOR_DB_URL / USERNAME / KEY / TIMEOUT / DATABASE
dotenv.load_dotenv()

# 1.创建嵌入模型实例
# 注意：本示例下方传入的是 embedding=None（内置 Embedding 模式），
#      因此这里创建的 embedding 变量实际上并未被使用。
#      若想改为外部 Embedding 模式，把下面的 embedding=None 改为 embedding=embedding 即可，
#      但需注意此时 collection 的向量维度会变为 1536，与内置模式的集合不可混用
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.创建带元数据字段声明的向量数据库实例
# TencentVectorDB 参数说明：
#   embedding=None
#       使用服务端内置 Embedding 模式，客户端只发送原文，由腾讯云完成向量化
#   connection_params=ConnectionParams(...)
#       url      -> 实例访问地址
#       username -> 登录用户名
#       key      -> API 密钥
#       timeout  -> 超时秒数（环境变量是字符串，需 int() 转换）
#   database_name=os.environ.get("TC_VECTOR_DB_DATABASE")
#       数据库名，从环境变量读取
#   collection_name="dataset-filter"
#       独立的集合名，用于演示过滤功能，与 builtin / external 两个集合隔离
#   meta_fields=[MetaField(name="page", data_type=META_FIELD_TYPE_UINT64)]
#       本示例的核心配置！声明 page 字段为无符号整数类型并建立标量索引，
#       只有经过声明的字段才能出现在后续 similarity_search 的 expr 过滤表达式中。
#       MetaField 参数：
#         name      -> 元数据字段名，需与 metadatas 中的 key 完全一致
#         data_type -> 字段类型常量，决定支持的运算符
#                      UINT64 支持数值比较（> >= < <= = != in），
#                      STRING 支持等值与 in
#       注意：此声明在 Collection 首次创建时固化，后期修改需重建集合
db = TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset-filter",
    meta_fields=[
        MetaField(name="page", data_type=META_FIELD_TYPE_UINT64),
    ]
)

# 3.准备原始文本数据
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

# 4.准备元数据，与 texts 一一对应
# page 字段：1~10，已在 meta_fields 中声明为 UINT64，因此可用于 expr 数值过滤
# account_id 字段：仅第 6 条拥有，但【未在 meta_fields 中声明】，
#   因此它可以随数据写入存储，却无法出现在 expr 过滤条件里。
#   这正是 TCVectorDB（有模式）与 Pinecone（无模式）的关键差异体现：
#   同样的数据结构在 Pinecone 中可直接 filter={"account_id": 1}，在这里则不行
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

# 5.写入数据（文本 + 元数据）
# add_texts(texts, metadatas) 参数与返回：
#   texts     : Iterable[str]，待写入文本列表
#   metadatas : List[dict]，元数据列表，长度必须与 texts 严格相等且顺序对应
#   返回      : List[str]，写入成功的记录 id 列表
# 执行流程（内置 Embedding 模式 + 元数据）：
#   a) 把 text 与对应 metadata 合并组装为 tcvectordb 的 Document
#   b) 调用 SDK upsert 写入服务端（本地不做向量化）
#   c) 服务端用内置模型生成向量，同时为已声明的 page 字段建立标量索引
#   d) 返回记录 id 列表
ids = db.add_texts(texts, metadatas)
print("添加文档id列表:", ids)

# 6.执行带 expr 过滤条件的相似性检索
# similarity_search_with_score(query, k=4, expr=...) 参数与返回：
#   query  : str，查询文本
#   k      : int，返回条数，默认 4
#   expr   : str，TCVectorDB 的类 SQL 过滤表达式，字段必须已在 meta_fields 声明
#   返回   : List[Tuple[Document, float]]，float 为原生 cosine 相似度，越大越相似
#
# 本例过滤条件解析：
#   expr="page>=9"  ->  只保留 page 大于等于 9 的记录
#   命中记录仅 2 条：
#     · page=9  -> "他们一起计划了一次周末的野餐，希望天气能好。"
#     · page=10 -> "我的狗喜欢追逐球，看起来非常开心。"
#
# 关键观察点：
#   1) query 问的是猫（「我养了一只猫，叫笨笨」），但语义最匹配的
#      「笨笨是一只很喜欢睡觉的猫咪」（page=1）被过滤条件直接排除
#   2) 由于候选集只有 2 条，即使 k 默认为 4，最终也最多只能返回 2 条结果
#   3) 返回结果中 page=10 的「狗」大概率排在 page=9 的「野餐」之前，
#      因为狗与猫同属宠物语义域，相似度更高
#   => 再次印证：元数据过滤的优先级高于语义相似度，条件过窄会严重损害召回
print(db.similarity_search_with_score("我养了一只猫，叫笨笨", expr="page>=9"))
