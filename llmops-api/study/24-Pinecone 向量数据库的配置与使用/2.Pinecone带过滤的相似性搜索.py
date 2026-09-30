#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 8:12
@Author  : thezehui@gmail.com
@File    : 2.Pinecone带过滤的相似性搜索.py

===================================================================================
知识点讲解：Pinecone 带 metadata 过滤的相似性搜索
===================================================================================

1. 为什么需要元数据过滤
   - 纯向量检索只考虑「语义相似」，无法表达业务约束
   - 真实 RAG 场景几乎都需要附加条件，典型需求：
     · 多租户隔离：只能检索当前用户 account_id 下的文档
     · 权限控制：过滤掉无权访问的私有文档
     · 范围限定：只在指定知识库 / 指定时间段 / 指定页码范围内检索
     · 类型筛选：只检索 PDF 类型或已审核通过的文档
   - 元数据过滤 + 向量检索的组合，是构建生产级 RAG 的必备能力

2. Pre-filtering vs Post-filtering（重要原理）
   - Post-filtering（后置过滤）：先取 top-k 向量，再用条件筛掉不合格的
       缺陷：若 top-k 全被过滤掉，最终结果可能为空，召回率不可控
       Faiss / Chroma 的部分实现属于此类
   - Pre-filtering（前置过滤）：先按条件缩小候选集，再在候选集内做向量检索
       优势：保证能返回 k 条满足条件的结果，召回质量稳定
       Pinecone 采用的正是 pre-filtering，元数据索引与向量索引协同工作
   - 这是 Pinecone 相比 Faiss 在生产场景的核心优势之一

3. Pinecone 过滤语法（类 MongoDB 风格）
   比较操作符：
     · $eq  等于        {"page": {"$eq": 5}} 或简写 {"page": 5}
     · $ne  不等于      {"page": {"$ne": 5}}
     · $gt  大于        {"page": {"$gt": 5}}
     · $gte 大于等于    {"page": {"$gte": 5}}
     · $lt  小于        {"page": {"$lt": 5}}
     · $lte 小于等于    {"page": {"$lte": 5}}
     · $in  在列表中    {"page": {"$in": [1, 3, 5]}}
     · $nin 不在列表中  {"page": {"$nin": [1, 3, 5]}}
     · $exists 字段存在 {"account_id": {"$exists": True}}
   逻辑操作符：
     · $and 全部满足    {"$and": [{"page": {"$gte": 5}}, {"page": {"$lte": 8}}]}
     · $or  任一满足    {"$or": [{"page": 5}, {"account_id": 1}]}
   隐式 AND：
     · 同一层级写多个字段等价于 $and，如 {"page": 5, "account_id": 1}

4. metadata 字段的类型限制
   - Pinecone 的 metadata 只支持 string、number、boolean、list[string] 四种类型
   - 不支持嵌套对象（dict），需要层级信息时应把 key 扁平化（如 "doc.author" -> "doc_author")
   - 单条记录的 metadata 总大小上限为 40KB
   - 字段无需预先声明（无模式），但用于过滤的字段建议在所有记录上保持一致存在，
     否则缺失该字段的记录会被静默排除在过滤结果之外

5. 最佳实践建议
   - 为避免 $exists 带来的不确定性，多租户场景应给每条记录都写入 account_id
   - 过滤条件不要过度收窄，否则候选集不足 k 条会导致返回结果偏少
   - 高选择性（能筛掉大部分数据）的字段放在过滤条件中最有价值
   - metadata 只存检索与展示必需的字段，大段文本不要塞进去（有 40KB 上限）
   - 复杂动态过滤条件可交给 SelfQueryRetriever 由 LLM 自动生成（见第 48 课）

===================================================================================
"""
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 从 .env 文件加载环境变量（PINECONE_API_KEY、OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建嵌入模型，维度需与 Pinecone Index 的 dimension 一致（1536）
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.准备原始文本数据（与上一个示例相同，便于对比过滤前后的差异）
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

# 3.准备元数据，注意这里的设计意图
# page 字段：1~10，模拟文档页码，用于演示数值范围过滤（$gte / $lte 等）
# account_id 字段：只有第 6 条记录（page=6）拥有该字段，
#   用于演示「字段存在性」带来的过滤效果 —— 用 account_id 作为条件时，
#   其余 9 条因缺失该字段会被直接排除，这正是多租户隔离的实现基础
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

# 4.创建向量数据库实例
#   index_name="llmops"  : 目标 Index
#   embedding=embedding  : 用于 query 文本向量化
#   namespace="dataset"  : 限定读写的逻辑分区
db = PineconeVectorStore(index_name="llmops", embedding=embedding, namespace="dataset")

# 5.写入数据（已注释，数据已在上一个示例中写入，重复执行会产生重复记录）
# db.add_texts(texts, metadatas, namespace="dataset")

# 6.执行带过滤条件的相似性检索
query = "我养了一只猫，叫笨笨"

# similarity_search_with_relevance_scores(query, filter=...) 参数说明：
#   query  : str，查询文本，会被向量化后做语义匹配
#   k      : int，返回条数，默认 4
#   filter : dict，Pinecone 元数据过滤表达式（类 MongoDB 语法）
#   返回   : List[Tuple[Document, float]]，score 归一化到 [0, 1]，越大越相关
#
# 本例过滤条件解析：
#   {"$or": [{"page": 5}, {"account_id": 1}]}
#   含义：page 等于 5 「或」 account_id 等于 1
#   命中记录：
#     · page=5 -> "我最喜欢的食物是意大利面..."
#     · account_id=1 -> page=6 的 "昨晚我做了一个奇怪的梦..."
#   => 候选集仅剩这 2 条，其余 8 条（含最相关的「笨笨是一只很喜欢睡觉的猫咪」）
#      在向量检索之前就已被排除
#
# 关键观察点：
#   虽然 query 问的是猫，但由于 pre-filtering 先把候选集锁定为「意大利面」和
#   「太空飞行的梦」这两条，最终返回的必然是这 2 条语义不相关的内容，且分数很低。
#   这直观说明：过滤条件的优先级高于语义相似度，条件设置过窄会严重损害召回质量。
print(db.similarity_search_with_relevance_scores(
    query,
    filter={"$or": [{"page": 5}, {"account_id": 1}]}
    # 备选写法：数值范围过滤，只检索 page >= 5 的记录（命中 page 5~10 共 6 条）
    # 这种范围过滤在「只检索最近 N 天文档」「只检索某章节」等场景非常实用
    # filter={"page": {"$gte": 5}}
))
