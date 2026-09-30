#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 16:05
@Author  : thezehui@gmail.com
@File    : 1.TCVectorDB内置Embedding使用示例.py

===================================================================================
知识点讲解：TCVectorDB 内置 Embedding 模式
===================================================================================

1. 什么是 TCVectorDB
   - 腾讯云向量数据库（Tencent Cloud VectorDB）是一款全托管的自研企业级分布式向量数据库，
     专用于存储、检索、分析多维向量数据
   - 能力指标：支持多种索引类型与相似度计算方法，索引支持千亿级向量规模、
     百万级 QPS 及毫秒级查询延迟；认证账号可免费试用一个月
   - 与 Pinecone 设计理念非常接近，整体用法（建库、写入、检索）几乎一模一样，
     只是 filter、namespace 等概念的操作存在细微差异
   - 部署约束：默认只能在内网访问，生产环境尽量不暴露外网；
     开发阶段需配置并开启外网访问、获取 API 密钥，并在项目中导入对应环境变量
   - 依赖安装：pip install tcvectordb

2. 普通数据库 vs AI 数据库（TCVectorDB 的两个形态）
   - 普通向量数据库：只接收外部程序传递的数据，自身没有处理能力，但更可定制
   - AI 数据库：无需外部配置文本分割、Embedding、文档解析，底层全部由腾讯云实现
   - 本课程的 LangChain 集成走的是「普通数据库」形态：向量化由我们选择的
     Embedding 模型（内置或外部）完成，数据库只负责存向量与检索

3. TCVectorDB 的层级结构
   - Database（数据库）  : 顶层容器，相当于 MySQL 的 database
   - Collection（集合）  : 存放数据的表，创建时需确定向量维度与索引类型
   - Document（记录）    : 集合中的一条数据，含 id、vector、text 及自定义字段
   - 与 Pinecone 的 Index + Namespace 二级结构相比，TCVectorDB 的
     Database + Collection 更接近传统数据库的组织方式

4. 核心知识点：内置 Embedding 模式（本示例重点）
   - 传统模式（外部 Embedding）：客户端先调用嵌入模型把文本转成向量，再把向量发给数据库
   - 内置 Embedding 模式：客户端只发送「原始文本」，由 TCVectorDB 服务端
     自动调用其内置的嵌入模型完成向量化并写入
   - 触发方式：构造 TencentVectorDB 时传入 embedding=None
     此时 LangChain 判定使用服务端内置嵌入能力，不会在本地做任何向量计算

5. 内置 Embedding 的优势、代价与 Caveat
   优势：
     - 客户端无需配置 OpenAI/千帆等 API Key，架构简化
     - 省去一次外部 API 调用，网络往返减少，写入更快
     - 无需关心向量维度与模型一致性问题（服务端统一管理）
     - 不产生额外的嵌入模型调用费用（已含在数据库服务中）
     - 课程文档特别指出：目前 LangChain 封装的 TCVectorDB 使用内置 Embedding 没有 bug
   代价：
     - 只能使用腾讯云提供的模型（内置支持 5 种嵌入方式，如 bge-base-zh、m3e-base 等），模型选择受限
     - 与其他向量库的数据不可互通（向量由服务端生成，模型未必对外公开）
     - 检索时同样由服务端做 query 向量化，客户端无法干预（如加指令前缀）
     - 迁移数据库时，所有文本必须重新生成向量（无法复现服务端向量）

6. ConnectionParams 连接参数
   - url      : 向量数据库实例的访问地址（控制台获取，形如 http://lb-xxx.clb.ap-xx.tencentclb.com:10000）
   - username : 用户名，通常为 "root"
   - key      : API Key（控制台的密钥管理中获取）
   - timeout  : 请求超时时间（秒）
   - 这些敏感信息应统一放在 .env 中，通过 os.environ.get 读取，不要硬编码
   - 所需环境变量：TC_VECTOR_DB_URL / USERNAME / KEY / TIMEOUT（本例无需 OPENAI_API_KEY）

7. 自动建库建表能力
   - 与 Pinecone 必须先在控制台手动创建 Index 不同，
     LangChain 的 TencentVectorDB 在实例化时会检查 database/collection 是否存在，
     不存在则自动创建，因此本示例可以直接运行而无需前置控制台操作

8. 典型输出示例与观察点
   - 内置模式写入后执行 similarity_search_with_score("我养了一只猫，叫笨笨") 的返回：
       [('笨笨是一只很喜欢睡觉的猫咪', 0.874507),
        ('我的狗喜欢追逐球，看起来非常开心。', 0.757709),
        ('猫咪在窗台上打盹，看起来非常可爱。', 0.748665),
        ('昨晚我做了一个奇怪的梦，梦见自己在太空飞行。', 0.735823)]
   - 观察点 1：直接命中猫名「笨笨」的句子分数最高（0.87）
   - 观察点 2：「狗」与猫同属宠物语义域，分数（0.76）高于其它主题，体现中文嵌入的语义聚类
   - 观察点 3：返回的是数据库原生 cosine 相似度，越大越相关

9. 最佳实践建议
   - 纯中文、追求简化架构的项目，内置 Embedding 模式是最快的上手方案
   - 需要精确控制嵌入模型（如统一使用 OpenAI 向量、或需要跨库迁移）时，
     应改用外部 Embedding 模式（见本课第 2 个示例）
   - 内置与外部两种模式的 Collection 不可混用：向量来源不同，语义空间不一致
   - 生产环境应把 collection_name 与业务知识库 ID 绑定，实现天然隔离
   - 本示例每次运行都会执行 add_texts，反复执行会产生重复数据，调试时注意清理

===================================================================================
"""
import os

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams,
)

# 从 .env 文件加载环境变量
# 本示例需要的变量：TC_VECTOR_DB_URL、TC_VECTOR_DB_USERNAME、
#                  TC_VECTOR_DB_KEY、TC_VECTOR_DB_TIMEOUT
# 注意：本示例不需要 OPENAI_API_KEY，因为向量化完全由服务端完成
dotenv.load_dotenv()

# 1.创建 TCVectorDB 向量数据库实例（内置 Embedding 模式）
# TencentVectorDB 参数说明：
#   embedding=None
#       核心配置！传 None 表示启用服务端内置 Embedding 模式，
#       LangChain 不会在本地做任何向量化，只把原始文本发给数据库，
#       由 TCVectorDB 用其内置模型自动生成向量
#   connection_params=ConnectionParams(...)
#       连接配置对象，各字段含义：
#         url      -> 实例访问地址，从腾讯云控制台获取
#         username -> 登录用户名，一般为 root
#         key      -> API 密钥，用于鉴权
#         timeout  -> 请求超时秒数，注意环境变量读出的是字符串，需 int() 转换
#   database_name="llmops-test"
#       目标数据库名。若不存在，LangChain 会自动创建
#   collection_name="dataset-builtin"
#       目标集合（表）名。命名中的 builtin 用于标识这是内置 Embedding 模式的集合，
#       与外部 Embedding 模式的集合区分开，避免向量空间混用
# 执行副作用：实例化时会建立连接，并按需自动创建 database 与 collection
db = TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name="llmops-test",
    collection_name="dataset-builtin",
)

# 2.准备原始文本数据
# 主题刻意分散（猫咪、音乐、学习、食物、梦境、手机、阅读、野餐、狗），
# 便于观察服务端内置中文嵌入模型的语义区分能力
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

# 3.写入数据
# add_texts(texts) 参数与返回：
#   texts     : Iterable[str]，待写入的文本列表
#   metadatas : Optional[List[dict]]，元数据列表（本例未传）
#               注意：TCVectorDB 的 metadata 字段需通过构造时的 meta_fields 预先声明，
#               这与 Pinecone 的无模式 metadata 不同（详见本课第 3 个示例）
#   ids       : Optional[List[str]]，可指定记录 id，不传则自动生成
#   返回      : List[str]，写入成功的记录 id 列表
# 执行流程（内置 Embedding 模式的关键差异）：
#   a) 直接把原始文本组装成 Document 列表（本地不做 embed_documents 调用）
#   b) 通过 tcvectordb SDK 的 upsert 接口发送到服务端
#   c) 服务端用内置嵌入模型把 text 字段自动转为向量并建立索引
#   d) 返回记录 id 列表
# 对比外部模式：外部模式此处会先发起一次 OpenAI API 调用，内置模式完全省去
ids = db.add_texts(texts)
print("添加文档id列表:", ids)

# 4.执行归一化相关性检索
# similarity_search_with_relevance_scores(query, k=4) 参数与返回：
#   query  : str，查询文本
#   k      : int，返回条数，默认 4
#   返回   : List[Tuple[Document, float]]，score 归一化到 [0, 1]，越大越相关
# 执行流程（内置 Embedding 模式）：
#   a) 把 query 原文直接发给服务端（不在本地做 embed_query）
#   b) 服务端用同一个内置模型把 query 向量化
#   c) 在 collection 内做 ANN 检索，返回 top-k 结果与原生分数
#   d) LangChain 把原生分数归一化到 [0, 1] 后返回
# 预期结果：「笨笨是一只很喜欢睡觉的猫咪」应排第一（直接命中猫名「笨笨」），
#          「猫咪在窗台上打盹」排第二（同为猫主题），
#          「我的狗喜欢追逐球」可能排第三（同属宠物语义域）
print(db.similarity_search_with_relevance_scores("我养了一只猫，叫笨笨"))
