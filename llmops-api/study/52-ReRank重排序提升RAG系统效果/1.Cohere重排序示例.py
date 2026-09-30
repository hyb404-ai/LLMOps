#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/7 19:28
@Author  : thezehui@gmail.com
@File    : 1.Cohere重排序示例.py

===================================================================================
知识点讲解：ReRank 重排序提升 RAG 效果（Cohere）
===================================================================================

1. ReRank 重排序的定位与价值
   - 筛选阶段的两类优化：重排序、纠正性 RAG；其中重排序使用频率最高、性价比最高，通常与混合检索搭配，是 Dify、Coze、智谱及大部分开源 Agent 项目的主流选择
   - 核心思想：对检索到的文档「调整顺序」，并额外增加「剔除无关/多余数据」的步骤
   - 逻辑：输入文档列表、输出仍是文档列表，与 DocumentTransformer 类似；但在 LangChain 中它是一个 DocumentCompressor（压缩）组件

2. 两阶段检索：粗排 + 精排
   - 第一阶段（粗排）：向量检索快速召回候选集（如 MMR、Top 20-50），追求高召回
   - 第二阶段（精排）：重排序模型评估查询与每个候选文档的相关性，重新排序并剔除非相关项，得到 Top 5-10
   - 这是「用快而糙的检索换召回，用准而慢的重排换精度」的分工

3. ContextualCompressionRetriever 的组合方式
   - 重排序可单独使用，也可通过 ContextualCompressionRetriever 把「检索器 + 压缩器」二次包装
   - base_retriever=db.as_retriever()：第一阶段粗排；base_compressor=rerank：第二阶段精排
   - 包装后 retriever.invoke(query) 直接返回已重排的文档列表，下游无需改动

4. Cohere Rerank 模型选型
   - 可用的重排模型不多：Cohere 在线模型（API 访问，付费，注册有免费额度，海外部署国内访问偏慢）体验最佳
   - 开源替代：bge-rerank-base / bge-rerank-large（企业内部部署可选，bge-rerank-large 评分最好）
   - 安装依赖：pip install langchain-cohere
   - 多语言：rerank-multilingual-v3.0 支持中文；rerank-english-v3.0 英文专用

5. CohereRerank 关键参数
   - model：重排模型名（如 rerank-multilingual-v3.0）
   - 模型会为每个（查询，文档）对计算相关性分数，并写入 metadata.relevance_score
   - top_n：最终返回条数，可在压缩器层控制（默认跟随基础检索器召回量）

6. 典型输出示例与观察点
   - 对「关于LLMOps应用配置的信息有哪些」检索，结果每条 metadata 带 relevance_score（如 0.9298、0.7358、0.0988）
   - 观察点：原始检索召回 4 条，经重排后只返回相关性高的 3 条（可设置），一条被剔除
   - 这正是「压缩器」与「文档转换器」的区别：压缩器会主动丢弃低相关内容（旧版 LangChain 两者无区别，故很多文档转换器官方文档里能看到压缩器）

7. 技术优势
   - 排序质量高：Rerank 模型专为排序任务训练，深度理解查询-文档交互关系
   - 易于集成：无需修改现有检索器，包裹一层即可生效
   - 与多语言、混合检索天然契合，是 RAG 性价比最高的后处理优化

8. 最佳实践
   - 先用快速向量检索召回较大候选集（Top 20-50），再用 Rerank 精排到 Top 5-10
   - 基础检索器建议配 MMR 等多样化策略提供候选
   - 根据语料语言选择模型（中文优先 rerank-multilingual-v3.0）
   - 重排会引入一次额外模型调用，注意延迟与配额；与上一节 Multi-Query 的膨胀文档配合使用效果最佳

===================================================================================
"""
import dotenv
import weaviate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载环境变量配置（需要 COHERE_API_KEY）
dotenv.load_dotenv()

# 1.创建向量数据库与重排组件

# OpenAIEmbeddings：创建嵌入模型实例
# 用于将文档和查询转换为向量
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# WeaviateVectorStore：创建 Weaviate 向量数据库实例
# 这是基础检索器，负责第一阶段的粗排
db = WeaviateVectorStore(
    # connect_to_wcs：连接到 Weaviate Cloud Services
    client=weaviate.connect_to_wcs(
        cluster_url="https://mbakeruerziae6psyex7ng.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("ZltPVa9ZSOxUcfafelsggGyyH6tnTYQYJvBx"),
    ),
    # index_name：索引名称
    index_name="DatasetDemo",
    # text_key：文档内容字段名
    text_key="text",
    # embedding：嵌入模型
    embedding=embedding,
)

# CohereRerank：创建 Cohere 重排序模型实例
# - model：指定重排序模型
#   - rerank-multilingual-v3.0：多语言重排序模型，支持中文
#   - rerank-english-v3.0：英文专用模型
# 重排序模型会为每个（查询，文档）对计算相关性分数
rerank = CohereRerank(model="rerank-multilingual-v3.0")

# 2.构建压缩检索器
# ContextualCompressionRetriever：上下文压缩检索器
# 它包装了基础检索器，并添加了文档压缩/重排序功能
retriever = ContextualCompressionRetriever(
    # base_retriever：基础检索器（第一阶段粗排）
    # - search_type="mmr"：使用最大边际相关性算法
    #   MMR 在相似度和多样性之间取得平衡，避免返回内容重复的文档
    base_retriever=db.as_retriever(search_type="mmr"),
    # base_compressor：文档压缩器（第二阶段精排）
    # 这里使用 CohereRerank 作为压缩器，实际上是重排序器
    base_compressor=rerank,
)

# 3.执行搜索并排序
# invoke：执行两阶段检索
# 执行流程：
# 1. base_retriever 使用 MMR 算法召回相关文档（如 Top 20）
# 2. base_compressor (Rerank) 计算查询与每个文档的相关性分数
# 3. 根据相关性分数重新排序
# 4. 返回重排后的文档列表（默认返回 Top 4）
search_docs = retriever.invoke("关于LLMOps应用配置的信息有哪些呢？")

# 输出结果
print(search_docs)
print(len(search_docs))

# 效果对比：
# 不使用 Rerank：db.as_retriever().invoke(query)
# 使用 Rerank：retriever.invoke(query)
#
# Rerank 提升效果：
# - 更精确的排序：相关文档排在前面
# - 更好的语义理解：理解查询意图和文档内容的匹配
# - 更高的用户满意度：用户更容易找到需要的信息
#
# 性能考虑：
# - Rerank 需要额外的 API 调用，增加延迟
# - 适合对准确性要求高的场景
# - 可以通过控制候选集大小平衡速度和质量
