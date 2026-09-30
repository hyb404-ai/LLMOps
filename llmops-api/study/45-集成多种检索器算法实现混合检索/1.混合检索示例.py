#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 10:17
@Author  : thezehui@gmail.com
@File    : 7.混合检索示例.py

===================================================================================
知识点讲解：EnsembleRetriever 集成多种检索算法实现混合检索
===================================================================================

1. 稀疏检索与稠密检索的互补关系
   - 稀疏检索（BM25 为代表）：基于词频统计，向量维度=词表大小且极稀疏；强项是精确术语匹配（产品型号、错误码、人名、API 名）；弱项是不懂语义
   - 稠密检索（Embedding 向量检索）：基于神经网络语义表示，向量维度固定（如 1536）且稠密；强项是理解语义与同义改写；弱项是对罕见专有名词泛化不足
   - 两者失效场景几乎正交，混合使用可显著提升整体召回率，即业界公认的 Hybrid Search

2. BM25 算法要点
   - TF-IDF 的改进版，传统信息检索事实标准
   - 词频饱和：某词出现 10 次与 100 次得分差距有限，避免关键词堆砌
   - 文档长度归一化：惩罚长文档以保证公平
   - BM25Retriever 纯本地内存计算，零 API 成本、毫秒级延迟

3. EnsembleRetriever 的融合机制（RRF）
   - 接受检索器列表 retrievers 与权重 weights，并行调用各子检索器的 get_relevant_documents() 得到有序结果
   - 用带权 RRF 融合：score(d) = Σ weight_i * 1 / (k + rank_i(d))，k 默认 60
   - 为何用 RRF 而非直接加权得分：BM25 得分域 [0,+∞)、余弦相似度 [-1,1]，量纲不同无法直接相加；RRF 只用"排名"天然解决跨系统量纲问题
   - 多路都出现的文档累加得分，形成"多路共识"加权

4. weights 权重调优
   - 列表长度必须与 retrievers 一致、一一对应，内部自动归一化
   - 技术文档 / 代码库 / 含大量专有名词 → 提高 BM25 权重（如 [0.6, 0.4]）
   - 口语化提问 / 客服对话 / 语义泛化强 → 提高向量权重（如 [0.3, 0.7]）
   - 无先验从 [0.5, 0.5] 起步，按评测集调整

5. 本例的经典示例设计
   - 查询："除了猫，你养了什么宠物呢？"
   - BM25 路：精确命中含"猫"文档（page=1、3），但无法理解"宠物"上位概念，命中不了"狗"
   - 向量路：理解"宠物"语义，召回"狗"（page=10）与"猫"相关文档
   - 融合后：字面精确 + 语义泛化，完整覆盖用户真实意图，直观展示 1+1>2

6. EnsembleRetriever 的扩展能力与生态
   - retrievers 可放任意数量、任意类型 BaseRetriever，也能混入自定义检索器、MultiQueryRetriever、HyDERetriever、ES / 图数据库检索器
   - 混合检索被 Dify、Coze、智谱 等 AI 开发平台广泛采用，也是本课程 LLMOps 项目使用的检索策略
   - 运行时还可通过 configurable_fields 动态控制某路检索的 k 值与权重

7. 生产实践：多路召回 → RRF 融合 → ReRank 精排
   - 这是构建生产级 RAG 召回层的标准做法
   - 各路 k 值建议设为最终需要条数的 2~3 倍，给 RRF 留足融合空间
   - 混合检索之后再接 Cross-Encoder ReRank 可进一步提升精排质量

===================================================================================
最佳实践建议
===================================================================================

- 生产级 RAG 的召回层强烈建议使用混合检索，纯向量检索在专有名词场景漏召回严重
- weights 必须基于真实业务评测集调优，不要凭直觉拍定
- 各路 k 值建议设为最终需要条数的 2~3 倍，给 RRF 留足融合空间
- BM25 的中文分词需要特别注意：BM25Retriever 默认按空格切分，对中文几乎等同于整句匹配；生产环境应传入 jieba 等中文分词器作为 preprocess_func，否则 BM25 路效果大打折扣
- BM25Retriever 数据常驻内存，不适合超大规模语料；大数据量应改用 Elasticsearch / OpenSearch 的 BM25 能力
- 两路检索器的数据源应保持一致（同一批 documents），否则融合结果会不完整
- 混合检索之后再接 Cross-Encoder ReRank（第 52 节）可进一步提升精排质量，形成"多路召回 → RRF 融合 → 精排"的标准三段式架构
- EnsembleRetriever 内部是顺序调用子检索器的，若某路较慢会拖累整体延迟，高性能场景可自行实现基于 abatch 的并发版本

===================================================================================
"""
import dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 加载 .env 环境变量，提供 OpenAI 凭证（向量检索路需要）
dotenv.load_dotenv()

# 1.创建文档列表
# Document(page_content: str, metadata: dict) -> Document
#   作用：构建 LangChain 标准文档对象
#   参数：page_content - 正文内容，同时作为 BM25 的分词语料与向量化的输入
#         metadata     - 元数据，本例用 page 标记页码
#   数据设计分析（针对查询"除了猫，你养了什么宠物呢？"）：
#     page=1  "笨笨是一只很喜欢睡觉的猫咪" - 字面含"猫"，BM25 强命中
#     page=3  "猫咪在窗台上打盹"           - 字面含"猫"，BM25 强命中
#     page=10 "我的狗喜欢追逐球"           - 字面不含"猫"也不含"宠物"，
#                                            BM25 必然漏召回，只有向量检索能命中
#   这组数据专门设计用来暴露单一检索方式的盲区
documents = [
    Document(page_content="笨笨是一只很喜欢睡觉的猫咪", metadata={"page": 1}),
    Document(page_content="我喜欢在夜晚听音乐，这让我感到放松。", metadata={"page": 2}),
    Document(page_content="猫咪在窗台上打盹，看起来非常可爱。", metadata={"page": 3}),
    Document(page_content="学习新技能是每个人都应该追求的目标。", metadata={"page": 4}),
    Document(page_content="我最喜欢的食物是意大利面，尤其是番茄酱的那种。", metadata={"page": 5}),
    Document(page_content="昨晚我做了一个奇怪的梦，梦见自己在太空飞行。", metadata={"page": 6}),
    Document(page_content="我的手机突然关机了，让我有些焦虑。", metadata={"page": 7}),
    Document(page_content="阅读是我每天都会做的事情，我觉得很充实。", metadata={"page": 8}),
    Document(page_content="他们一起计划了一次周末的野餐，希望天气能好。", metadata={"page": 9}),
    Document(page_content="我的狗喜欢追逐球，看起来非常开心。", metadata={"page": 10}),
]

# 2.构建BM25关键词检索器
# BM25Retriever.from_documents(
#     documents: list[Document],
#     bm25_params: dict = None,
#     preprocess_func: Callable = default_preprocessing_func
# ) -> BM25Retriever
#   作用：基于文档列表构建 BM25 稀疏检索器
#   参数：documents - 数据源文档列表
#   返回：BM25Retriever 实例
#   内部执行流程：
#     1) 用 preprocess_func 对每篇 page_content 分词（默认 text.split()，按空格切分）
#     2) 统计词频 TF 与逆文档频率 IDF，构建 rank_bm25.BM25Okapi 索引
#     3) 索引完全驻留内存，检索时纯本地计算
#   重要提示：默认分词器对中文极不友好（中文没有空格，整句会被当作一个"词"），
#             生产环境应传入 preprocess_func=lambda x: jieba.lcut(x) 做中文分词，
#             否则 BM25 路的召回能力会严重受限
#   优势：零 API 成本、毫秒级延迟、对专有名词精确匹配
bm25_retriever = BM25Retriever.from_documents(documents)

# 设置 BM25 返回条数
# bm25_retriever.k = 4
#   作用：BM25Retriever 的返回条数通过实例属性设置（而非构造参数）
#   说明：设为 4 而非最终需要的条数，是为 RRF 融合预留候选空间——
#         多路各给 4 条，融合后择优，比各给 2 条效果更好
bm25_retriever.k = 4

# 3.创建FAISS向量数据库检索
# FAISS.from_documents(documents: list[Document], embedding: Embeddings) -> FAISS
#   作用：一次性完成向量化 + 建索引 + 存文档，构建内存向量库
#   参数：documents - 数据源文档列表（与 BM25 使用同一批数据，保证两路可比）
#         embedding - OpenAIEmbeddings(model="text-embedding-3-small")，
#                     输出 1536 维稠密向量
#   返回：FAISS 向量库实例
#   内部执行流程：
#     1) 提取所有 page_content
#     2) 调用 embedding.embed_documents() 批量生成向量（唯一的网络请求）
#     3) 构建 IndexFlatL2 精确检索索引，并用 InMemoryDocstore 保存原文
faiss_db = FAISS.from_documents(documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

# faiss_db.as_retriever(search_kwargs: dict) -> VectorStoreRetriever
#   作用：把 FAISS 向量库包装为标准检索器
#   参数：search_kwargs={"k": 4} - 返回 4 条，与 BM25 路保持对等，
#                                  保证 RRF 融合时两路的话语权均衡
#   返回：VectorStoreRetriever 实例（search_type 未指定，默认 "similarity"）
#   优势：理解语义，能召回"狗"这类字面不匹配但语义相关（同属宠物）的文档
faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 4})

# 4.初始化集成检索器
# EnsembleRetriever(
#     retrievers: list[BaseRetriever],
#     weights: list[float] = None,
#     c: int = 60,
#     id_key: str = None
# ) -> EnsembleRetriever
#   作用：把多个检索器组合为一个，内部用带权重的 RRF 算法融合各路结果
#   参数详解：
#     retrievers - 子检索器列表，可放入任意数量、任意类型的 BaseRetriever
#                  本例为 [BM25（稀疏/字面）, FAISS（稠密/语义）]
#     weights    - 各路权重，长度必须与 retrievers 一致，内部会自动归一化
#                  [0.5, 0.5] 表示字面匹配与语义匹配等权重
#     c          - RRF 平滑常数，默认 60（本例使用默认值）
#   返回：EnsembleRetriever 实例，本身也是 BaseRetriever，可继续嵌套组合
#   RRF 融合计算逻辑：
#     score(d) = Σ_i  weights[i] * 1 / (c + rank_i(d))
#     其中 rank_i(d) 是文档 d 在第 i 路结果中的排名（从 0 开始）
#     只依赖排名而不依赖原始得分，因此能融合 BM25 分数（[0,+∞)）
#     与余弦相似度（[-1,1]）这两种量纲完全不同的评分体系
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
)

# 5.执行检索
# ensemble_retriever.invoke(input: str) -> list[Document]
#   作用：执行混合检索
#   参数：input - 查询语句
#   返回：RRF 融合排序后的去重文档列表
#   完整数据流转（关键执行流程）：
#     1) invoke("除了猫，你养了什么宠物呢？")
#     2) BM25 路：对查询分词后计算 BM25 得分
#        - 命中含"猫"字的 page=1、page=3（字面精确匹配）
#        - 完全不理解"宠物"是"猫/狗"的上位概念，无法命中 page=10
#        → 返回 4 条，头部是猫相关文档
#     3) FAISS 路：embed_query(查询) 生成 1536 维向量并计算余弦相似度
#        - "宠物"的语义与"猫""狗"都接近，能同时召回 page=1、page=3、page=10
#        → 返回 4 条，覆盖猫和狗
#     4) RRF 融合：
#        - page=1、page=3 在两路中都靠前 → 得分累加，排名最高（多路共识）
#        - page=10 只在向量路命中 → 得分较低但仍进入结果集
#          这正是混合检索的价值：单靠 BM25 永远拿不到这条
#        → 按融合得分降序，去重后返回
#   预期效果：结果中同时出现猫（字面命中）与狗（语义命中），
#             完整回应了用户"除了猫还有什么宠物"的真实意图
docs = ensemble_retriever.invoke("除了猫，你养了什么宠物呢？")

# 打印融合后的文档列表（顺序即 RRF 得分降序）
print(docs)

# 打印文档条数
# 由于两路各返回 4 条且存在重叠，去重融合后通常为 4~8 条
# 观察要点：条数多于任一单路，且内容覆盖面更广，这就是混合检索的召回增益
print(len(docs))
