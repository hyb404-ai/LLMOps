#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/3 17:06
@Author  : thezehui@gmail.com
@File    : 1.带阈值的相似性搜索.py

===================================================================================
知识点讲解：VectorStore 带阈值的相似性搜索
===================================================================================
1. VectorStore 基类与四类核心检索方法
   - 市面上向量数据库众多、操作方式无统一标准，但存在公共特征，LangChain 基于这些
     特征抽象出 VectorStore 基类，其方法可划分为 6 类：相似性搜索、最大边际相关性搜索、
     通用搜索(search 按 search_type 分发)、添加删除与精确查找、检索器(as_retriever)、
     创建数据库(from_texts / from_documents)
   - 四类核心检索方法（本节只展开前两类，后两类见本目录后续示例）：
     * similarity_search(query, k)：最常用，返回 k 条最相似文档（只有 Document，无得分）
     * similarity_search_with_score(query, k)：返回 (Document, score)，score 是"原始距离"，
       不同向量库语义不同（FAISS 默认 L2 距离，越小越相似）
     * similarity_search_with_relevance_scores(query, k, score_threshold)：返回 (Document,
       relevance_score)，分数归一化到 [0,1]，越大越相似，可用统一阈值过滤
     * max_marginal_relevance_search(query, k, fetch_k, lambda_mult)：最大边际相关性搜索
   - 掌握这 6 类方法后，切换 FAISS / Weaviate / Pinecone 等任何实现时用法基本一致

2. 为什么需要"相关性得分"而不是"原始距离"
   - 原始距离的量纲依赖底层索引与距离函数（L2、内积、余弦），无法跨库比较
   - LangChain 通过 _select_relevance_score_fn() 把距离转换为 [0,1] 的相关性分数，
     例如 L2 距离的默认转换是 relevance = 1 - distance / sqrt(2)
   - 归一化之后，score_threshold=0.4 这类"业务阈值"才有可移植的含义

3. score_threshold 阈值过滤的意义（RAG 的关键防线）
   - 不带阈值时，即使知识库里完全没有相关内容，也一定会强行返回 k 条最"不那么远"的文档，
     这些噪声进入 Prompt 会显著提升大模型幻觉概率
   - 加上阈值后，低于阈值的文档被直接丢弃，可能返回空列表，
     上层可走"我不知道"的兜底回复，这比编造答案更安全
   - 设计动机：RAG 优化的目标始终是"提升 LLM 生成内容的准确性"，对 Transformer 类模型
     只需做到三件事——传递更准确的内容、让重要内容更靠前、尽可能不传递不相关内容，
     阈值过滤直接服务第三点，是成本最低、最立竿见影的一招
   - 重要提醒：RAG 优化策略并非越多越好，策略叠加会抬高单次响应成本、降低性能，
     需要合理使用，并非所有场合都适合复杂优化

4. 阈值的取值经验与调试原则
   - 0.3 ~ 0.5：召回优先，适合长文档问答；0.5 ~ 0.7：精度优先，适合 FAQ / 客服等准确场景
   - 不存在通用"最佳值"，必须结合具体 Embedding 模型实测校准
   - 中文场景下 text-embedding-3-small 的相关性得分普遍偏低（强相关也仅 0.46 左右），
     阈值不宜设得过高，否则很可能检索不到任何内容
   - 阈值还与向量库数据规模间接相关：数据集越大，检索准确度相对少量数据更高，
     因此小库上不宜把阈值调得过高
   - 一句话原则：阈值过大检索不到内容，阈值过小容易检索到不相关内容，
     对不同的文档 / 分割策略 / 向量数据库，得分阈值都不一致，必须调试

5. 典型输出示例与观察点（10 条语料中仅第 1、3 条与"猫"强相关）
   - 不加阈值检索 "我养了一只猫，叫笨笨"（默认 k=4）：
       ('笨笨是一只很喜欢睡觉的猫咪',   0.4592)   ← page 1，强相关
       ('猫咪在窗台上打盹，看起来非常可爱。', 0.2296)   ← page 3，中等相关
       ('我的狗喜欢追逐球，看起来非常开心。', 0.0216)   ← page 10，弱相关
       ('我的手机突然关机了，让我有些焦虑。', -0.0984)  ← page 7，完全无关，得分为负
   - 加上 score_threshold=0.4 后，输出只剩 1 条：
       ('笨笨是一只很喜欢睡觉的猫咪', 0.4592)
   - 观察点 1：不带阈值时，即使"手机关机"这种毫不相关的内容也会被强行返回，
     且相关性得分可以是负数——这些噪声一旦进入 Prompt 就推高幻觉概率
   - 观察点 2：中文语料上 text-embedding-3-small 的强相关得分也只有 0.46 左右，
     印证"阈值必须实测校准"，盲目设 0.6/0.7 会什么都检索不到

6. FAISS.from_documents() 的执行流程
   - 取出所有 Document.page_content → 调用 embedding.embed_documents() 批量向量化
   - 构建内存中的 FAISS 索引（默认 IndexFlatL2 精确暴力检索）
   - 用 InMemoryDocstore 保存原始 Document（含 metadata），并维护 index → doc_id 映射
   - 因此检索结果能同时拿到向量相似度和原始文本 + metadata

7. metadata 的作用
   - 本例中每条文档带 {"page": n}，检索结果原样携带
   - 上层可用于引用溯源（告诉用户答案出自第几页）或做元数据过滤（filter 参数）

8. 为检索器结果附加相关性得分（进阶思路）
   - 直接调用 retriever.invoke() 返回的是 Document 列表，并不携带相关性得分
   - 若想让检索结果带分，可自定义一个函数：内部调用
     similarity_search_with_relevance_scores() 把每条得分写入 document.metadata，
     再用 RunnableLambda 把该函数包装成 Runnable 接入链路
   - 这正是"检索器即 Runnable"带来的灵活性的体现

9. 版本对照说明（示例基于 LangChain 1.x）
   - similarity_search_with_relevance_scores 的签名与语义在 0.x → 1.x 之间未变，
     score_threshold 依然是可选关键字参数
   - 本仓库统一使用 DeepSeek 作为生成模型，与部分文档示例的 gpt-3.5-turbo 不同，
     不要按文档回改模型名
   - FAISS 仍可从 langchain_community.vectorstores 导入，新代码也可考虑独立 langchain-faiss 系包

10. 最佳实践建议
   - 生产环境优先使用 similarity_search_with_relevance_scores + score_threshold，
     避免无相关内容时返回噪声文档
   - 阈值必须在真实业务语料上做一轮校准，并随 Embedding 模型变更重新校准
   - 命中为空时不要直接报错，应返回友好的"未找到相关资料"提示
   - FAISS 索引在内存中，进程退出即丢失；正式项目应使用 save_local()/load_local() 持久化
   - 一次 from_documents 会把全部文本送去 Embedding，注意 token 成本与限流，
     大批量数据建议分批 add_documents 或搭配 CacheBackedEmbeddings 使用

===================================================================================
"""
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE 等）
# 必须在创建 OpenAIEmbeddings 之前执行，否则读取不到密钥
dotenv.load_dotenv()

# OpenAIEmbeddings(model: str, ...) -> OpenAIEmbeddings
#   作用：创建文本嵌入模型客户端，负责把文本转换为高维向量
#   参数：model - 嵌入模型名称，text-embedding-3-small 输出 1536 维向量，性价比高
#   返回：OpenAIEmbeddings 实例，提供两个核心方法：
#         embed_documents(texts: list[str]) -> list[list[float]]  批量嵌入文档
#         embed_query(text: str) -> list[float]                   嵌入单条查询
#   注意：入库与检索必须使用同一个嵌入模型，否则向量空间不一致，相似度完全失效
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 构建测试文档列表
# Document(page_content: str, metadata: dict) -> Document
#   作用：LangChain 的标准文档载体，是所有检索链路中流转的基本单位
#   参数：page_content - 会被向量化的正文内容
#         metadata    - 附加元数据，不参与向量化，但会随检索结果一起返回
#   说明：这 10 条语料中只有第 1、3 条与"猫"强相关，第 10 条是"狗"（弱相关），
#         其余为完全无关内容，正好用来观察阈值过滤的效果
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

# FAISS.from_documents(documents: list[Document], embedding: Embeddings, **kwargs) -> FAISS
#   作用：一次性完成"向量化 + 建索引 + 存文档"三件事，返回可直接检索的向量库实例
#   参数：documents - 待入库的 Document 列表
#         embedding - 嵌入模型，用于把 page_content 转成向量
#   返回：FAISS 向量库实例（VectorStore 子类）
#   内部执行流程：
#     1) texts = [d.page_content for d in documents]
#     2) embeddings = embedding.embed_documents(texts)   ← 唯一的网络请求，批量发出
#     3) 创建 faiss.IndexFlatL2(dim) 并 index.add(vectors)
#     4) 创建 InMemoryDocstore 保存 Document，建立 index_to_docstore_id 映射
#   注意：索引完全驻留内存，脚本结束即销毁；需要复用请调用 db.save_local("路径")
db = FAISS.from_documents(documents, embedding)

# db.similarity_search_with_relevance_scores(
#     query: str,
#     k: int = 4,
#     score_threshold: float | None = None,
#     **kwargs
# ) -> list[tuple[Document, float]]
#   作用：执行相似性搜索，返回归一化到 [0,1] 的相关性得分，并按阈值过滤低相关结果
#   参数：query           - 查询语句，会先经 embedding.embed_query() 转成向量
#         k               - 最多返回的文档条数，默认 4
#         score_threshold - 相关性阈值，只保留 relevance_score >= 阈值的文档
#   返回：[(Document, relevance_score), ...]，按 relevance_score 从高到低排列
#   内部执行流程：
#     1) query_vector = embedding.embed_query(query)
#     2) 调用 similarity_search_with_score_by_vector 拿到 (doc, L2距离)
#     3) 通过 _select_relevance_score_fn() 得到距离→相关性的转换函数
#        （FAISS + L2 时为 relevance = 1 - distance / sqrt(2)）
#     4) 过滤掉 relevance_score < score_threshold 的条目并返回
#   本例数据流转：
#     "我养了一只猫，叫笨笨"
#       → 1536 维查询向量
#       → FAISS 计算与 10 条文档向量的 L2 距离并排序
#       → 距离归一化为 [0,1] 相关性得分
#       → 丢弃得分 < 0.4 的文档
#       → 预期只剩 page=1（笨笨是猫咪）和 page=3（猫咪打盹）这类强相关文档
#   提示：若结果为空列表，说明知识库确实没有相关内容，应走兜底回复而非硬编答案
print(db.similarity_search_with_relevance_scores("我养了一只猫，叫笨笨", score_threshold=0.4))
