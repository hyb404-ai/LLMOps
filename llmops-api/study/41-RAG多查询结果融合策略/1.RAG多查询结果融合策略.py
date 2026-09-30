#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/3 23:45
@Author  : thezehui@gmail.com
@File    : 1.RAG多查询结果融合策略.py

===================================================================================
知识点讲解：RAG-Fusion 多查询结果融合（RRF 倒数排序融合）
===================================================================================

1. Multi-Query 遗留的问题：只管召回，不管排序
   - MultiQueryRetriever 的 unique_union 是「基于集合的去重」，本质是无序的
   - 结果：召回率上去了，但最相关的文档可能排在列表末尾
   - LLM 对上下文存在「首尾注意力偏好」（Lost in the Middle 现象），排在中间/末尾的关键文档很可能被忽略
   - 因此「如何给多路召回结果重新排序」成为必须解决的问题

2. RAG-Fusion 的定义
   - RAG-Fusion = Multi-Query（多查询扩展）+ RRF（倒数排序融合）
   - 先用 LLM 把问题扩展成多个查询，多路召回；再用 RRF 把多个有序结果列表融合成一个统一排序的列表
   - 核心价值：同时获得「高召回」与「好排序」
   - 运行流程：多查询重写 → 分别检索 → RRF 重排 → 取 Top K → 喂给 LLM 生成答案

3. RRF（Reciprocal Rank Fusion）算法原理
   - 公式：score(d) = Σ over all lists  1 / (k + rank(d))，rank(d) 是文档 d 在某结果列表中的排名，k 为平滑常数（经典取 60）
   - 三个关键特性：
     * 只依赖「排名」，不依赖「原始得分」——可融合量纲完全不同的检索系统（向量相似度、BM25、业务权重都能混合），这是 RRF 最大优势
     * 排名越靠前贡献越大（1/61 > 1/62 > ...），符合直觉
     * 在多个列表中重复出现的文档会累加得分，实现「多路投票共识」
   - k=60 的由来：由滑铁卢大学与 Google 合作提出，论文通过四个试点实验（各 30 种搜索配置应用于不同 TREC 集合）发现 k≈60 接近最优
   - 重要结论：k 值大小并非关键，它的作用是让低排名文档的重要性不会像指数函数那样骤然消失（高排名更重要，但低排名仍保留一定权重）

4. RRF 的 Python 实现（具象化）
   - 标准实现即对文档全集做二重遍历，按文档字符串累计 1/(rank + k)：
       def rrf(results: list[list], k: int = 60) -> list[tuple]:
           fused_scores = {}
           for docs in results:
               for rank, doc in enumerate(docs):
                   doc_str = dumps(doc)          # Document 不可哈希，序列化为字符串作 key
                   fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
           return [(loads(d), s) for d, s in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
   - 注意枚举从 0 开始，所以公式中实际用的是 rank+1 的位置；本例 k 取 60 与论文一致

5. 本例自定义检索器的两个重写点
   - retrieve_documents()：父类返回「扁平化的单层列表」（丢失每路的排名信息），重写后返回「嵌套列表 List[List[Document]]」，保留每路结果的内部顺序，这是 RRF 能够工作的前提
   - unique_union()：父类用集合去重，重写后用 RRF 计算得分并排序，最后取 Top-K（self.k，默认 4）

6. dumps / loads 的妙用（序列化作为哈希键）
   - 问题：Document 对象不可哈希（未实现 __hash__），不能直接作为 dict 的 key
   - 解决：langchain_core.load.dumps(obj) -> str 把对象序列化为 JSON 字符串，字符串天然可哈希，且内容相同的 Document 必然序列化为相同字符串
   - 排序完成后再用 loads(str) -> Document 反序列化还原对象
   - 这是 LangChain 中处理「对象去重/计分」的通用技巧

7. 继承 MultiQueryRetriever 的复用价值
   - 查询生成（llm_chain）、Prompt、输出解析、回调传递等逻辑全部复用父类
   - 子类只需替换「结果收集方式」和「融合算法」两处，改动面极小
   - 这是非常典型的「模板方法模式 + 最小侵入扩展」实践范例

8. 典型输出示例与观察点
   - 对「关于 LLMOps 应用配置的文档有哪些」执行融合检索，最终返回 self.k=4 条文档：
       [Document(metadata={'source': './项目API文档.md', ...}, page_content='LLMOps 项目 API 文档...'),
        Document(... page_content='json { "code": "success", "data": {...} }...'),
        Document(...), Document(...)]
       4
   - 观察点 1：输出条数被截断为 4（self.k），且经过 RRF 重排，最相关文档排在最前
   - 观察点 2：LangChain 没有直接实现 RAG-Fusion 检索器，本例正是通过继承 MultiQueryRetriever 并重写 retrieve_documents() 与 unique_union() 落地

===================================================================================
最佳实践建议
===================================================================================

- k=60 是论文推荐的经典值，通常不需要调整；若头部结果不够突出可适当减小
- RRF 的最大价值在于"跨系统融合"：向量检索 + BM25 + ES + 业务规则都能统一排序，
  所以生产环境的混合检索几乎都以 RRF 为标准融合算法
- self.k（最终返回条数）建议设为 4~8，太多会让 Prompt 过长且引入噪声
- 生成查询的 LLM 必须 temperature=0，否则每次融合结果都会波动
- 性能注意：RRF 本身是 O(N) 的纯内存计算，开销可忽略；
  真正的耗时在"1 次 LLM 改写 + N 次向量检索"，可用 abatch 并发检索优化
- dumps/loads 的序列化开销在文档量大时不可忽略，超大规模场景可改用
  page_content 的 hash 值作为 key，换取更高性能
- 若对精度要求极高，可在 RRF 之后再接一层 Cross-Encoder ReRank（第 52 节）

===================================================================================
"""
from typing import List

import dotenv
import weaviate
from langchain_core.load import dumps, loads
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()


class RAGFusionRetriever(MultiQueryRetriever):
    """RAG多查询结果融合策略检索器

    继承 MultiQueryRetriever，复用其"LLM 生成多查询"的能力，
    仅替换结果收集方式与融合算法，实现 RRF 倒数排序融合。
    """

    # 【Pydantic 字段】最终返回的文档条数上限，默认 4
    # 注意这个 k 是"输出截断的 k"，与 RRF 公式中的平滑常数 60 是两个不同概念
    k: int = 4

    def retrieve_documents(
            self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[List]:
        """重写检索文档函数，返回值变成一个嵌套的列表

        父类实现返回 List[Document]（把所有查询的结果拍平到一个列表），
        这样就丢失了"每个查询内部的排名信息"，无法执行 RRF。
        因此这里改为返回 List[List[Document]]，每个子列表对应一个查询的有序结果。

        参数：
            queries: List[str] - 由父类 generate_queries() 生成的查询列表
                                 （LLM 改写产生的多个等价查询）
            run_manager: CallbackManagerForRetrieverRun - 回调管理器

        返回：
            List[List[Document]] - 嵌套列表，外层长度 = 查询数，
                                   内层是该查询的检索结果（保持相关性顺序）

        执行流程：
            1) 遍历每一个改写后的查询
            2) 调用底层 retriever.invoke() 执行真实检索
            3) 通过 run_manager.get_child() 传递回调，保持调用链可追踪
            4) 把每路结果作为一个整体（保留顺序）追加到外层列表
        """
        documents = []
        for query in queries:
            # self.retriever 是父类字段，即构造时传入的底层基础检索器
            # retriever.invoke(query, config) -> list[Document]
            #   参数：query  - 单个改写查询
            #         config - 运行时配置；callbacks 传入子回调管理器，
            #                  使得每一路子检索都能在 LangSmith 上形成嵌套调用记录
            #   返回：该查询的检索结果，已按相关性降序排列（顺序至关重要）
            docs = self.retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
            # 关键点：整个 docs 列表作为"一路结果"整体追加，
            # 不做 extend 拍平，从而保留每路内部的排名（index 即 rank）
            documents.append(docs)
        return documents

    def unique_union(self, documents: List[List]) -> List[Document]:
        """使用RRF算法来去重合并对应的文档，参数为嵌套列表，返回值为文档列表

        父类实现是基于集合的简单去重（无序），
        这里改为 RRF（Reciprocal Rank Fusion）倒数排序融合，
        既完成去重又给出统一的相关性排序。

        参数：
            documents: List[List[Document]] - retrieve_documents 返回的嵌套列表

        返回：
            List[Document] - 按 RRF 得分降序排列、截断为前 self.k 条的文档列表

        RRF 计算示例：
            假设文档 A 在查询1 中排第 0 位、在查询2 中排第 2 位：
                score(A) = 1/(0+60) + 1/(2+60) = 0.01667 + 0.01613 = 0.03280
            文档 B 只在查询1 中排第 1 位：
                score(B) = 1/(1+60) = 0.01639
            → A 得分更高，因为它获得了"多路共识"
        """
        # 1.定义一个变量存储每个文档的得分信息
        # 结构：{序列化后的文档字符串: 累加的 RRF 得分}
        # 用序列化字符串作为 key，因为 Document 对象本身不可哈希
        fused_result = {}

        # 2.循环两层获取每一个文档信息
        # 外层 docs：某一个查询的完整结果列表（一路召回）
        # 内层 enumerate：rank 为该文档在这一路中的排名（从 0 开始），doc 为文档对象
        for docs in documents:
            for rank, doc in enumerate(docs):
                # 3.使用dumps函数将类示例转换成字符串
                # dumps(obj) -> str
                #   作用：把 LangChain 对象（Document）序列化为 JSON 字符串
                #   目的：① 让不可哈希的 Document 变成可作为 dict key 的字符串
                #         ② 内容完全相同的 Document 序列化结果一致，天然实现去重
                doc_str = dumps(doc)

                # 4.判断下该文档的字符串是否已经计算过得分
                # 首次出现则初始化得分为 0，为后续累加做准备
                if doc_str not in fused_result:
                    fused_result[doc_str] = 0

                # 5.计算新的分
                # RRF 核心公式：score += 1 / (rank + k)，此处平滑常数 k = 60
                # 含义解析：
                #   - rank=0（排第一）贡献 1/60 ≈ 0.01667，贡献最大
                #   - rank=3（排第四）贡献 1/63 ≈ 0.01587，略小
                #   - 使用 += 累加，意味着在多路结果中重复出现的文档得分更高，
                #     体现"多个查询都认为它相关"的投票共识
                fused_result[doc_str] += 1 / (rank + 60)

        # 6.执行排序操作，获取相应的数据，使用的是降序
        # sorted(fused_result.items(), key=lambda x: x[1], reverse=True)
        #   作用：按累加得分从高到低排序
        #   参数：key     - 取元组的第 2 个元素（即得分）作为排序依据
        #         reverse - True 表示降序，得分高的排前面
        # loads(doc_str) -> Document
        #   作用：把序列化字符串反序列化还原为 Document 对象，与 dumps 互为逆操作
        # 结果结构：[(Document, score), ...]，已按 score 降序
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_result.items(), key=lambda x: x[1], reverse=True)
        ]

        # 取前 self.k 条，并剥离得分只保留 Document 对象
        # reranked_results[:self.k] → 截断为 Top-K
        # item[0]                   → 取元组的第 1 个元素（Document），丢弃得分
        # 说明：得分在这里被丢弃了；如果业务需要展示置信度，
        #       可以把 score 写入 document.metadata 后再返回
        return [item[0] for item in reranked_results[:self.k]]


# 1.构建向量数据库与检索器
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点
#         auth_credentials - AuthApiKey 包装的 API Key
#   返回：Weaviate v4 客户端
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：把 Weaviate Collection 包装为 LangChain 向量库
#   参数：index_name - Collection 名称 "DatasetDemo"
#         text_key   - 正文字段名
#         embedding  - 嵌入模型，需与入库时一致
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://mbakeruerziae6psyex7ng.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("ZltPVa9ZSOxUcfafelsggGyyH6tnTYQYJvBx"),
    ),
    index_name="DatasetDemo",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# db.as_retriever(search_type="mmr") -> VectorStoreRetriever
#   作用：构建底层基础检索器，供 RAGFusionRetriever 的每一路查询调用
#   参数：search_type="mmr" - 每路检索内部先保证多样性，
#                             与多查询的多角度扩展形成互补
#   返回：VectorStoreRetriever 实例（默认 k=4，即每路返回 4 条）
retriever = db.as_retriever(search_type="mmr")

# RAGFusionRetriever.from_llm(retriever, llm) -> RAGFusionRetriever
#   作用：继承自 MultiQueryRetriever 的工厂方法，自动装配查询生成链
#   参数：retriever - 底层基础检索器，每个改写查询都会调用它
#         llm       - 负责生成多个改写查询的大语言模型，temperature=0 保证可复现
#   返回：RAGFusionRetriever 实例
#   说明：此处未传 include_original，使用父类默认值 False，
#         即只用 LLM 改写后的查询检索；
#         若担心 LLM 改写跑偏，可显式传 include_original=True 作为保底
rag_fusion_retriever = RAGFusionRetriever.from_llm(
    retriever=retriever,
    llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
)

# 3.执行检索
# rag_fusion_retriever.invoke(input: str) -> list[Document]
#   作用：执行完整的 RAG-Fusion 检索流程
#   参数：input - 用户原始问题
#   返回：经 RRF 融合排序后的 Top-K（k=4）文档列表
#   完整数据流转（关键执行流程）：
#     1) invoke("关于LLMOps应用配置的文档有哪些")
#     2) generate_queries()（复用父类）→ LLM 生成 3 个改写查询
#     3) retrieve_documents()（本类重写）→ 对 3 个查询分别 mmr 检索，
#        得到嵌套列表 [[d1,d2,d3,d4], [d2,d5,d1,d6], [d7,d1,d3,d8]]
#     4) unique_union()（本类重写）→ 遍历嵌套列表计算 RRF 得分：
#        d1 在三路中分别排 0/2/1 → 得分 1/60 + 1/62 + 1/61 ≈ 0.0492（最高）
#        d2 在两路中分别排 1/0   → 得分 1/61 + 1/60 ≈ 0.0331
#        其余文档只出现一次，得分约 0.016
#     5) 按得分降序排序，取前 4 条返回
#   最终效果：d1 这类"被多个查询共同认可"的文档稳定排在最前，
#             解决了 Multi-Query 输出无序的问题
docs = rag_fusion_retriever.invoke("关于LLMOps应用配置的文档有哪些")

# 打印融合排序后的文档列表（顺序即相关性置信度顺序）
print(docs)

# 打印文档条数，预期为 4（由 self.k 默认值决定）
# 对比 Multi-Query：条数更少但排序更可信，更适合直接投喂给 LLM
print(len(docs))
