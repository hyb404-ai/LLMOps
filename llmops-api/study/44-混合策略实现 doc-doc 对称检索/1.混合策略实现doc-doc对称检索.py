#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 0:52
@Author  : thezehui@gmail.com
@File    : 1.混合策略实现doc-doc对称检索.py

===================================================================================
知识点讲解：HyDE 假设性文档嵌入与 doc-doc 对称检索
===================================================================================

1. 什么是"非对称检索"及其固有缺陷
   - 常规 RAG 是 query-doc 非对称检索：用一个"短问句"的向量去匹配一堆"长文档"的向量
   - 两者在形态上严重不对称：
     * 长度不对称：问题 10~30 字，文档 300~500 字
     * 句式不对称：问题是疑问句，文档是陈述句
     * 词汇不对称：问题用口语词，文档用书面术语
   - Embedding 模型是在"文本对相似度"上训练的，形态差异本身就会拉大向量距离，
     即使语义完全相关，相似度也上不去——这是 RAG 效果的一个系统性瓶颈
   - 此外 query 与答案可能只存在弱相关性，导致难以找到最相关的文档

2. HyDE（Hypothetical Document Embeddings）的核心思想
   - 论文《Precise Zero-Shot Dense Retrieval without Relevance Labels》（arXiv:2212.10496）
   - 思路：既然 doc-doc 比 query-doc 更对称，那就先把 query 变成 doc！
   - 具体做法：让 LLM 针对用户问题"凭空写一篇假设性答案文档"
     （即使内容可能有事实错误也不要紧），然后用这篇假设文档的向量去检索真实文档
   - 于是检索变成了 doc-doc 对称检索：长度、句式、词汇分布都高度一致，向量匹配准确度显著提升

3. 为什么"事实可能错误的假设文档"依然有效
   - 检索阶段只关心"语义/主题/术语分布"的相似性，不关心事实正确性
   - LLM 生成的假设文档虽然细节可能编造，但会自然使用
     该领域的正确术语、表达方式和论述结构，把查询向量"拉近"到真实文档所在的向量空间区域
   - 关键认知：假设文档只是"检索的桥梁"，绝不会进入最终答案；
     最终答案仍基于检索出的真实文档生成，所以不存在幻觉扩散风险

4. HyDE Prompt 的设计要点（本例分析）
   - 本例使用 "请写一篇科学论文来回答这个问题" 这一表述
   - "科学论文"这个体裁提示至关重要：引导 LLM 输出长篇、书面化、术语密集的文本，
     形态上最接近技术文档
   - 若换成"请简短回答"，生成内容太短，就退化成 query-doc 非对称检索，效果打折
   - 生产环境应根据知识库文体调整体裁提示：技术文档→"技术文档/API 说明"；
     法律文本→"法律条文"；客服 FAQ→"标准答复"

5. 与 Step-Back 策略的实现对比
   - 两者的代码骨架几乎完全相同：BaseRetriever 子类 + {"question": Passthrough} | prompt | llm | parser | retriever
   - 唯一区别在 Prompt 的意图：
     * Step-Back：把问题"抽象化"，输出仍是一个问题（query → query）
     * HyDE：把问题"文档化"，输出是一段文章（query → doc）
   - 这体现了 LangChain 自定义检索器的可塑性：同一套骨架，只换 Prompt 就能实现不同检索策略

6. 链式管道的精妙之处
     {"question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser() | self.retriever
   - 最后一环直接把 retriever 接入管道，因为它实现了 Runnable
     （输入 str、输出 List[Document]），正好承接 StrOutputParser 的输出
   - 一条管道同时完成"生成假设文档 + 用它检索"，代码极简

7. HyDE 的代价
   - 每次检索多一次 LLM 调用，且要生成较长文本（几百 token），延迟通常增加 2~5 秒，成本上升明显
   - 生成的假设文档很长，embed_query 的 token 消耗也随之增加
   - 因此 HyDE 更适合"离线批处理"或"对精度要求极高、能容忍延迟"的场景

8. 局限性与失败案例（来自课程文档 02 节）
   - 场景一：query 缺乏足够上下文时，HyDE 容易误解词语，产生错误假设文档导致检索失败
     例如问 "Bel 是什么？"，无上下文时 HyDE 把 Bel 误当成某人的化名，
     生成的假设文档完全偏题，从而检索不到相关文档
   - 场景二：开放式查询，HyDE 可能引入偏见
     例如 "作者会如何评价艺术与工程的区别？"，直接检索已能给出合理回答，
     经 HyDE 转换后反而生成带偏见的文档
   - 结论：HyDE 是不完全依赖 embedding、强调"答案与查找内容相似性"的无监督方法，
     能提升 RAG 效果，但若 LLM 无法理解用户问题，就不会产生最佳结果，需按场景决定是否采用

9. 最佳实践建议
   - 体裁提示要匹配知识库文体（论文/技术文档/法律条文/FAQ），这是效果的关键开关
   - 生成假设文档的 LLM 用 temperature=0，保证检索结果可复现；
     想增加召回多样性也可生成多篇假设文档再做 RRF 融合（HyDE + RAG-Fusion）
   - 明确认知：假设文档只用于检索，绝不能进入最终答案的上下文，否则会引入幻觉
   - 延迟敏感场景慎用；可对高频查询缓存假设文档，或降级为普通向量检索
   - 可与 Step-Back、Multi-Query 组合：先回退再 HyDE，或多个假设文档并行检索
   - 本例在 _get_relevant_documents 内部每次重建 Prompt 与链，属教学写法；
     生产环境应提前构建一次以复用
   - 建议内部调用时传 config={"callbacks": run_manager.get_child()}，保持追踪链完整
   - 假设文档过长可能超出 Embedding 模型输入上限（8191 token），可在 Prompt 中限定字数或截断

===================================================================================
"""
from typing import List

import dotenv
import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()


class HyDERetriever(BaseRetriever):
    """HyDE混合策略检索器

    继承 BaseRetriever 实现 HyDE（假设性文档嵌入）策略：
    先让 LLM 针对问题生成一篇假设性答案文档，
    再用这篇文档去检索真实文档，从而把 query-doc 非对称检索
    转换为 doc-doc 对称检索，提升向量匹配精度。
    """

    # 【Pydantic 字段声明】
    # retriever: 底层基础检索器，用假设文档执行真实的向量检索
    # llm:       负责生成假设性文档的大语言模型
    # 使用抽象基类标注类型，便于替换具体实现
    retriever: BaseRetriever
    llm: BaseLanguageModel

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """传递检索query实现HyDE混合策略检索

        参数：
            query: str - 用户的原始问题（短问句形态）
            run_manager: CallbackManagerForRetrieverRun - 回调管理器（关键字参数）

        返回：
            List[Document] - 用假设性文档检索到的真实文档列表

        执行流程：
            1) 构建"生成假设性文档"的 Prompt
            2) 组装 "生成假设文档 → 用它检索" 的一体化链
            3) 执行链并直接返回真实文档列表
        """
        # 1.构建生成假设性文档的prompt
        # ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
        #   作用：创建生成假设性文档的提示模板
        #   参数：template - 含 {question} 占位符的模板字符串
        #   返回：ChatPromptTemplate 实例
        #   Prompt 设计三个关键点：
        #     ① "请写一篇科学论文" —— 体裁提示，引导 LLM 输出长篇书面化、
        #        术语密集的文本，形态上贴近真实技术文档（这是 HyDE 生效的核心）
        #     ② "问题: {question}" —— 注入用户的原始问题作为主题约束
        #     ③ "文章: " —— 尾部引导词，明确告知模型从这里开始续写正文，
        #        避免输出"好的，我来写一篇……"这类寒暄前缀
        prompt = ChatPromptTemplate.from_template(
            "请写一篇科学论文来回答这个问题。\n"
            "问题: {question}\n"
            "文章: "
        )

        # 2.构建HyDE混合策略检索链
        # LCEL 管道五个环节：
        #   环节1: {"question": RunnablePassthrough()}
        #     作用：把传入的字符串原样包装为 {"question": "原字符串"}，
        #           满足 prompt 需要 dict 输入的要求
        #     RunnablePassthrough() - 恒等 Runnable，输入即输出
        #     输出：dict
        #   环节2: prompt
        #     作用：填充 question 变量，渲染出提示消息
        #     输出：PromptValue
        #   环节3: self.llm
        #     作用：调用 LLM 生成假设性文档（一篇论文体的长文本）
        #     输出：AIMessage
        #   环节4: StrOutputParser()
        #     作用：提取 AIMessage.content，得到假设文档的纯字符串
        #     输出：str（这就是"假设性文档"，可能长达数百字）
        #   环节5: self.retriever   ← HyDE 的关键一步
        #     作用：用假设文档作为检索输入（而非原始问题），
        #           内部会调用 embed_query(假设文档) 得到"文档形态"的向量，
        #           再与库中真实文档的向量比对 —— 这就是 doc-doc 对称检索
        #     输出：List[Document]（真实文档，不含假设文档）
        # 整体数据流转：
        #   str(短问句) → dict → PromptValue → AIMessage
        #     → str(假设性长文档) → List[Document](真实文档)
        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
                | self.retriever
        )

        # chain.invoke(query) -> List[Document]
        #   作用：执行完整链路，返回值恰好满足 _get_relevant_documents 的契约
        #   参数：query - 用户原始问题
        #   返回：真实文档列表（假设文档已在管道中被消耗，不会出现在返回值里）
        #   优化建议：可传 config={"callbacks": run_manager.get_child()}
        #             以在 LangSmith 上看到完整的嵌套调用链
        return chain.invoke(query)


# 1.构建向量数据库与检索器
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点
#         auth_credentials - AuthApiKey 包装的 API Key
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：包装已有 Collection 为 LangChain 向量库
#   参数：index_name="DatasetDemo" - 存放 LLMOps 文档的 Collection
#         text_key="text"          - 正文字段名
#         embedding                - 嵌入模型；注意它既用于库中文档的向量化，
#                                    也用于假设文档的向量化，保证在同一向量空间
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
#   作用：构建底层基础检索器，作为 HyDERetriever 内部链的最后一环
#   参数：search_type="mmr" - 假设文档内容较宽泛，可能同时命中多个相似段落，
#                             用 MMR 保证返回内容的多样性
#   返回：VectorStoreRetriever 实例（默认 k=4）
retriever = db.as_retriever(search_type="mmr")

# 2.创建HyDE检索器
# HyDERetriever(retriever: BaseRetriever, llm: BaseLanguageModel) -> HyDERetriever
#   作用：实例化 HyDE 检索器
#   参数：retriever - 底层基础检索器，用假设文档执行真实检索
#         llm       - 生成假设文档的模型；temperature=0 保证假设文档稳定，
#                     从而使检索结果可复现
#   返回：HyDERetriever 实例，具备完整 Runnable 能力
#   说明：构造与类型校验由 BaseRetriever 的 Pydantic 基类自动完成
hyde_retriever = HyDERetriever(
    retriever=retriever,
    llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
)

# 3.检索文档
# hyde_retriever.invoke(input: str) -> list[Document]
#   作用：执行 HyDE 检索
#   参数：input - 用户原始问题（短问句）
#   返回：Document 列表（真实文档）
#   完整数据流转（关键执行流程）：
#     1) invoke("关于LLMOps应用配置的文档有哪些？")
#     2) BaseRetriever.invoke 触发回调，调用 _get_relevant_documents
#     3) LLM 生成一篇论文体的假设性文档，内容大致会包含：
#          "LLMOps（大语言模型运维）平台的应用配置管理……
#           配置项通常包括模型选择、温度参数、提示模板、知识库绑定、
#           工具插件启用、长记忆开关……配置的持久化与版本管理……"
#        注意：这些内容可能与真实文档不完全一致（甚至有编造），
#              但它使用了正确的领域术语与书面化表达
#     4) 这篇长文档作为检索输入 → embed_query(假设文档) 生成向量
#        此时查询向量的"形态特征"（长度、术语密度、句式）
#        与库中真实文档高度一致，这就是 doc-doc 对称检索的优势
#     5) Weaviate MMR 检索 → 返回 4 条真实文档
#   效果对比：
#     直接用 "关于LLMOps应用配置的文档有哪些？" 检索时，
#     短问句向量与长文档向量的形态差异会压低相似度；
#     HyDE 消除了这种形态差异，通常能提升相关文档的命中率与排名
documents = hyde_retriever.invoke("关于LLMOps应用配置的文档有哪些？")

# 打印检索到的真实文档列表
# 注意：这里打印的全是库中的真实文档，假设性文档已在管道内部被消耗，不会出现
print(documents)

# 打印文档条数，预期为 4（由底层 retriever 的默认 k=4 决定）
print(len(documents))
