#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 19:18
@Author  : thezehui@gmail.com
@File    : 2.回答回退策略检索器.py

===================================================================================
知识点讲解：Step-Back Prompting 回答回退策略
===================================================================================

1. 过于具体的问题为什么检索不到
   - 用户提问往往包含大量具体限定词，例如"人工智能会让世界发生翻天覆地的变化吗？"
   - 这类问题的向量带有强烈的"具体表述特征"，而知识库中的文档通常是
     概念性、原理性的通用描述（如"人工智能的发展趋势与社会影响"）
   - 两者向量距离较远，导致明明有相关资料却召回失败
   - 另一种典型场景：问题中含有知识库不存在的专有名词，向量被这个词"带偏"

2. Step-Back Prompting 的核心思想
   - 源自 Google DeepMind 论文《Take a Step Back: Evoking Reasoning through
     Abstraction in Large Language Models》
   - 思路：先"后退一步"，把具体问题抽象为更宽泛的前置问题（Step-Back Question），
     用这个更通用的问题去检索，从而命中知识库中的概念性文档
   - 类比人类思维：回答"笨笨这只猫为什么爱睡觉"之前，
     先想"猫的睡眠习性是怎样的"，从一般原理推导具体答案

3. 抽象方向的三种典型模式（本例示例体现得很清楚）
   - 范围放大："慕课网上有关于AI应用开发的课程吗？" → "慕课网上有哪些课程？"
     （从具体品类退到全集）
   - 维度上升："慕小课出生在哪个国家？" → "慕小课的人生经历是什么样的？"
     （从单一属性退到整体背景）
   - 约束去除："司机可以开快车吗？" → "司机可以做什么？"
     （从特定行为退到行为集合）

4. 为什么必须用 Few-Shot 来实现问题回退
   - "把问题抽象化"是一个高度依赖语感的任务，纯文字指令很难描述清楚
     "抽象到什么程度才合适"——退得太少没效果，退得太多就跑题了
   - 通过 3 个示例，模型能直接类比出合适的抽象粒度
   - 这正是上一个示例（1.少量示例提示模板.py）的实际应用

5. 本例检索器的链式设计（注意最后一环）
     {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser() | self.retriever
   - 前四环生成回退问题的字符串
   - 关键点：最后一环直接把 retriever 接入管道！
     因为 BaseRetriever 实现了 Runnable，输入 str 输出 list[Document]，
     正好能承接 StrOutputParser 的字符串输出
   - 所以 chain.invoke(query) 的返回值直接就是 List[Document]，
     一条管道同时完成"改写 + 检索"，代码极其简洁

6. Step-Back 与其他策略的区别
   - Multi-Query：横向扩展，生成多个同层级的等价查询（广度）
   - Step-Back：纵向抽象，生成一个更高层级的问题（深度/高度）
   - 问题分解：向下拆解，生成多个更细的子问题（与 Step-Back 方向相反）
   - HyDE：生成假设性答案文档，做 doc-doc 对称检索（形态转换）
   - 四者可以组合使用，例如先 Step-Back 再 Multi-Query

7. 本例实现的一个局限
   - 只用"回退问题"检索，完全丢弃了原始问题的检索结果
   - 论文中的标准做法是"原始问题 + 回退问题"都检索，然后合并，
     既保留具体细节的命中，又补充概念性背景
   - 生产环境建议改为双路召回 + RRF 融合

8. 最佳实践建议
   - 回退问题的示例要覆盖多种抽象方向（放大范围、上升维度、去除约束），
     示例质量直接决定改写效果
   - 生成回退问题的 LLM 必须 temperature=0，保证同一问题的回退结果稳定
   - 推荐改为"原始问题 + 回退问题"双路检索并用 RRF 融合，避免丢失具体细节
   - 不要过度回退：退得太宽泛会召回大量无关文档，反而降低精度；
     可在 system 提示中加入"保持与原问题同一领域"的约束
   - 适用场景：概念性提问、含知识库不存在的专有名词、用户表述过于具体；
     不适用：精确参数查询、ID/编号查找（回退会丢失关键限定词）
   - 本示例在 _get_relevant_documents 内部每次都重建 Prompt 与链，
     属于教学写法；生产环境应在 __init__ 或类级别构建一次以复用，减少开销
   - 内部调用 self.retriever 时建议传 config={"callbacks": run_manager.get_child()}，
     保持 LangSmith 上的调用链完整可追踪

===================================================================================
"""
from typing import List

import dotenv
import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()


class StepBackRetriever(BaseRetriever):
    """回答回退检索器

    继承 BaseRetriever 实现自定义检索逻辑：
    先用 LLM 把具体问题抽象为更宽泛的前置问题，再用该问题执行向量检索。
    """

    # 【Pydantic 字段声明】
    # retriever: 底层基础检索器，真正执行向量检索的组件
    # llm:       负责生成回退问题的大语言模型
    # 使用抽象基类做类型标注（BaseRetriever / BaseLanguageModel），
    # 便于替换具体实现（换向量库、换模型）而不用改这个类
    retriever: BaseRetriever
    llm: BaseLanguageModel

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """根据传递的query执行问题回退并检索

        参数：
            query: str - 用户的原始（通常较具体的）问题
            run_manager: CallbackManagerForRetrieverRun - 回调管理器（关键字参数）

        返回：
            List[Document] - 用回退问题检索到的文档列表

        执行流程：
            1) 构建 Few-Shot 示例，教会 LLM 如何做问题抽象
            2) 组装完整 Prompt（system 指令 + 示例 + 真实问题）
            3) 构建 "生成回退问题 → 检索" 的一体化链
            4) 执行链并直接返回文档列表
        """
        # 1.构建少量示例提示模板
        # 示例数据：每条包含 input（原始具体问题）与 output（回退后的宽泛问题）
        # 三个示例分别示范三种抽象方向：
        #   ① 范围放大：具体品类 → 全集（AI课程 → 所有课程）
        #   ② 维度上升：单一属性 → 整体背景（出生国家 → 人生经历）
        #   ③ 约束去除：特定行为 → 行为集合（开快车 → 能做什么）
        # 这三个示例共同界定了"抽象粒度"，是效果好坏的关键
        examples = [
            {"input": "慕课网上有关于AI应用开发的课程吗？", "output": "慕课网上有哪些课程？"},
            {"input": "慕小课出生在哪个国家？", "output": "慕小课的人生经历是什么样的？"},
            {"input": "司机可以开快车吗？", "output": "司机可以做什么？"},
        ]

        # ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
        #   作用：定义单条示例的渲染格式（一问一答的对话形式）
        #   参数：("human", "{input}") - 原始问题
        #         ("ai", "{output}")   - 回退后的问题
        #   注意：变量名 input/output 必须与 examples 中 dict 的 key 完全一致
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        # FewShotChatMessagePromptTemplate(examples, example_prompt)
        #     -> FewShotChatMessagePromptTemplate
        #   作用：把 3 条示例批量渲染为 Human/AI 交替的 6 条消息
        #   参数：examples       - 示例数据列表
        #         example_prompt - 单条示例的渲染模板
        #   返回：可嵌入到其他 ChatPromptTemplate 中的模板对象
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        # 2.构建生成回退问题的模板
        # ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
        #   作用：组装完整 Prompt，采用"系统指令 + 少量示例 + 真实问题"三段式结构
        #   参数：messages 列表的三个元素：
        #     ("system", "...")       - 设定角色（世界知识专家）与任务
        #                               （把问题改述为更一般或前置的问题）
        #     few_shot_prompt         - 嵌入的示例模板，渲染时展开为 6 条消息
        #     ("human", "{question}") - 真实待回退问题的占位符
        #   最终渲染消息序列（共 8 条）：
        #     [System, Human(例1), AI(例1), Human(例2), AI(例2),
        #      Human(例3), AI(例3), Human(真实问题)]
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个世界知识的专家。你的任务是回退问题，将问题改述为更一般或者前置问题，这样更容易回答，请参考示例来实现。"),
            few_shot_prompt,
            ("human", "{question}"),
        ])

        # 3.构建链应用，生成回退问题，并执行相应的检索
        # LCEL 管道五个环节：
        #   环节1: {"question": RunnablePassthrough()}
        #     作用：把传入的字符串原样包装为 {"question": "原字符串"}，
        #           以满足 prompt 需要 dict 输入的要求
        #     RunnablePassthrough() - 恒等 Runnable，输入即输出
        #   环节2: prompt
        #     作用：填充 question 变量，渲染出含示例的 8 条消息
        #     输出：PromptValue
        #   环节3: self.llm
        #     作用：调用大语言模型，模仿示例生成回退问题
        #     输出：AIMessage（content 即回退后的问题）
        #   环节4: StrOutputParser()
        #     作用：提取 AIMessage.content，得到回退问题的纯字符串
        #     输出：str
        #   环节5: self.retriever   ← 本例最精妙的一步
        #     作用：因 BaseRetriever 实现了 Runnable（输入 str、输出 List[Document]），
        #           可以直接接在字符串输出之后，形成"改写即检索"的一体化管道
        #     输出：List[Document]
        # 整体数据流转：
        #   str → dict → PromptValue → AIMessage → str(回退问题) → List[Document]
        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
                | self.retriever
        )

        # chain.invoke(query) -> List[Document]
        #   作用：执行完整链路，返回值类型恰好满足 _get_relevant_documents 的契约
        #   参数：query - 用户的原始问题
        #   返回：用回退问题检索到的文档列表
        #   优化建议：可传 config={"callbacks": run_manager.get_child()}
        #             让 LangSmith 记录完整的嵌套调用链
        return chain.invoke(query)


# 1.构建向量数据库与检索器
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点
#         auth_credentials - AuthApiKey 包装的 API Key
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：包装已有 Collection 为 LangChain 向量库
#   参数：index_name="DatasetDemo" - 目标 Collection
#         text_key="text"          - 正文字段名
#         embedding                - 嵌入模型，需与入库时一致
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
#   作用：构建底层基础检索器，作为 StepBackRetriever 内部链的最后一环
#   参数：search_type="mmr" - 回退问题较为宽泛，用 MMR 保证召回内容的多样性，
#                             避免返回一堆讲同一件事的段落
#   返回：VectorStoreRetriever 实例（默认 k=4）
retriever = db.as_retriever(search_type="mmr")

# 2.创建回答回退检索器
# StepBackRetriever(retriever: BaseRetriever, llm: BaseLanguageModel) -> StepBackRetriever
#   作用：实例化自定义的 Step-Back 检索器
#   参数：retriever - 底层基础检索器，用回退问题执行真实的向量检索
#         llm       - 生成回退问题的模型，temperature=0 保证改写结果稳定可复现
#   返回：StepBackRetriever 实例，具备完整的 Runnable 能力
#   说明：构造与类型校验由 BaseRetriever 的 Pydantic 基类自动完成，无需写 __init__
step_back_retriever = StepBackRetriever(
    retriever=retriever,
    llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
)

# 3.检索文档
# step_back_retriever.invoke(input: str) -> list[Document]
#   作用：执行 Step-Back 检索
#   参数：input - 一个偏具体、偏观点性的问题
#   返回：Document 列表
#   完整数据流转（关键执行流程）：
#     1) invoke("人工智能会让世界发生翻天覆地的变化吗？")
#     2) BaseRetriever.invoke 触发回调，调用 _get_relevant_documents
#     3) Few-Shot Prompt 引导 LLM 生成回退问题，预期类似：
#          "人工智能对世界有哪些影响？" 或 "人工智能是什么，它能做什么？"
#        （从"是否会翻天覆地"这个带有观点色彩的具体判断，
#          退到"有哪些影响"这个中性、宽泛的信息型问题）
#     4) 回退问题字符串直接流入 self.retriever
#     5) embed_query → Weaviate MMR 检索 → 返回 4 条文档
#   效果说明：直接用原问题检索时，"翻天覆地"这类修辞词会干扰向量；
#             回退后的问题更贴近文档的客观描述语言，命中率显著提升
documents = step_back_retriever.invoke("人工智能会让世界发生翻天覆地的变化吗？")

# 打印检索到的完整文档列表
print(documents)

# 打印文档条数，预期为 4（由底层 retriever 的默认 k=4 决定）
# 注意：本例只用回退问题检索，未合并原始问题的结果；
#       生产环境建议双路召回 + RRF 融合（见知识点 7）
print(len(documents))
