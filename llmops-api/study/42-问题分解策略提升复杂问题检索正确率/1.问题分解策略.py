#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 9:30
@Author  : thezehui@gmail.com
@File    : 3.问题分解策略.py

===================================================================================
知识点讲解：问题分解策略（Decomposition / Least-to-Most Prompting）
===================================================================================

1. 复杂问题为什么检索不准
   - 复杂问题往往包含多个子意图，例如"LLMOps 的应用配置怎么做、有哪些限制、出错怎么排查"，这是三个独立的知识点
   - 把它们压缩成一个向量后，得到的是多个语义的"平均值"，这个平均向量可能不接近任何一个具体文档，导致三个方面都召回不佳
   - 这是向量检索的固有缺陷：一个向量只能表达一个聚焦的语义

2. 问题分解策略的核心思想
   - 先让 LLM 把复杂问题拆解为若干个可以独立回答的子问题
   - 每个子问题语义聚焦，单独检索时向量匹配更精准
   - 分别回答子问题，最后把子答案汇总形成对原问题的完整回答
   - 这就是论文中的 Least-to-Most Prompting：从最简单的子问题逐步推进到复杂问题

3. 两种子问题处理模式
   - 并行模式（Individual Answer）：各子问题独立检索、独立回答，最后统一汇总。
     优点是可并发、延迟低；缺点是子答案之间缺乏关联
   - 串行模式（Recursive Answer，本例采用）：按顺序回答，
     把"已回答的 QA 对"作为上下文传给下一个子问题，形成递进推理链。
     优点是后面的回答能利用前面的结论，逻辑更连贯；缺点是无法并发、延迟叠加

4. 本例迭代问答链的三路输入（关键设计）
   - question：当前正在处理的子问题
   - qa_pairs：之前所有子问题的"问题+答案"累积文本（递进推理的载体）
   - context：针对当前子问题实时检索到的文档（RAG 的知识来源）
   - 三者共同构成 Prompt，让 LLM 既看到历史推理又看到新鲜资料

5. itemgetter 在 LCEL 中的作用
   - operator.itemgetter("question") 等价于 lambda x: x["question"]
   - 在 LCEL 并行字典中用于从上游输入 dict 中"抽取指定字段"
   - itemgetter("question") | retriever 是一个子链：
     先抽出 question 字符串，再把它作为输入喂给 retriever，
     这样就把 dict 输入适配成了 retriever 需要的 str 输入
   - 用它比 lambda 更简洁，且 LangChain 能更好地序列化和追踪

6. 分解链的输出后处理
   - LLM 返回的是多行文本，需要 (lambda x: x.strip().split("\n")) 转为列表
   - 这个 lambda 会被 LCEL 自动包装为 RunnableLambda
   - 风险点：LLM 可能输出带序号的行（"1. xxx"）或包含空行，
     生产环境应做更健壮的清洗（去序号、过滤空行、限制数量）

7. 最佳实践建议
   - 分解链的 LLM 必须 temperature=0，否则子问题每次都不同，结果无法复现
   - Prompt 中明确限定子问题数量（本例为 3 个），否则 LLM 可能生成过多导致延迟爆炸
   - LLM 输出解析要健壮：过滤空行、剥离 "1." 之类的序号前缀、去重、限制上限
   - 串行模式的延迟是"子问题数 × (1 次检索 + 1 次 LLM)"，3 个子问题约 5~10 秒；
     对延迟敏感的场景改用并行模式，配合 RunnableParallel 或 abatch 并发执行
   - qa_pairs 会随迭代不断变长，需注意上下文窗口上限，必要时做摘要压缩
   - 本例只打印了各子问题的答案，生产环境还应再加一次"汇总链"，
     把所有 QA 对综合成对原始问题的最终回答
   - 简单问题不要用分解策略，会白白增加数倍成本与延迟；
     可先用一个轻量分类器判断问题复杂度，再决定是否走分解流程

===================================================================================
"""
from operator import itemgetter

import dotenv
import weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()


def format_qa_pair(question: str, answer: str) -> str:
    """格式化传递的问题+答案为单个字符串

    作用：把一组"子问题 + 子答案"拼装成结构化文本，
          作为后续子问题的历史推理上下文使用。

    参数：
        question: str - 子问题文本
        answer: str   - 该子问题的回答

    返回：
        str - 形如 "Question: xxx\nAnswer: yyy" 的字符串
              （末尾的换行被 strip() 清除，避免累积过多空行）

    设计意图：使用固定的 "Question:/Answer:" 前缀，
              让 LLM 能清晰识别这是历史问答对而非当前待答问题
    """
    return f"Question: {question}\nAnswer: {answer}\n\n".strip()


# 1.定义分解子问题的prompt
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：根据字符串模板创建提示模板，{question} 为变量占位符
#   参数：template - 多行字符串通过 Python 字面量拼接而成
#   返回：ChatPromptTemplate 实例（渲染后是单条 HumanMessage）
#   Prompt 设计要点：
#     ① 明确角色定位（乐于助人的 AI 助理）
#     ② 明确任务目标（拆分成可独立回答的子问题）
#     ③ 明确输出格式（换行符分割）与数量（3 个），
#        这一点至关重要，否则后续的 split("\n") 无法正确解析
decomposition_prompt = ChatPromptTemplate.from_template(
    "你是一个乐于助人的AI助理，可以针对一个输入问题生成多个相关的子问题。\n"
    "目标是将输入问题分解成一组可以独立回答的子问题或者子任务。\n"
    "生成与一下问题相关的多个搜索查询：{question}\n"
    "并使用换行符进行分割，输出（3个子问题/子查询）："
)

# 2.构建分解问题链
# LCEL 管道，四个环节依次处理：
#   环节1: {"question": RunnablePassthrough()}
#     作用：把外部传入的字符串原样包装成 {"question": "原字符串"} 的字典，
#           以满足 prompt 需要 dict 输入的要求
#     RunnablePassthrough() - 恒等 Runnable，输入即输出，不做任何变换
#   环节2: decomposition_prompt
#     作用：填充模板变量，输出 PromptValue（内含消息列表）
#   环节3: ChatOpenAI(model, temperature=0)
#     作用：调用大语言模型生成子问题
#     参数：temperature=0 - 关闭随机性，保证同一问题的分解结果稳定可复现
#     返回：AIMessage 对象
#   环节4: StrOutputParser()
#     作用：从 AIMessage 中提取 content 字段，输出纯字符串
#   环节5: (lambda x: x.strip().split("\n"))
#     作用：后处理，把多行文本切分成子问题列表
#     x.strip()       - 去掉首尾空白，防止产生空元素
#     x.split("\n")   - 按换行切分为 list[str]
#     说明：该 lambda 会被 LCEL 自动包装为 RunnableLambda
#     健壮性提示：若 LLM 输出带序号（"1. xxx"）或含空行，这里不会清理，
#                 生产环境应增加过滤与序号剥离逻辑
# 整体数据流转：str → dict → PromptValue → AIMessage → str → list[str]
decomposition_chain = (
        {"question": RunnablePassthrough()}
        | decomposition_prompt
        | ChatOpenAI(model="deepseek-v4-pro", temperature=0)
        | StrOutputParser()
        | (lambda x: x.strip().split("\n"))
)

# 3.构建向量数据库与检索器
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点
#         auth_credentials - AuthApiKey 包装的 API Key
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：包装已有 Collection 为 LangChain 向量库
#   参数：index_name="DatasetDemo" - 存放 LLMOps 文档的 Collection
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
#   作用：构建检索器，为每个子问题提供背景资料
#   参数：search_type="mmr" - 用最大边际相关性保证单个子问题召回内容的多样性
#   返回：VectorStoreRetriever 实例（默认 k=4）
retriever = db.as_retriever(search_type="mmr")

# 4.执行提问获取子问题
# 原始的复杂问题
question = "关于LLMOps应用配置的文档有哪些"

# decomposition_chain.invoke(input: str) -> list[str]
#   作用：调用分解链，把复杂问题拆解为子问题列表
#   参数：input - 原始问题字符串（由 RunnablePassthrough 包装为 dict）
#   返回：list[str]，3 个语义聚焦的子问题
#   预期输出示例：
#     ["LLMOps 平台的应用配置包含哪些配置项？",
#      "如何在 LLMOps 中修改应用配置？",
#      "LLMOps 应用配置相关的 API 接口有哪些？"]
sub_questions = decomposition_chain.invoke(question)

# 5.构建迭代问答链：提示模板+链
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：构建迭代问答的提示模板，包含三个变量占位符
#   三个变量的语义分工（本策略的核心设计）：
#     {question} - 当前正在回答的子问题
#     {qa_pairs} - 之前所有子问题的 QA 对累积文本，提供递进推理的历史上下文
#     {context}  - 针对当前子问题实时检索到的文档，提供 RAG 知识来源
#   Prompt 结构要点：用 --- 分隔线明确区分三部分内容，
#                     降低 LLM 混淆"待答问题"与"背景资料"的概率
prompt = ChatPromptTemplate.from_template("""这是你需要回答的问题：
---
{question}
---

这是所有可用的背景问题和答案对：
---
{qa_pairs}
---

这是与问题相关的额外背景信息：
---
{context}
---""")

# 构建迭代问答链
# 环节1: 并行字典（RunnableParallel），三个 key 同时求值
#   "question": itemgetter("question")
#     作用：从输入 dict 中原样抽取 question 字段
#     itemgetter("question") 等价于 lambda x: x["question"]
#   "qa_pairs": itemgetter("qa_pairs")
#     作用：从输入 dict 中原样抽取历史 QA 对文本
#   "context": itemgetter("question") | retriever
#     作用：这是一个嵌套子链——
#           先用 itemgetter 抽出 question 字符串（适配 retriever 的 str 输入），
#           再交给 retriever 检索，输出 list[Document]
#           list[Document] 被填入模板时会自动转为字符串表示
# 环节2: prompt        - 填充三个变量，生成消息列表
# 环节3: ChatOpenAI    - temperature=0，保证回答稳定
# 环节4: StrOutputParser - 提取纯文本答案
# 整体数据流转：
#   {"question": str, "qa_pairs": str}
#     → {"question": str, "qa_pairs": str, "context": list[Document]}
#     → PromptValue → AIMessage → str
chain = (
        {
            "question": itemgetter("question"),
            "qa_pairs": itemgetter("qa_pairs"),
            "context": itemgetter("question") | retriever,
        }
        | prompt
        | ChatOpenAI(model="deepseek-v4-pro", temperature=0)
        | StrOutputParser()
)

# 5.循环遍历所有子问题进行检索并获取答案
# 历史 QA 对累积容器，初始为空字符串（第一个子问题没有历史上下文）
qa_pairs = ""

# 串行递进处理每个子问题（Recursive Answer 模式）
for sub_question in sub_questions:
    # chain.invoke(input: dict) -> str
    #   作用：针对当前子问题执行"检索 + 生成"
    #   参数：input - 包含两个 key 的字典
    #         "question": 当前子问题
    #         "qa_pairs": 之前所有子问题的问答历史（首轮为空字符串）
    #   返回：LLM 生成的子问题答案
    #   内部执行：并行求值三路输入 → 其中 context 会触发一次真实的向量检索
    #             → 三路结果填入 Prompt → 调用 LLM → 解析为字符串
    answer = chain.invoke({"question": sub_question, "qa_pairs": qa_pairs})

    # 把当前的"子问题 + 答案"格式化为结构化文本
    qa_pair = format_qa_pair(sub_question, answer)

    # 累积到历史上下文中，供下一个子问题使用
    # 这是"串行递进"的关键：后面的子问题能看到前面所有的结论，
    # 从而避免重复回答，并能在已有结论基础上继续推理
    # 注意：qa_pairs 会持续变长，子问题很多时需关注上下文窗口上限
    qa_pairs += "\n---\n" + qa_pair

    # 打印当前子问题与答案，便于观察分解与递进推理的效果
    print(f"问题: {sub_question}")
    print(f"答案: {answer}")

# 说明：本示例到此结束，只输出了各个子问题的独立答案。
# 完整的问题分解策略还应再加一步"汇总链"：
#   把最终的 qa_pairs 与原始 question 一起交给 LLM，
#   生成对原始复杂问题的综合回答。
