#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 8:06
@Author  : thezehui@gmail.com
@File    : 3.最大边际相关性示例.py

===================================================================================
知识点讲解：MMR 最大边际相关性搜索
===================================================================================
1. 纯相似度搜索的"冗余"缺陷
   - similarity_search 只按相似度排序取 Top-K，完全不考虑结果之间的关系
   - 真实知识库中往往存在大量近似重复的段落（同一功能在不同章节反复描述、
     chunk_overlap 造成的重叠、多版本文档并存）
   - 结果就是 Top-4 可能都在讲同一件事，占满宝贵的 Prompt 上下文，
     其他角度的关键信息被挤出候选集，最终答案覆盖面很窄

2. MMR（Maximal Marginal Relevance）的核心思想
   - 每次选下一篇文档时，不只看"它和查询多像"，还要看"它和已选文档有多不像"
   - 评分公式：MMR = λ * sim(doc, query) - (1 - λ) * max sim(doc, selected_doc)
     前半部分是相关性收益，后半部分是与已选集合的冗余惩罚
   - 注意方向：得分 = lambda_mult * 相关性 - (1 - lambda_mult) * 相似性，
     因此 0 代表最大多样性、1 代表最小多样性（"最小多样性"= 最重相关性），
     这与很多人的直觉相反，不要记反
   - 迭代贪心选择：每轮挑 MMR 得分最高的文档加入结果集，直到凑满 k 条

3. lambda_mult 参数的调节含义（0 ~ 1）
   - λ = 1.0：完全忽略多样性，等价于普通的相似度搜索
   - λ = 0.5（默认）：相关性与多样性各占一半，适合大多数场景
   - λ = 0.0：极端追求差异化，结果可能跑偏、相关性很差
   - 实践经验：文档冗余度高就调低（0.3~0.5），要求高精度就调高（0.7~0.8）

4. fetch_k 候选池的作用（默认 20）
   - MMR 是"先粗排再重排"的两阶段算法
   - 第一阶段：按纯相似度从向量库取 fetch_k 条候选（真正的向量检索发生在这里）
   - 第二阶段：在这 fetch_k 条中本地做 MMR 迭代，挑出最终的 k 条
   - fetch_k 必须显著大于 k，否则"可挑选空间"太小，多样性无从体现；
     但 fetch_k 过大会增加本地相似度矩阵的计算量
   - 取值经验：向量数据库规模越大，fetch_k 设置越大，一般在 k 的 2~3 倍左右；
     若额外添加 filter 对数据做筛选（会削减候选池），可把 fetch_k 扩大到 k 的 4~6 倍

5. MMR 的适用与不适用场景
   - 适用：开放式问答、"有哪些…"类需要多角度覆盖的查询、文档冗余严重的知识库、
           摘要生成前的素材召集
   - 不适用：精确事实查询（只要最准的一条）、FAQ 精确匹配、
           对延迟极度敏感的场景（MMR 需要额外的向量两两比较）

6. 本例与纯相似度搜索的对照实验
   - 代码中保留了注释掉的 db.similarity_search(...) 一行
   - 建议交替放开两行对比输出：相似度搜索的结果内容更雷同，
     MMR 的结果覆盖的章节/主题更分散，这就是多样性的直观体现

7. max_marginal_relevance_search 完整参数表
   - query       ：搜索语句，字符串，必填
   - k           ：搜索结果条数，整型，默认 4
   - fetch_k     ：要传递给 MMR 算法的文档数，默认 20
   - lambda_mult ：函数系数，范围 0~1，得分 = λ*相关性 - (1-λ)*相似性
   - kwargs      ：其他传给搜索方法的参数，例如 filter，用法与相似性搜索类似，
                   具体取决于所使用的向量数据库
   - 一句话概括：MMR 就是在一大堆最相似的文档中查找最不相似的，从而保证结果多样化

8. 典型输出示例与观察点
   - 查询 "关于应用配置的接口有哪些？"，使用默认 k=4 / fetch_k=20 / λ=0.5，4 条结果
     （各取前 100 字符）依次为：
       ① '1.2 [todo]更新应用草稿配置信息…涵盖：模型配置、长记忆模式等…'
       ② 'LLMOps 项目 API 文档…应用 API 接口统一以 JSON 格式返回，包含 3 个字段…'
       ③ '如果接口需要授权，需要在 headers 中添加 Authorization…'
       ④ 'memory_mode -> string：记忆类型…status -> string…'
   - 观察要点：这 4 条分别落在"配置更新接口 / 文档总览 / 鉴权说明 / 配置字段定义"
     四个完全不同的章节，主题几乎不重叠——这就是 MMR 多样性的实际体现；
     若换成 similarity_search，很可能返回同一章节前后相邻的几个 chunk

9. 版本对照说明（示例基于 LangChain 1.x）
   - max_marginal_relevance_search 的参数签名在 0.x → 1.x 之间保持稳定
   - 本示例有意省略 add_documents(chunks) 调用，复用上一节已写入的 Collection 数据，
     避免脏数据——这是相对文档示例的改进
   - 文档的 separators 写成普通字符串（如 "。|！|？" 这类分隔符写法），
     本示例改为原始字符串，语义一致但避免 Python 转义告警

10. 最佳实践建议
   - "有哪些 / 列举 / 总结"类查询优先用 MMR；"某个具体参数是多少"类查询用相似度搜索
   - fetch_k 建议设为 k 的 4~5 倍（如 k=4 配 fetch_k=20），在效果与性能间取平衡
   - λ 先在真实语料上从 0.5 起步，按结果冗余程度上下微调
   - MMR 本身不带阈值过滤，若知识库无相关内容它照样返回 k 条；
     高要求场景可先用阈值检索做一次筛选，再对结果做 MMR 重排
   - 注意本示例未调用 add_documents，直接复用上一节已写入的 Collection 数据，
     避免重复入库产生脏数据——这是脚本型 Demo 的好习惯

===================================================================================
"""
import dotenv
import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 相关凭证
dotenv.load_dotenv()

# 1.构建加载器与分割器
# UnstructuredMarkdownLoader(file_path: str) -> UnstructuredMarkdownLoader
#   作用：创建 Markdown 文档加载器
#   参数：file_path - Markdown 文件路径
#   返回：加载器实例（此处仅构建，本示例后续虽然加载分割了但并未再次入库）
loader = UnstructuredMarkdownLoader("./项目API文档.md")

# RecursiveCharacterTextSplitter(...) -> RecursiveCharacterTextSplitter
#   作用：递归字符分割器，与上一节配置完全一致，保证 chunk 粒度可复现
#   关键参数：
#     separators         - 分隔符优先级列表（段落 → 换行 → 句末标点 → 分号 → 逗号 → 空格 → 字符）
#     is_separator_regex - True，分隔符按正则解析，支持 "。|！|？" 这类多标点匹配
#     chunk_size         - 每个 chunk 最大 500 字符
#     chunk_overlap      - 相邻 chunk 重叠 50 字符，防止语义在边界处被切断
#     add_start_index    - 在 metadata 中记录 chunk 在原文的起始偏移，便于溯源
#   补充：chunk_overlap 本身就是一种"人为制造的冗余"，这也正是后续需要 MMR 去重的原因之一
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", r"；|;\s", r"，|,\s", " ", "", ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# 2.加载文档并分割
# loader.load() -> list[Document]
#   作用：读取并解析 Markdown，返回原始 Document 列表
documents = loader.load()

# text_splitter.split_documents(documents: list[Document]) -> list[Document]
#   作用：把长文档切成多个 chunk，metadata 自动继承并追加 start_index
#   返回：chunk 列表
#   注意：本示例只做了分割，没有调用 add_documents，
#         因为目标 Collection（DatasetDemo）中的数据已在上一节写入，
#         这里直接复用，避免重复入库污染检索结果
chunks = text_splitter.split_documents(documents)

# 3.将数据存储到向量数据库
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 地址
#         auth_credentials - AuthApiKey 形式的 API Key 鉴权凭证
#   返回：Weaviate v4 客户端
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：把已有 Weaviate Collection 包装为 LangChain 向量库对象
#   参数：index_name - Collection 名称，此处指向已有数据的 "DatasetDemo"
#         text_key   - 正文字段名，检索时用于还原 page_content
#         embedding  - 嵌入模型，必须与入库时保持一致，否则向量空间错位
#   返回：向量库实例，可直接执行各类检索
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://eftofnujtxqcsa0sn272jw.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("21pzYy0orl2dxH9xCoZG1O2b0euDeKJNEbB0"),
    ),
    index_name="DatasetDemo",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# 4.执行最大边际相关性搜索
# 【对照实验】放开下面这行并注释掉 MMR 那行，可以观察纯相似度搜索的输出：
#   db.similarity_search(query: str, k: int = 4) -> list[Document]
#     作用：纯相似度 Top-K 检索，不做任何去重，结果可能高度雷同
# search_documents = db.similarity_search("关于应用配置的接口有哪些？")

# db.max_marginal_relevance_search(
#     query: str,
#     k: int = 4,
#     fetch_k: int = 20,
#     lambda_mult: float = 0.5,
#     **kwargs
# ) -> list[Document]
#   作用：执行最大边际相关性搜索，在相关性与多样性之间取得平衡
#   参数：query       - 查询语句
#         k           - 最终返回条数，默认 4
#         fetch_k     - 第一阶段候选池大小，默认 20（先按相似度粗排取这么多）
#         lambda_mult - 相关性/多样性权重，默认 0.5；越接近 1 越看重相关性
#   返回：k 条 Document，彼此内容差异较大
#   内部执行流程（数据流转）：
#     1) query_vector = embedding.embed_query(query)
#     2) 向量库按相似度取出 fetch_k=20 条候选，并把它们的向量一并返回
#     3) 先选出与 query 最相似的 1 条，放入 selected
#     4) 迭代：对剩余候选逐个计算
#          score = λ*sim(doc, query) - (1-λ)*max(sim(doc, s) for s in selected)
#        取 score 最大者加入 selected
#     5) 重复第 4 步直到 selected 满 k 条，返回结果
#   本例语义：查询是"有哪些接口"，属于典型的列举式问题，
#             用 MMR 能覆盖更多不同的接口章节，而不是反复返回同一个接口的相邻 chunk
search_documents = db.max_marginal_relevance_search("关于应用配置的接口有哪些？")

# 5.打印搜索的结果
# 下面这行是一次性打印所有结果的紧凑写法，可读性不如逐条打印，故注释保留作对比
# print(list(document.page_content[:100] for document in search_documents))

# 逐条打印每篇文档的前 100 个字符，并用分隔线隔开
# 观察要点：MMR 结果中相邻两条的开头内容应当明显不同；
#           若改用 similarity_search，多条结果的内容往往高度重叠
for document in search_documents:
    print(document.page_content[:100])
    print("===========")
