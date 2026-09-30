#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/8/7 11:18
@Author  : thezehui@gmail.com
@File    : 16.RAPTOR递归文档树优化策略.py

===================================================================================
知识点讲解：RAPTOR 递归文档树检索优化
===================================================================================

1. 什么是 RAPTOR（定位与来源）
   - RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval，2024 年论文（arXiv 2401.18059）首次提出
   - 用树状结构组织检索的递归抽象处理技术：自下而上对文本块聚类、归纳，形成分层结构
   - 针对长上下文场景：传统 RAG 依赖短连续文本块检索，长上下文需要更好的最小化分块方法
   - 目标是在「多文档、超长上下文、高准确性、超低成本」之间取得平衡

2. 递归构建流程（六步）
   - ① 对原始文本分块，拆成合适大小
   - ② 对文档块嵌入/向量化，存入向量数据库（此时向量处于高维）
   - ③ 将高维向量降维（如降到 2/3 维），降低运算成本
   - ④ 对降维向量聚类，找出同一类的文档组
   - ⑤ 合并文档组文本，用 LLM 摘要汇总得到新文本，然后重复 ②–⑤
   - ⑥ 直到只剩一个文档且长度合适，结束整个流程
   - 本质：叶子是原文块，逐层向上聚类生成父节点（摘要），形成多层抽象

3. 两种检索策略
   - 树遍历（tree traversal）：从根节点逐层向下，每层按查询与节点嵌入的余弦相似度选最相关 top-k，把其子节点作为下一层候选，直到叶节点，拼接所选节点文本
   - 折叠树（collapsed tree）：把整棵树展平为单层，计算查询与所有节点嵌入的余弦相似度选 top-k
   - 论文更推荐折叠树；且折叠树与「向量数据库基础搜索」一样遍历所有向量（同层），
     只需把全部节点存入向量库并执行相似性搜索即可实现

4. 技术栈（降维 + 聚类）
   - UMAP：全局降维，保留向量的局部与全局结构，作为聚类预处理
   - GMM（高斯混合模型）：概率聚类，支持软分配（一个样本可属多个簇）
   - BIC（贝叶斯信息准则）：自动确定最优聚类数

5. 与多向量/表征索引的关系
   - 同属索引阶段「多向量/多表征」思路的延伸，但 RAPTOR 用树状结构的层层摘要
   - 区别在于：多向量是每个块生成若干派生文本（摘要/假设问题），RAPTOR 进一步把派生文本再聚类、再摘要，形成层次

6. 优势
   - 多粒度：同时提供细节（叶子/原文）与宏观视角（各层摘要），既能答具体问题也能答概念性/宏观问题
   - 自适应层次：自动构建合适的树结构
   - 小嵌入模型也能有不错效果：低成本、高性能
   - 通过摘要过滤冗余，减少噪声

7. 局限与成本
   - 构建成本高：每层都要用 LLM 摘要，适合离线构建
   - 更新/新增数据麻烦：树状结构使增删比其它策略复杂得多，外挂文档增加时复杂度成倍递增
   - 与多向量索引一样，额外 LLM 总结会引入信息失真，数据越多失真概率越大（谨慎使用）

8. 数据更新策略（论文未提供，常规做法）
   - 重新构建：新增大量文档时重建整棵树
   - 增量更新：新增不多时只更新部分树，把新块作新叶子，调整上层摘要
   - 利用现有节点：新文档与某些节点相似时，合并进现有节点并重新生成摘要
   - 层次化更新：按新文档重要性决定在树的哪一层更新

9. LangChain 现状与学习建议
   - LangChain 没有内置 RAPTOR 实现；该策略涵盖降维、聚类、递归检索，目前使用较少
   - 重点是理解其运行流程，无需深入实现细节

10. 索引阶段优化策略全景
   - 代表性策略共四类：分割策略（语义分割、递归字符分割）、多向量/表征检索、RAPTOR、特定 Embeddings（用更高维嵌入模型，无复杂技巧）
   - 实际用得最多的仍是「文档加载/分割」与「特定 Embeddings」两类不引入 LLM 改写的策略

11. 应用场景与最佳实践
   - 适用：长文档检索（书籍、报告、论文）、需理解整体结构的场景、既要细节又要概览的问答
   - 层数通常 2–3 层即可，过多会损失细节；聚类参数按文档数量调整；确保摘要保留关键信息

===================================================================================
"""
from typing import Optional

import dotenv
import numpy as np
import pandas
import pandas as pd
import umap
import weaviate
from langchain_unstructured import UnstructuredLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from sklearn.mixture import GaussianMixture
from weaviate.auth import AuthApiKey

# 加载环境变量配置
dotenv.load_dotenv()

# 1.定义随机数种子、文本嵌入模型、大语言模型、向量数据库
# RANDOM_SEED：固定随机种子，确保结果可复现
RANDOM_SEED = 224

# HuggingFaceEmbeddings：使用 HuggingFace 的嵌入模型
# - model_name：gte-small 是一个轻量级但效果好的嵌入模型
# - cache_folder：本地缓存目录，避免重复下载
# - encode_kwargs：编码参数，normalize_embeddings=True 归一化向量
embd = HuggingFaceEmbeddings(
    model_name="thenlper/gte-small",
    cache_folder="./embeddings/",
    encode_kwargs={"normalize_embeddings": True},
)

# ChatOpenAI：用于生成摘要的大语言模型
model = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# WeaviateVectorStore：用于存储文档树的向量数据库
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://mbakeruerziae6psyex7ng.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("ZltPVa9ZSOxUcfafelsggGyyH6tnTYQYJvBx"),
    ),
    index_name="RaptorRAG",
    text_key="text",
    embedding=embd,
)


def global_cluster_embeddings(
        embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine",
) -> np.ndarray:
    """
    使用 UMAP 对传递嵌入向量进行全局降维

    UMAP（Uniform Manifold Approximation and Projection）是一种降维算法，
    它在保留数据的局部和全局结构方面表现优异，特别适合作为聚类的预处理步骤。

    Args:
        embeddings: 需要降维的嵌入向量数组，shape 为 (n_samples, n_features)
        dim: 降低后的目标维度，通常设置为 10-50
        n_neighbors: 每个向量需要考虑的邻居数量，影响局部结构的保留程度
                     如果没有提供，默认为嵌入数量的开方
        metric: 用于 UMAP 的距离度量方式，默认为余弦相似性

    Returns:
        降维到指定维度的 numpy 嵌入数组，shape 为 (n_samples, dim)

    说明：
        全局聚类用于第一轮聚类，使用较少的邻居数，保留全局结构
    """
    if n_neighbors is None:
        # 默认邻居数为样本数的平方根
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    # 创建 UMAP 降维器并执行降维
    # fit_transform：训练模型并转换数据
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, n_neighbors: int = 10, metric: str = "cosine",
) -> np.ndarray:
    """
    使用 UMAP 对嵌入进行局部降维处理，通常在全局聚类之后进行

    局部降维使用更多的邻居数，更注重保留局部结构，
    适合在已经完成全局聚类后，对每个全局聚类内部进行细粒度聚类。

    Args:
        embeddings: 需要降维的嵌入向量数组
        dim: 降低后的目标维度
        n_neighbors: 每个向量需要考虑的邻居数量，默认为 10
        metric: 用于 UMAP 的距离度量方式，默认为余弦相似性

    Returns:
        降维到指定维度的 numpy 嵌入数组

    说明：
        局部聚类用于第二轮聚类，使用固定的邻居数，关注局部结构
    """
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric,
    ).fit_transform(embeddings)


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED,
) -> int:
    """
    使用高斯混合模型结合贝叶斯信息准则（BIC）确定最佳的聚类数目

    BIC 是一种模型选择标准，它在模型拟合度和复杂度之间取得平衡。
    BIC 值越小，表示模型越好。通过比较不同聚类数的 BIC 值，
    可以自动找到最优的聚类数，避免手动调参。

    Args:
        embeddings: 需要聚类的嵌入向量数组
        max_clusters: 最大聚类数，用于限制搜索范围
        random_state: 随机数种子，确保结果可复现

    Returns:
        最优聚类数（使 BIC 最小的聚类数）

    原理：
        - BIC = -2 * log-likelihood + k * log(n)
        - k 是参数数量，n 是样本数量
        - BIC 惩罚复杂模型，避免过拟合
    """
    # 1.获取最大聚类数，最大聚类数不能超过嵌入向量的数量
    max_clusters = min(max_clusters, len(embeddings))
    # 生成候选聚类数列表 [1, 2, 3, ..., max_clusters-1]
    n_clusters = np.arange(1, max_clusters)

    # 2.逐个设置聚类数并找出最优聚类数
    bics = []
    for n in n_clusters:
        # 3.创建高斯混合模型，并计算聚类结果
        # GaussianMixture：高斯混合模型，假设数据由多个高斯分布混合而成
        # - n_components：聚类数（高斯分量数）
        # - random_state：随机种子
        gm = GaussianMixture(n_components=n, random_state=random_state)
        # fit：训练模型，估计高斯分布的参数
        gm.fit(embeddings)
        # bic：计算贝叶斯信息准则
        bics.append(gm.bic(embeddings))

    # 返回使 BIC 最小的聚类数
    return n_clusters[np.argmin(bics)]


def gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0) -> tuple[list, int]:
    """
    使用基于概率阈值的高斯混合模型（GMM）对嵌入进行聚类

    与传统的 K-Means 等硬聚类不同，GMM 是软聚类，每个样本可以属于多个聚类。
    通过设置概率阈值，可以让一个文档同时属于多个聚类，
    这在文档涉及多个主题时特别有用。

    Args:
        embeddings: 需要聚类的嵌入向量数组（已降维）
        threshold: 概率阈值，样本属于某聚类的概率超过此值时才分配到该聚类
                   较低的阈值（如 0.1）允许更多的软分配
        random_state: 用于可重现的随机性种子

    Returns:
        包含聚类标签和聚类数目的元组
        - labels：每个样本的聚类标签列表，每个元素是一个数组（软聚类）
        - n_clusters：确定的聚类数目

    示例：
        如果 threshold=0.1，某文档对聚类 0 的概率为 0.7，对聚类 1 的概率为 0.2，
        那么该文档会被分配到 [0, 1] 两个聚类
    """
    # 1.获取最优聚类数
    n_clusters = get_optimal_clusters(embeddings)

    # 2.创建高斯混合模型对象并拟合数据
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)

    # 3.预测每个样本属于各个聚类的概率
    # predict_proba：返回概率矩阵，shape 为 (n_samples, n_clusters)
    # 每一行表示一个样本属于各聚类的概率，行和为 1
    probs = gm.predict_proba(embeddings)

    # 4.根据概率阈值确定每个嵌入的聚类标签
    # np.where(prob > threshold)[0]：找出概率超过阈值的聚类索引
    # 这样一个样本可能属于多个聚类（软聚类）
    labels = [np.where(prob > threshold)[0] for prob in probs]

    # 5.返回聚类标签和聚类数目
    return labels, n_clusters


def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> list[np.ndarray]:
    """
    对嵌入进行聚类，首先全局降维，然后使用高斯混合模型进行聚类，
    最后在每个全局聚类中进行局部聚类

    这是一个两级聚类策略：
    1. 全局聚类：将所有文档分成大类（如"技术文档"、"产品介绍"等）
    2. 局部聚类：在每个大类内部进一步细分（如"API文档"、"配置文档"等）

    Args:
        embeddings: 需要执行操作的嵌入向量列表
        dim: 指定的降维维度，通常设置为 10
        threshold: 概率阈值，用于 GMM 软聚类

    Returns:
        包含每个嵌入的聚类 ID 的列表，每个数组代表一个嵌入的聚类标签

    示例：
        如果某文档在全局聚类中属于聚类 0 和 1，在聚类 0 内部属于局部聚类 0，
        在聚类 1 内部属于局部聚类 1，那么最终标签可能是 [0, 3]
        （假设聚类 0 有 3 个局部聚类）
    """
    # 1.检测传入的嵌入向量，当数据量不足时不进行聚类
    # 如果样本数太少（<= dim + 1），聚类没有意义，直接分配到聚类 0
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # 2.调用函数进行全局降维
    # 将高维向量降到 dim 维，便于聚类
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)

    # 3.对降维后的数据进行全局聚类
    # 得到全局聚类标签和全局聚类数
    global_clusters, n_global_clusters = gmm_cluster(reduced_embeddings_global, threshold)

    # 4.初始化一个空列表，用于存储所有嵌入的局部聚类标签
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    # 记录当前已分配的聚类 ID 总数
    total_clusters = 0

    # 5.遍历每个全局聚类以执行局部聚类
    for i in range(n_global_clusters):
        # 6.提取属于当前全局聚类的嵌入向量
        # [i in gc for gc in global_clusters]：检查每个样本是否属于全局聚类 i
        # 注意这里是软聚类，一个样本可能属于多个全局聚类
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        # 7.如果当前全局聚类中没有嵌入向量则跳过循环
        if len(global_cluster_embeddings_) == 0:
            continue

        # 8.如果当前全局聚类中的嵌入量很少，直接将它们分配到一个聚类中
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 9.执行局部降维和聚类
            # 对全局聚类内部进行更细粒度的聚类
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = gmm_cluster(reduced_embeddings_local, threshold)

        # 10.分配局部聚类 ID，调整已处理的总聚类数目
        # 为了避免不同全局聚类的局部聚类 ID 冲突，加上 total_clusters 偏移
        for j in range(n_local_clusters):
            # 提取属于当前局部聚类的嵌入向量
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            # 找到这些嵌入在原始嵌入列表中的索引
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            # 为这些样本分配全局唯一的聚类 ID
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        # 更新已分配的聚类 ID 总数
        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts: list[str]) -> np.ndarray:
    """
    将传递的文本列表转换成嵌入向量列表

    Args:
        texts: 需要转换的文本列表

    Returns:
        生成的嵌入向量列表并转换成 numpy 数组

    说明：
        使用全局定义的 embd 嵌入模型（HuggingFace gte-small）
    """
    # embed_documents：批量将文本转换为嵌入向量
    text_embeddings = embd.embed_documents(texts)
    # 转换为 numpy 数组，便于后续的数值计算
    return np.array(text_embeddings)


def embed_cluster_texts(texts: list[str]) -> pandas.DataFrame:
    """
    对文本列表进行嵌入和聚类，并返回一个包含文本、嵌入和聚类标签的数据框
    该函数将嵌入生成和聚类结合成一个步骤

    Args:
        texts: 需要处理的文本列表

    Returns:
        返回包含文本、嵌入和聚类标签的数据框，包含三列：
        - text：原始文本
        - embd：嵌入向量
        - cluster：聚类标签（可能是多个）

    说明：
        这个函数是 RAPTOR 流程的核心步骤之一
    """
    # 将文本转换为嵌入向量
    text_embeddings_np = embed(texts)
    # 对嵌入向量进行两级聚类
    # dim=10：降维到 10 维
    # threshold=0.1：概率阈值 0.1，允许软聚类
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)

    # 构建数据框
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    将数据框中的文本格式化成单个字符串

    Args:
        df: 需要处理的数据框，内部包含 text、embd、cluster 三个字段

    Returns:
        返回合并格式化后的字符串

    说明：
        将同一聚类的文本合并成一个字符串，用于生成聚类摘要
    """
    # 提取所有文本
    unique_txt = df["text"].tolist()
    # 使用分隔符连接所有文本
    return "--- --- \n --- ---".join(unique_txt)


def embed_cluster_summarize_texts(texts: list[str], level: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对传入的文本列表进行嵌入、聚类和总结
    该函数首先为文本生成嵌入，基于相似性对它们进行聚类，
    扩展聚类分配以便处理，然后总结每个聚类中的内容

    Args:
        texts: 需要处理的文本列表
        level: 一个整数，表示当前处理的层级（用于追踪递归深度）

    Returns:
        包含两个数据框的元组：
        - 第一个 DataFrame (df_clusters)：包括原始文本、它们的嵌入以及聚类分配
        - 第二个 DataFrame (df_summary)：包含每个聚类的摘要信息、指定的处理级别以及聚类标识符

    说明：
        这是 RAPTOR 每一层处理的核心函数
    """
    # 1.嵌入和聚类文本，生成包含 text、embd、cluster 的数据框
    df_clusters = embed_cluster_texts(texts)

    # 2.定义变量，用于扩展数据框，以便更方便地操作聚类
    # 因为每个文本可能属于多个聚类（软聚类），需要展开
    expanded_list = []

    # 3.扩展数据框条目，将文档和聚类配对，便于处理
    for index, row in df_clusters.iterrows():
        # 为每个文本-聚类对创建一行
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # 4.从扩展列表创建一个新的数据框
    expanded_df = pd.DataFrame(expanded_list)

    # 5.获取唯一的聚类标识符以进行处理
    all_clusters = expanded_df["cluster"].unique()

    # 6.创建汇总 Prompt、汇总链
    # 这个 Prompt 要求模型为每个聚类的文档生成详细摘要
    template = """Here is a sub-set of LangChain Expression Language doc. 

    LangChain Expression Language provides a way to compose chain in LangChain.

    Give a detailed summary of the documentation provided.

    Documentation:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # 构建摘要生成链
    chain = prompt | model | StrOutputParser()

    # 7.格式化每个聚类中的文本以进行总结
    summaries = []
    for i in all_clusters:
        # 提取属于当前聚类的所有文本
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        # 格式化成单个字符串
        formatted_txt = fmt_txt(df_cluster)
        # 生成摘要
        summaries.append(chain.invoke({"context": formatted_txt}))

    # 8.创建一个 DataFrame 来存储总结及其对应的聚类和级别
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,  # 摘要内容
            "level": [level] * len(summaries),  # 当前层级
            "cluster": list(all_clusters),  # 聚类 ID
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
        texts: list[str], level: int = 1, n_levels: int = 3,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    递归地嵌入、聚类和总结文本，直到达到指定的级别或唯一聚类数变为 1，
    将结果存储在每个级别处

    这是 RAPTOR 算法的核心递归函数，自底向上构建文档树。

    Args:
        texts: 要处理的文本列表（初始调用时是叶子节点）
        level: 当前递归级别（从 1 开始）
        n_levels: 递归的最大深度（默认为 3）

    Returns:
        一个字典，其中键是递归级别，值是包含该级别处聚类 DataFrame 和总结 DataFrame 的元组

    工作流程：
        Level 1: 原始文档块 -> 聚类 -> 生成摘要
        Level 2: Level 1 的摘要 -> 聚类 -> 生成更高层次的摘要
        Level 3: Level 2 的摘要 -> 聚类 -> 生成最高层次的摘要

    终止条件：
        1. 达到最大层数 n_levels
        2. 只剩下一个聚类（无法继续细分）
    """
    # 1.定义字典用于存储每个级别处的结果
    results = {}

    # 2.对当前级别执行嵌入、聚类和总结
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # 3.存储当前级别的结果
    results[level] = (df_clusters, df_summary)

    # 4.确定是否可以继续递归并且有意义
    # 获取当前层的唯一聚类数
    unique_clusters = df_summary["cluster"].nunique()
    # 如果还没达到最大层数，且聚类数大于 1（还能继续细分）
    if level < n_levels and unique_clusters > 1:
        # 5.使用总结作为下一级递归的输入文本
        # 这是关键：下一层的输入是当前层的摘要，而不是原始文本
        new_texts = df_summary["summaries"].tolist()
        # 递归调用，处理下一层
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # 6.将下一级的结果合并到当前结果字典中
        results.update(next_level_results)

    return results


# 2.定义文档加载器、文本分割器(中英文场景)
# 加载多个不同类型的文档
loaders = [
    UnstructuredLoader("./流浪地球.txt"),  # 中文小说
    UnstructuredLoader("./电商产品数据.txt"),  # 中文产品描述
    UnstructuredLoader("./项目API文档.md"),  # 技术文档
]

# RecursiveCharacterTextSplitter：递归字符文本分割器
# 特别配置了中英文分隔符，支持中英文混合文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个块约 500 字符
    chunk_overlap=0,  # 不重叠，因为 RAPTOR 通过摘要提供上下文
    # separators：分隔符列表，按优先级递归尝试
    # 中文标点和英文标点都包含
    separators=["\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", r"；|;\s", r"，|,\s", " ", ""],
    is_separator_regex=True,  # 使用正则表达式匹配分隔符
)

# 3.循环分割并加载文本
docs = []
for loader in loaders:
    # load_and_split：加载并立即分割
    docs.extend(loader.load_and_split(text_splitter))

# 4.构建文档树，最多 3 层
# 提取所有文档的文本内容作为叶子节点
leaf_texts = [doc.page_content for doc in docs]

# 递归构建文档树
# level=1：从第一层开始
# n_levels=3：构建 3 层树
# 返回的 results 包含每一层的聚类和摘要信息
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

# 5.遍历文档树结果，从每个级别提取总结并将它们添加到 all_texts 中
# 初始化：包含所有叶子节点（原始文档块）
all_texts = leaf_texts.copy()

# 遍历每一层，提取摘要
for level in sorted(results.keys()):
    # results[level][1] 是 df_summary
    summaries = results[level][1]["summaries"].tolist()
    # 将当前层的摘要添加到总文本列表
    all_texts.extend(summaries)

# 6.将 all_texts 添加到向量数据库
# all_texts 现在包含：
# - 所有原始文档块（叶子节点）
# - Level 1 的摘要（第一层父节点）
# - Level 2 的摘要（第二层父节点）
# - Level 3 的摘要（顶层节点）
# 检索时可以匹配到任意层次的内容
db.add_texts(all_texts)

# 7.执行相似性检索（折叠树）
# as_retriever：将向量数据库转换为检索器
# search_type="mmr"：使用最大边际相关性，平衡相似度和多样性
retriever = db.as_retriever(search_type="mmr")

# 执行检索
# 这个问题可能会匹配到：
# - 原始文档中的细节描述
# - 第一层摘要中的章节总结
# - 更高层摘要中的宏观信息
search_docs = retriever.invoke("流浪地球中的人类花了多长时间才流浪到新的恒星系？")

print(search_docs)
print(len(search_docs))

# RAPTOR 的优势：
# 1. 既能找到细节（如"2500年"），也能找到背景（如"流浪地球计划的目标"）
# 2. 不同层次的摘要提供不同粒度的上下文
# 3. 特别适合需要宏观理解的复杂问题
#
# 适用场景：
# - 长篇文档问答
# - 需要理解文档整体结构的任务
# - 既需要概览又需要细节的场景

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
