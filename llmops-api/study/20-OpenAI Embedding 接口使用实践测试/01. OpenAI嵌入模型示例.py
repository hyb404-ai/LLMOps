#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/25 11:17
@Author  : thezehui@gmail.com
@File    : 01. OpenAI嵌入模型示例.py

===================================================================================
知识点讲解：OpenAI Embeddings 文本嵌入模型
===================================================================================

1. 什么是 Embedding（文本嵌入）
   - Embedding 是把自然语言文本映射为一个固定长度的浮点数向量（vector）的过程
   - 语义相近的文本，其向量在高维空间中的方向也相近；语义无关的文本方向差异大
   - Embedding 是 RAG（检索增强生成）的基石：先把知识库向量化存入向量数据库，
     检索时把用户问题也向量化，再用向量相似度找出最相关的片段
   - Embedding 向量是一个 N 维的实值向量，通过学习可以更准确地表示对应特征的
     内在含义，使几何距离相近的向量对应的物体有相近的含义

2. Embedding 的核心价值（为什么需要它）
   - 降维：将高维数据（如词汇表规模可达数十万）映射到低维空间（几百到几千维），
     大大减少模型复杂度
   - 捕捉语义信息：语义上相近的词在向量空间中也会相近，保留并利用原始数据的
     重要信息
   - 适应性：通过数据驱动方式学习，能够自动适应数据特性，无需人工设计特征
   - 泛化能力：对训练数据中未出现的数据，Embedding 仍能给出合理表示
   - 可解释性：可通过 t-SNE 等可视化工具观察和理解 Embedding 的结构

3. LangChain 的 Embeddings 抽象接口
   - 所有嵌入模型都实现 langchain_core.embeddings.Embeddings 接口
   - 核心方法只有两个：
     · embed_query(text)        -> 嵌入单条「查询」文本，返回 list[float]
     · embed_documents(texts)   -> 批量嵌入「文档」列表，返回 list[list[float]]
   - 异步版本：aembed_query / aembed_documents
   - 之所以区分 query 与 documents，是因为部分模型（如 BGE、E5）对查询和文档
     需要添加不同的指令前缀，两个方法让上层调用保持统一
   - Embeddings 类不是 Runnable 组件，不能直接接入 LCEL 序列链，
     需要通过 RunnableLambda 函数进行转换

4. OpenAIEmbeddings 组件
   - LangChain 对 OpenAI /v1/embeddings 接口的封装，兼容 OpenAI 格式的第三方服务
   - 自动从环境变量读取 OPENAI_API_KEY 与 OPENAI_API_BASE
   - 常用模型与维度：
     · text-embedding-3-small  1536 维，性价比最高（本例使用）
     · text-embedding-3-large  3072 维，效果更好、成本更高
     · text-embedding-ada-002  1536 维，上一代模型，最长输入 8191 tokens
   - text-embedding-3 系列支持 dimensions 参数进行「维度裁剪」，
     例如 OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
     可在少量精度损失下显著降低存储与计算成本
   - OpenAI Embeddings 虽然效果是市面上最好的嵌入模型，且价格非常低廉，
     但由于没有本地版本，且接口响应速度相对较慢，在需要对大量文档进行嵌入时
     效率较低，国内一般会使用本地或国内服务提供商的嵌入模型

5. 余弦相似度（Cosine Similarity）
   - 公式：cos(θ) = (A · B) / (||A|| × ||B||)
   - 只关注向量方向、忽略向量长度，取值范围 [-1, 1]，越接近 1 表示语义越相近
   - OpenAI 的 embedding 向量默认已做 L2 归一化（模长为 1），
     因此余弦相似度在数值上等价于点积，也与欧几里得距离单调相关

6. 最佳实践建议
   - 批量调用：优先用 embed_documents 一次提交多条文本，减少 HTTP 往返与限流风险
   - 模型一致性：写入向量库与检索时必须使用同一个嵌入模型，
     不同模型（甚至同模型不同维度）产生的向量空间不可混用
   - 成本控制：嵌入结果建议缓存（参见第 21 课 CacheBackedEmbeddings），
     OpenAI Embeddings 按 token 计费，重复文本会导致无用功
   - 文本长度：单条文本不要超过模型的 max tokens（8191），过长需先做文档分割

===================================================================================
"""
import dotenv
import numpy as np
from langchain_openai import OpenAIEmbeddings
from numpy.linalg import norm

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()


def cosine_similarity(vec1: list, vec2: list) -> float:
    """计算传入两个向量的余弦相似度

    :param vec1: 第一个向量（list[float]）
    :param vec2: 第二个向量（list[float]），维度必须与 vec1 一致
    :return: 余弦相似度，取值范围 [-1, 1]，越接近 1 表示两段文本语义越相近
    """
    # 1.计算两个向量的点积
    # np.dot(vec1, vec2)：逐元素相乘再求和，得到标量，对应公式中的 A · B
    dot_product = np.dot(vec1, vec2)

    # 2.计算向量的长度
    # numpy.linalg.norm(vec)：默认计算 L2 范数（欧几里得长度），即 sqrt(Σ xi²)
    vec1_norm = norm(vec1)
    vec2_norm = norm(vec2)

    # 3.计算余弦相似度
    # 点积除以两个模长的乘积，消除向量长度影响，只保留方向上的夹角信息
    return dot_product / (vec1_norm * vec2_norm)


# 1.创建文本嵌入模型
# OpenAIEmbeddings 关键参数说明：
#   model      : 嵌入模型名称，text-embedding-3-small 输出 1536 维向量
#   dimensions : 可选，输出维度裁剪（仅 text-embedding-3 系列支持）
#   chunk_size : 可选，单次请求最多提交的文本条数，默认 1000，超出会自动分批
#   api_key / base_url : 可选，不传则回落到环境变量
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.嵌入文本
# embed_query(text) 作用：把「单条查询文本」转换为向量
#   参数：text -> str，待嵌入的文本
#   返回：list[float]，长度等于模型维度（此处为 1536）
#   底层：发起一次 POST /v1/embeddings 请求，input 为单个字符串
query_vector = embeddings.embed_query("我叫慕小课，我喜欢打篮球")

# 打印完整向量：由 1536 个浮点数组成的列表
print(query_vector)
# 打印向量维度，用于确认模型输出规格（text-embedding-3-small => 1536）
print(len(query_vector))

# 3.嵌入文档列表/字符串列表
# embed_documents(texts) 作用：批量把「多条文档文本」转换为向量
#   参数：texts -> list[str]，待嵌入的文本列表
#   返回：list[list[float]]，外层长度 = 输入条数，内层长度 = 向量维度
#   优势：一次 HTTP 请求完成多条嵌入，比循环调用 embed_query 更快、更省配额
#   注意：返回结果的顺序与输入列表严格一一对应
documents_vector = embeddings.embed_documents([
    "我叫慕小课，我喜欢打篮球",
    "这个喜欢打篮球的人叫慕小课",
    "求知若渴，虚心若愚"
])
# 打印外层长度，即成功嵌入的文档条数（此处为 3）
print(len(documents_vector))

# 4.计算余弦相似度
# 向量1 与 向量2：两句话表达的是同一件事（慕小课喜欢打篮球），只是语序不同
#   => 相似度应显著偏高（通常 > 0.8）
print("向量1和向量2的相似度:", cosine_similarity(documents_vector[0], documents_vector[1]))

# 向量1 与 向量3：一句是自我介绍，一句是格言，语义完全无关
#   => 相似度应明显偏低（通常 < 0.3）
# 这组对比直观体现了 embedding「语义相近则向量相近」的核心特性，
# 也是向量数据库能够做语义检索（而非关键词匹配）的根本原因
print("向量1和向量3的相似度:", cosine_similarity(documents_vector[0], documents_vector[2]))
