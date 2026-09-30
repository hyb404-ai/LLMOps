#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/30 8:12
@Author  : thezehui@gmail.com
@File    : 1.对接自定义向量数据库示例.py

===================================================================================
知识点讲解：自定义 VectorStore 对接 LangChain 生态
===================================================================================

1. 为什么需要自定义 VectorStore
   - LangChain 内置了 Faiss、Pinecone、Weaviate、TCVectorDB 等数十种向量库封装
   - 但实际项目中仍可能遇到：公司自研向量引擎、特殊存储介质（Redis/内存/文件）、需要定制打分逻辑或多租户隔离策略等场景
   - 只要继承 VectorStore 并实现约定的抽象方法，自定义实现即可无缝接入 Retriever、RAG 链、MultiVectorRetriever 等上层组件

2. 自定义向量数据库的两种实现模式
   - 模式一：继承已封装好的数据库类，重写部分方法进行扩展或修复 bug（适用于官方封装有缺陷时）
   - 模式二：直接继承抽象基类 VectorStore，对接全新的向量数据库（本示例采用）
   - 两者本质都是把向量库的「写入 / 检索」能力集成到约定的方法名下，从而被 LangChain 生态识别

3. VectorStore 抽象基类的契约
   - 必须实现的三个方法：
     * add_texts()        : 写入数据，返回每条记录的 id 列表
     * similarity_search(): 相似性检索，返回 List[Document]
     * from_texts()       : 类方法工厂，从原始文本一步构建实例
   - 其他方法使用频率不高，VectorStore 未把它们设为抽象方法，但「没有实现就直接调用会报错」，涵盖：
     * delete()、_select_relevance_score_fn()、similarity_search_with_score()
     * similarity_search_by_vector()、max_marginal_relevance_search()、max_marginal_relevance_search_by_vector()
   - 只实现必须的三个方法，就能获得 as_retriever() 等一整套能力（模板方法模式）

4. Embeddings 接口的两个核心方法
   - embed_documents(texts: List[str]) -> List[List[float]]：批量向量化「文档」，用于写入阶段
   - embed_query(text: str) -> List[float]：向量化单条「查询」，用于检索阶段
   - 区分 document 与 query 是因为部分嵌入模型（如 BGE、E5）采用非对称编码，对查询和文档使用不同指令前缀以提升检索效果

5. 向量相似度的度量方式
   - 欧几里得距离（L2，本例采用）：衡量向量空间中的直线距离，值域 [0, +∞)，越小越相似；公式 sqrt(Σ(a_i - b_i)²)，numpy 实现为 np.linalg.norm(a - b)
   - 余弦相似度（Cosine）：衡量向量夹角，值域 [-1, 1]，越大越相似；对向量长度不敏感，NLP 最常用
   - 内积（Dot Product / IP）：等价于未归一化的余弦，向量已 L2 归一化时与余弦等价
   - 重要结论：若向量已做 L2 归一化，欧几里得距离与余弦相似度单调等价，排序结果完全一致（OpenAI embedding 默认已归一化，故本例排序有效）

6. 本例的 score 语义陷阱
   - 本实现把「欧几里得距离」直接写入 metadata["score"]
   - 距离越小越相似，而 LangChain 生态中 score 通常约定为「越大越相似」
   - 若要接入 similarity_score_threshold 检索模式，需转换为相似度：similarity = 1 / (1 + distance) 或 1 - distance / max_distance
   - 生产实现应统一 score 语义，避免上层过滤逻辑反向筛选

7. store 定义为类属性的隐患
   - 本例 store: dict = {} 是「类属性」而非实例属性
   - 后果：所有 MemoryVectorStore 实例共享同一份数据，多实例场景会互相污染
   - 正确写法应在 __init__ 中初始化：self.store = {}
   - 这是 Python 可变默认值 / 类属性的经典陷阱，学习时需特别注意

8. 生产级自定义 VectorStore 的补充考量
   - 持久化：内存实现进程退出即丢失，需落盘或对接外部存储
   - 检索性能：本例为 O(n) 全量暴力扫描，数据量大时需引入 HNSW/IVF 等索引
   - 元数据过滤：应支持 filter 参数，在检索阶段做多租户 / 权限隔离
   - 并发安全：多线程写入需加锁，或使用线程安全的数据结构
   - 批量优化：add_texts 应支持分批调用嵌入 API，避免单次请求过大超限

===================================================================================
"""
import uuid
from typing import List, Optional, Any, Iterable, Type

import dotenv
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings


class MemoryVectorStore(VectorStore):
    """基于内存+欧几里得距离的向量数据库

    继承 langchain_core.vectorstores.VectorStore 抽象基类，
    实现 add_texts / similarity_search / from_texts 三个必要方法后，
    即可作为标准 VectorStore 接入 LangChain 的 Retriever 与 RAG 生态。

    注意：store 声明为类属性，所有实例共享同一份数据（见知识点 6）。
    """
    store: dict = {}  # 存储向量的临时变量

    def __init__(self, embedding: Embeddings):
        """初始化向量数据库

        参数：
            embedding: Embeddings - 嵌入模型实例，需实现 embed_documents 与 embed_query
                       写入时用 embed_documents 批量向量化，检索时用 embed_query 向量化查询

        说明：
            以单下划线 _embedding 保存，表示内部私有属性，不对外暴露
        """
        self._embedding = embedding

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        """将数据添加到向量数据库中

        VectorStore 抽象方法之一，负责数据写入。

        参数：
            texts: Iterable[str]                - 待写入的文本列表
            metadatas: Optional[List[dict]]     - 与 texts 一一对应的元数据列表，可为 None
                                                  元数据用于后续的过滤检索与来源溯源
            **kwargs: Any                       - 预留的扩展参数（如 namespace、batch_size）

        返回：
            List[str] - 每条记录的唯一 id 列表，顺序与 texts 一致，可用于后续更新/删除

        异常：
            ValueError - 当 metadatas 长度与 texts 不一致时抛出
        """
        # 1.检测metadata的数据格式
        # 元数据必须与文本严格一一对应，否则会出现数据错位
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas格式错误")

        # 2.将数据转换成文本嵌入/向量和ids
        # Embeddings.embed_documents(texts: List[str]) -> List[List[float]]
        #   作用：批量将文档文本向量化
        #   参数：texts - 文本列表
        #   返回：二维浮点数组，embeddings[i] 对应 texts[i] 的向量
        #   说明：批量调用比逐条调用大幅节省 API 往返开销
        embeddings = self._embedding.embed_documents(texts)

        # uuid.uuid4() -> UUID
        #   作用：生成随机的 128 位唯一标识符，转为 str 作为记录主键
        #   为每条文本生成独立 id，避免内容相同导致的覆盖
        ids = [str(uuid.uuid4()) for _ in texts]

        # 3.通过for循环组装数据记录
        # 以 id 为键存入字典，实现 O(1) 的按 id 读取
        for idx, text in enumerate(texts):
            self.store[ids[idx]] = {
                "id": ids[idx],            # 记录主键
                "text": text,              # 原始文本，检索时作为 page_content 返回
                "vector": embeddings[idx],  # 文本对应的向量，用于距离计算
                # 元数据缺省时填充空字典，保证下游访问不报 KeyError
                "metadata": metadatas[idx] if metadatas is not None else {},
            }

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """传入对应的query执行相似性搜索

        VectorStore 抽象方法之一，负责相似性检索。
        实现该方法后，基类的 as_retriever() 即可自动工作。

        参数：
            query: str      - 用户的查询文本
            k: int          - 返回最相似的前 k 条结果，默认 4
            **kwargs: Any   - 预留扩展参数（如 filter 元数据过滤条件）

        返回：
            List[Document] - 按相似度降序（距离升序）排列的文档列表
                             每个 Document 的 metadata 中附加了 score 字段（此处为距离值）

        复杂度：
            O(n * d)，n 为库中记录数，d 为向量维度。为暴力全量扫描，
            大规模数据需引入 HNSW / IVF 等近似最近邻索引
        """
        # 1.将query转换成向量
        # Embeddings.embed_query(text: str) -> List[float]
        #   作用：将单条查询文本向量化
        #   参数：text - 查询字符串
        #   返回：一维浮点数组，即查询向量
        #   说明：与 embed_documents 区分，部分模型对 query 使用不同的指令前缀
        embedding = self._embedding.embed_query(query)

        # 2.循环和store中的每一个向量进行比较，计算欧几里得距离
        result = []
        for key, record in self.store.items():
            # 计算查询向量与当前记录向量的欧几里得距离
            distance = self._euclidean_distance(embedding, record["vector"])
            # 字典解包 **record 把 id/text/vector/metadata 一并展开，附加 distance 字段
            result.append({"distance": distance, **record})

        # 3.排序，欧几里得距离越小越靠前
        # sorted(iterable, key) -> list
        #   key=lambda x: x["distance"] 指定按距离字段升序排列
        #   距离越小代表向量越接近，即语义越相似
        sorted_result = sorted(result, key=lambda x: x["distance"])

        # 4.取数据，取k条数据
        # 列表切片取前 k 条，k 大于总数时不会报错，返回全部
        result_k = sorted_result[:k]

        # 转换为 LangChain 标准的 Document 对象
        # Document(page_content: str, metadata: dict)
        #   page_content - 文档正文，下游 RAG 链会把它填充进提示词
        #   metadata     - 元数据，这里合并原始元数据并注入 score（实为距离值）
        # 注意：score 此处语义为「距离」，越小越相似，与生态惯例相反（见知识点 5）
        return [
            Document(page_content=item["text"], metadata={**item["metadata"], "score": item["distance"]})
            for item in result_k
        ]

    @classmethod
    def from_texts(cls: Type["MemoryVectorStore"], texts: List[str], embedding: Embeddings,
                   metadatas: Optional[List[dict]] = None,
                   **kwargs: Any) -> "MemoryVectorStore":
        """从文本和元数据中去构建向量数据库

        VectorStore 抽象方法之一，工厂类方法，提供「一步建库」的便捷入口。

        参数：
            cls: Type[MemoryVectorStore]     - 类本身，classmethod 自动传入，支持子类继承
            texts: List[str]                 - 初始文本列表
            embedding: Embeddings            - 嵌入模型实例
            metadatas: Optional[List[dict]]  - 与 texts 对应的元数据列表
            **kwargs: Any                    - 透传给 add_texts 的扩展参数

        返回：
            MemoryVectorStore - 已完成初始化与数据写入的实例

        设计模式：
            工厂方法模式。等价于 __init__ + add_texts 两步操作的封装，
            LangChain 所有 VectorStore 都提供该统一入口，便于互相替换
        """
        # 实例化自身（cls 而非硬编码类名，保证子类继承时返回正确类型）
        memory_vector_store = cls(embedding=embedding)
        # 复用 add_texts 完成数据写入
        memory_vector_store.add_texts(texts, metadatas, **kwargs)
        return memory_vector_store

    @classmethod
    def _euclidean_distance(cls, vec1: list, vec2: list) -> float:
        """计算两个向量的欧几里得距离

        参数：
            vec1: list - 向量 1（通常为查询向量）
            vec2: list - 向量 2（通常为库中文档向量）

        返回：
            float - 两向量的 L2 距离，值域 [0, +∞)，越小越相似

        实现说明：
            np.array(vec) 将 Python 列表转为 NumPy 数组，支持向量化运算
            np.linalg.norm(v) 默认计算 L2 范数，即 sqrt(Σ v_i²)
            两者组合即 sqrt(Σ(vec1_i - vec2_i)²)，为欧几里得距离定义
            使用 NumPy 而非纯 Python 循环，可获得数量级的性能提升
        """
        return np.linalg.norm(np.array(vec1) - np.array(vec2))


# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE 等）
dotenv.load_dotenv()

# 1.创建初始数据与嵌入模型
# 构造一批语义各异的中文短句，用于验证检索是否能命中语义相关内容
texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]

# 元数据列表，与 texts 严格一一对应
# page 模拟文档页码，account_id 模拟多租户归属，可用于后续过滤检索
metadatas = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6, "account_id": 1},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]

# OpenAIEmbeddings(model: str) -> OpenAIEmbeddings
#   作用：创建 OpenAI 文本嵌入模型实例
#   参数：model - 模型名称，text-embedding-3-small 输出 1536 维向量
#   返回：实现了 Embeddings 接口的对象，提供 embed_documents 与 embed_query
#   说明：该模型输出的向量已做 L2 归一化，因此欧几里得距离排序与余弦相似度等价
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.构建自定义向量数据库
# 直接实例化，此时 store 为空（注意 store 是类属性，见知识点 6）
db = MemoryVectorStore(embedding=embedding)

# MemoryVectorStore.add_texts(texts, metadatas) -> List[str]
#   执行流程：
#     1. 校验 metadatas 长度与 texts 一致
#     2. 调用 embed_documents 批量向量化 10 条文本，得到 10 个 1536 维向量
#     3. 生成 10 个 uuid4 作为主键
#     4. 组装成 {id, text, vector, metadata} 结构写入 store 字典
#   返回：10 个 id 字符串组成的列表
ids = db.add_texts(texts, metadatas)
print(ids)

# 3.执行检索
# MemoryVectorStore.similarity_search(query, k=4) -> List[Document]
#   执行流程：
#     1. embed_query("笨笨是谁？") → 查询向量
#     2. 遍历 store 中 10 条记录，逐一计算欧几里得距离
#     3. 按距离升序排序
#     4. 取前 4 条转为 Document 返回
#   预期结果：
#     "笨笨是一只很喜欢睡觉的猫咪" 距离最小排第一（直接包含"笨笨"）
#     "猫咪在窗台上打盹..." 次之（语义相关，同为猫咪主题）
print(db.similarity_search("笨笨是谁？"))

# 最佳实践与扩展写法
#
# 1. 修正 store 的类属性问题（生产必做）：
#    def __init__(self, embedding: Embeddings):
#        self._embedding = embedding
#        self.store = {}          # 改为实例属性，避免多实例共享数据
#
# 2. 统一 score 语义为「越大越相似」，以兼容 similarity_score_threshold：
#    similarity = 1 / (1 + distance)
#    Document(page_content=item["text"], metadata={**item["metadata"], "score": similarity})
#
# 3. 实现 similarity_search_with_score，让上层能显式获取分数：
#    def similarity_search_with_score(self, query, k=4, **kwargs):
#        ...
#        return [(Document(page_content=i["text"], metadata=i["metadata"]), i["distance"]) for i in result_k]
#
# 4. 支持元数据过滤，实现多租户隔离：
#    def similarity_search(self, query, k=4, filter=None, **kwargs):
#        records = self.store.values()
#        if filter:
#            records = [r for r in records if all(r["metadata"].get(x) == y for x, y in filter.items())]
#
# 5. 实现 delete 支持按 id 删除：
#    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]:
#        for _id in (ids or []):
#            self.store.pop(_id, None)
#        return True
#
# 6. 使用工厂方法一步建库（等价于上面的两步操作）：
#    db = MemoryVectorStore.from_texts(texts, embedding, metadatas)
#
# 7. 转为 Retriever 接入 RAG 链（基类已实现，无需额外编码）：
#    retriever = db.as_retriever(search_kwargs={"k": 3})
#    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser
#
# 8. 向量化计算性能优化（把逐条循环改为矩阵运算）：
#    matrix = np.array([r["vector"] for r in self.store.values()])
#    distances = np.linalg.norm(matrix - np.array(embedding), axis=1)
#    top_k_idx = np.argsort(distances)[:k]
