#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/25 13:36
@Author  : thezehui@gmail.com
@File    : 01.CacheBackEmbedding使用示例.py

===================================================================================
知识点讲解：CacheBackedEmbeddings 嵌入缓存组件
===================================================================================

1. 为什么需要嵌入缓存
   - 调用嵌入模型 API 存在三大成本：金钱（按 token 计费）、时间（网络往返）、配额（限流）
   - 在实际开发中大量文本会被反复嵌入：程序调试重跑、文档增量更新、同一 query 多次检索
   - CacheBackedEmbeddings 在嵌入模型外面包一层「读写缓存」，相同文本只需真正计算一次
   - 核心价值：通过嵌入模型计算传递数据的向量需要昂贵的算力，对于重复的内容，
     Embeddings 计算的结果肯定是一致的，如果数据重复仍然二次计算，会导致效率非常低，
     而且增加无用功

2. CacheBackedEmbeddings 工作原理
   - 它本身也实现了 Embeddings 接口，所以可以无感替换原有的 embeddings 对象
   - 缓存 key 的生成规则：对「namespace + 文本内容」做 SHA-1 哈希，得到定长字符串
   - 缓存 value：把 list[float] 向量序列化为 bytes 后写入 ByteStore
   - 执行流程（以 embed_documents 为例）：
     a) 对每条文本计算 cache key
     b) 用 mget 批量查询 ByteStore，命中的直接反序列化为向量
     c) 仅把未命中的文本提交给底层真实嵌入模型
     d) 新算出的向量通过 mset 批量写回缓存
     e) 按原始输入顺序拼装结果返回
   - 底层本质：封装了一个持久化存储的数据存储仓库，在每次进行数据嵌入前，
     从数据存储仓库中检索对应的向量，然后逐个匹配对应的数据是否相等，
     找到缓存中没有的文本，将这些文本调用嵌入生成向量，
     最后将生成的新向量存储到数据仓库中

3. from_bytes_store() 类方法
   - 这是创建 CacheBackedEmbeddings 的推荐入口（构造函数需要手动包装编解码器）
   - 关键参数：
     · underlying_embedder     : 用于嵌入的底层嵌入模型，缓存未命中时调用它
     · document_embedding_cache: ByteStore 实现，用于缓存文档嵌入的任何存储库
     · batch_size              : 可选参数，默认为 None，在存储更新之间要嵌入的文档数量
     · namespace               : 缓存命名空间前缀，默认为 ""，用于文档缓存的命名空间，
                                 用于避免与其他缓存发生冲突，建议设置为嵌入模型的名称
     · query_embedding_cache   : 可选，默认为 None 或不缓存，用于缓存查询/文本嵌入的
                                 ByteStore，或设为 True 以使用与 document_embedding_cache
                                 相同的存储

4. namespace 参数的重要性（关键注意事项）
   - 不同嵌入模型对同一文本会产生完全不同的向量，且维度可能不同
   - 如果不设置 namespace，切换模型后会命中「旧模型」留下的缓存，导致向量空间错乱，
     检索结果完全不可用，而且不会报错，属于极难排查的静默 bug
   - 正确做法：namespace=embeddings.model，让模型名参与哈希，天然隔离不同模型的缓存
   - 注意：CacheBackedEmbedding 默认不缓存 embed_query 生成的向量，
     如果要缓存，需要设置 query_embedding_cache 的值
   - 请尽可能设置 namespace，以避免使用不同嵌入模型嵌入的相同文本发生冲突

5. ByteStore 存储后端选型
   - LocalFileStore(path) : 本地文件系统，每个 key 一个文件，适合开发调试与单机脚本
   - InMemoryByteStore()  : 纯内存，进程结束即丢失，适合单次运行内的去重
   - RedisStore / UpstashRedisByteStore : 分布式缓存，适合多实例生产环境
   - 所有 ByteStore 都实现 BaseStore[str, bytes] 接口，可自行扩展

6. 使用注意事项与最佳实践
   - query 缓存要谨慎：用户查询长尾且重复率低，开启后可能积累大量无用缓存文件
   - LocalFileStore 会随文档量增长产生海量小文件，生产环境应换成 Redis
   - 缓存目录（如 ./cache/）属于运行时产物，应加入 .gitignore，不要提交到仓库
   - 缓存没有内置过期机制，更换模型版本时要么改 namespace，要么手动清空目录
   - 验证缓存是否生效的最直观方式：连续运行本脚本两次，第二次耗时应明显下降

===================================================================================
"""
import dotenv
import numpy as np
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from numpy.linalg import norm

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()


def cosine_similarity(vector1: list, vector2: list) -> float:
    """计算传入两个向量的余弦相似度

    :param vector1: 第一个向量（list[float]）
    :param vector2: 第二个向量（list[float]），维度需与 vector1 相同
    :return: 余弦相似度，取值范围 [-1, 1]，越接近 1 语义越相近
    """
    # 1.计算内积/点积
    # np.dot：对应元素相乘后求和，得到标量结果
    dot_product = np.dot(vector1, vector2)

    # 2.计算向量的范数/长度
    # norm 默认求 L2 范数，即向量在欧氏空间中的长度
    norm_vec1 = norm(vector1)
    norm_vec2 = norm(vector2)

    # 3.计算余弦相似度
    # 用模长归一化点积，抵消向量长度差异，只比较方向夹角
    return dot_product / (norm_vec1 * norm_vec2)


# 1.创建底层真实的嵌入模型
# 这是缓存未命中时实际发起 API 请求的对象，输出 1536 维向量
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.用缓存包装嵌入模型
# CacheBackedEmbeddings.from_bytes_store() 参数说明：
#   第 1 个位置参数 underlying_embeddings   : 底层嵌入模型（上面创建的 embeddings）
#   第 2 个位置参数 document_embedding_cache: ByteStore 实例，这里用本地文件存储
#       LocalFileStore("./cache/") -> 在当前目录下的 cache 文件夹中，
#       以哈希值为文件名逐条保存序列化后的向量字节
#   namespace=embeddings.model : 缓存命名空间，取值为 "text-embedding-3-small"
#       作用是让「模型名」参与缓存 key 的哈希计算，避免更换模型后误命中旧缓存
#   query_embedding_cache=True : 同时缓存 embed_query 的结果，
#       传 True 表示复用 document 的同一个 store（内部会加 query 专用前缀）
# 返回值：CacheBackedEmbeddings 实例，它同样实现 Embeddings 接口，
#        因此可以直接当作普通嵌入模型传给 VectorStore、Retriever 等组件
embeddings_with_cache = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    LocalFileStore("./cache/"),
    namespace=embeddings.model,
    query_embedding_cache=True,
)

# 3.嵌入单条查询文本
# embed_query(text) 执行流程：
#   a) 计算 key = sha1(namespace + text)
#   b) 到 LocalFileStore 查找该 key 对应的文件
#   c) 命中 -> 直接反序列化返回向量，零 API 调用、零费用
#      未命中 -> 调用 OpenAI 接口获取向量，并写入缓存文件后返回
# 返回：list[float]，长度 1536
query_vector = embeddings_with_cache.embed_query("你好，我是慕小课，我喜欢打篮球")

# 4.批量嵌入文档列表
# embed_documents(texts) 执行流程（比 embed_query 多了批量优化）：
#   a) 为每条文本计算 cache key
#   b) 通过 store.mget(keys) 一次性批量读取，区分命中与未命中
#   c) 只把未命中的文本打包成一次 API 请求提交给底层模型
#   d) 通过 store.mset(...) 批量写回新算出的向量
#   e) 按输入顺序合并「缓存命中」与「新计算」的结果后返回
# 注意：第一条文本与上面 embed_query 的内容相同，但由于 query 与 document
#      使用不同的 key 前缀，两者缓存相互独立，不会互相命中
# 返回：list[list[float]]，外层 3 条，内层各 1536 维
documents_vector = embeddings_with_cache.embed_documents([
    "你好，我是慕小课，我喜欢打篮球",
    "这个喜欢打篮球的人叫慕小课",
    "求知若渴，虚心若愚"
])

# 打印 query 向量内容与维度，确认缓存包装后输出格式与原模型完全一致
print(query_vector)
print(len(query_vector))

print("============")

# 打印成功嵌入的文档条数（3 条）
print(len(documents_vector))

# 5.验证语义相似度，确认走缓存后向量数值依然正确
# vector1 与 vector2：同一语义的两种表达，相似度应偏高
print("vector1与vector2的余弦相似度:", cosine_similarity(documents_vector[0], documents_vector[1]))
# vector1 与 vector3：自我介绍 vs 格言，语义无关，相似度应偏低
# 若两次运行本脚本得到完全相同的相似度数值，即可侧面证明缓存命中且数据无损
print("vector2与vector3的余弦相似度:", cosine_similarity(documents_vector[0], documents_vector[2]))
