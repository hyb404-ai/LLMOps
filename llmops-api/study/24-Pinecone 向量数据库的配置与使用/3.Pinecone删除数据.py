#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/17 9:18
@Author  : thezehui@gmail.com
@File    : 3.Pinecone删除数据.py

===================================================================================
知识点讲解：Pinecone 数据的删除与更新
===================================================================================

1. 为什么需要删除与更新能力
   - 知识库不是一次性写入就完结的，真实业务中必然面临：
     · 文档被删除 -> 需同步移除对应向量，否则会检索到已失效内容
     · 文档被修改 -> 需更新向量与原文，保持知识库与源数据一致
     · 用户注销 -> 需清理该租户名下的全部向量数据（合规要求）
   - 一个只能写不能删的向量库无法支撑生产级 RAG 系统

2. VectorStore.delete() 标准接口
   - 签名：delete(ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]
   - ids 从哪来：add_texts() / add_documents() 的返回值就是 id 列表，
     因此业务系统必须把「源文档 ID」与「向量库记录 ID」的映射关系持久化到关系数据库，
     这是实现精准增量更新的前提，也是最容易被忽略的工程要点

3. Pinecone 的三种删除方式
   - 按 id 删除：db.delete(ids=["id1", "id2"], namespace="dataset")
       最精确，适合单文档级别的同步
   - 按 metadata 过滤删除：pinecone_index.delete(filter={"doc_id": "xxx"}, namespace="...")
       适合「一个源文档被切分成多个 chunk」的批量清理场景
       注意：Serverless 索引对 filter 删除有限制，需查阅当前版本文档
   - 删除整个 namespace：pinecone_index.delete(delete_all=True, namespace="dataset")
       适合租户注销、知识库整体重建等场景，效率远高于逐条删除

4. 更新数据的两种思路
   - upsert 语义（推荐）：
     Pinecone 以 id 为主键，用相同 id 再次写入会直接覆盖旧记录，
     因此 add_texts(texts, metadatas, ids=["已存在的id"]) 天然实现了更新
   - 原生 index.update()：
     通过 get_pinecone_index() 拿到底层原生 Index 对象后调用，
     支持只更新向量或只更新部分 metadata（局部更新），
     比 upsert 更精细，但需要自己处理向量生成

5. get_pinecone_index() 逃生舱机制
   - LangChain 的 VectorStore 接口是各家数据库能力的「最小公约数」，
     必然无法覆盖某个数据库的全部特性
   - PineconeVectorStore.get_pinecone_index(index_name) 返回原生 pinecone.Index 对象，
     可调用 fetch / update / describe_index_stats / list 等 LangChain 未封装的方法
   - 这是一种常见的框架设计模式：提供统一抽象的同时保留访问底层的通道

6. 重要注意事项
   - namespace 必须匹配：删除时若不传 namespace，会在 default 命名空间中查找，
     导致「删除成功但数据依然存在」的假象，这是最高频的踩坑点
   - 删除是最终一致的（eventually consistent）：调用返回成功后，
     检索结果可能仍短暂包含被删数据，需等待几秒生效
   - delete() 对不存在的 id 不会报错，返回成功，因此不能用返回值判断是否真的删掉了
   - 删除操作不可逆，生产环境务必先在测试 namespace 验证

7. 最佳实践建议
   - 在业务数据库中维护 (源文档ID -> 向量库记录ID列表) 的映射表
   - 文档更新采用「先按源文档 ID 删除全部旧 chunk，再写入新 chunk」的策略，
     避免因切分结果条数变化导致残留脏数据
   - 批量删除单次不超过 1000 个 id，超出需分批
   - 删除后调用 describe_index_stats() 核对记录总数，确认操作真实生效

===================================================================================
"""
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 从 .env 文件加载环境变量（PINECONE_API_KEY、OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.创建嵌入模型与向量数据库实例
# 注意：删除操作本身并不需要调用嵌入模型（按 id 删除无需向量化），
#      但 PineconeVectorStore 构造函数要求传入 embedding，
#      因此这里仍需创建，只是在本脚本中实际未被使用
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# PineconeVectorStore 构造参数：
#   index_name="llmops" : 目标 Index
#   embedding=embedding : 嵌入模型（本例删除操作不会用到）
#   namespace="dataset" : 默认命名空间，需与写入时使用的 namespace 保持一致
db = PineconeVectorStore(index_name="llmops", embedding=embedding, namespace="dataset")

# 2.指定待删除的记录 id
# 该 id 是 uuid4 格式，来源于此前 add_texts() 的返回值
# 工程提醒：id 必须由业务系统持久化保存（通常存在关系数据库的文档表中），
#          否则一旦丢失就只能用 metadata 过滤或全量重建来清理数据
id = "23cb7d6f-f77d-4465-8634-9c1ca7f93895"

# 3.执行删除
# delete(ids, namespace) 参数说明：
#   ids       : List[str]，待删除的记录 id 列表，支持批量（单次建议不超过 1000 条）
#   namespace : str，目标命名空间。此处显式重复传入 "dataset"，
#               虽然构造时已指定，但显式传参可避免因默认值变更导致误删/漏删
#   delete_all: 可选 bool，为 True 时清空整个 namespace（危险操作，谨慎使用）
#   filter    : 可选 dict，按 metadata 条件批量删除
# 返回值：None（Pinecone 删除接口不返回被删条数）
# 重要特性：
#   · 幂等性 —— 对不存在的 id 执行删除不会报错，同样返回成功
#   · 最终一致 —— 返回后向量可能仍短暂可被检索到，需稍等几秒
#   · 不可逆 —— 没有回收站，删除即永久丢失
db.delete([id], namespace="dataset")

# 4.进阶用法：通过原生 Index 对象执行局部更新（示例代码，未启用）
# get_pinecone_index(index_name) 返回底层 pinecone.Index 原生对象，
# 用于调用 LangChain VectorStore 接口未封装的高级方法，是框架的「逃生舱」设计
# pinecone_index = db.get_pinecone_index("llmops")
#
# index.update(...) 参数说明：
#   id        : str，待更新记录的主键 id（必填）
#   values    : List[float]，新的向量值；传空列表表示不更新向量
#   metadata  : dict，需要更新的 metadata 字段（局部合并，未提及的字段保持原值）
#   namespace : str，目标命名空间
# 与 upsert 的差异：update 支持只改 metadata 而保留原向量，
#                 适合「文档内容未变但标签/权限变了」的场景，无需重新计算 embedding
# pinecone_index.update(id="xxx", values=[], metadata={}, namespace="xxx")
