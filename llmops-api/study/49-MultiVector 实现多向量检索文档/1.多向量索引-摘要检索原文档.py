#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/6 19:35
@Author  : thezehui@gmail.com
@File    : 1.多向量索引-摘要检索原文档.py

===================================================================================
知识点讲解：多向量检索器与摘要检索策略
===================================================================================

1. 设计动机：为什么要多向量/多表征索引
   - 常规做法是「一个文档块 → 一条向量」，这条向量要独自承担该块的全部特征信息
   - 若能从多个维度记录同一文档块的信息（即给它生成多条向量），会大大增加该块被检索命中的概率
   - 这正是索引阶段（RAG 6 阶段中的第 4 阶段）的代表性优化策略之一

2. 三种实现路径（共用同一检索骨架）
   - 多向量检索的核心思路是「往向量库里塞什么派生文本」，三条路径差别仅在此：
     * 更小子块检索父文档：检索小块、返回其父文档（ParentDocumentRetriever，见 study/50）
     * 摘要：用 LLM 为每个块生成摘要，与原文一起嵌入或代替嵌入，返回时回原文（本示例）
     * 假设性查询：用 LLM 为每个块生成适合回答的假设性问题，同上处理（见本目录 2 文件）

3. 核心概念与双存储架构
   - MultiVectorRetriever：支持「检索派生文档、返回原始文档」的多向量检索器
   - 摘要检索策略：用摘要做匹配，用原文做回答
   - 双存储架构：向量库存摘要（派生文本），文档库存原文（LocalFileStore）
   - 两者通过 metadata 中的唯一标识 doc_id 关联

4. 工作原理（六个步骤）
   - 分块：将原始文档切成文档块
   - 摘要：为每个块生成简洁摘要
   - 入向量库：把摘要转向量存入向量数据库
   - 入文档库：把原始文档存入文档数据库
   - 关联：摘要与原文档 metadata 写入同一 doc_id
   - 检索：用摘要匹配查询 → 取其 metadata 的 doc_id → 去文档库换回原文

5. 关联机制闭环
   - 原始文档与摘要文档都在 metadata 写入同一个唯一标识 doc_id
   - 从向量库命中摘要后，取出其 metadata 中的 doc_id
   - 再用该 id 去文档数据库匹配出原文档，完成整个多表征检索闭环

6. 本示例关键代码要点
   - summary_chain 用 batch(max_concurrency=5) 并发地为所有块生成摘要，提升效率
   - FAISS.from_documents 存入的是摘要文档（不是原文）
   - MultiVectorRetriever 三个核心入参：vectorstore（摘要向量库）、byte_store（原文库）、id_key（关联字段名，本例 "doc_id"）
   - retriever.docstore.mset(zip(doc_ids, docs)) 建立 doc_id → 原文档 的映射

7. 成本与开销权衡
   - 存储开销：向量库存摘要（比原文小）+ 文档库存原文，总体约等于「原文 + 摘要」两份
   - 构建成本较高：每个文档块都要额外消耗一次 LLM 调用来生成摘要，量大时费用与耗时可观，适合离线构建
   - 检索延迟几乎不增加：检索期只多一次按 id 取原文的 KV 查询，无额外 LLM 调用
   - 风险提示：多向量/多表征索引与 RAPTOR 都会用额外 LLM 对文档做总结/提取/分类，
     数据越多信息失真的概率越大，需谨慎；实际索引阶段用得最多的仍是「分割」与「特定 Embeddings」两类不引入 LLM 改写的策略

8. 典型输出示例与观察点（查询「推荐一些潮州特产?」）
   - 返回 4 条 Document，page_content 是原始产品数据全文，而非当初存进向量库的摘要
   - metadata 也是原文档的 source，不含 doc_id
   - 这正说明：检索命中摘要后已按 doc_id 换回了原文——多向量检索生效的标志

9. 应用场景
   - 长文档 / 复杂文档检索（技术文档、学术论文等）
   - 多粒度检索（用不同详细程度的文本检索）
   - 知识库问答（用简洁描述匹配，用详细内容回答）

10. 最佳实践
   - 摘要应简洁但保留关键信息；合理设置 chunk_size 与 chunk_overlap
   - 使用 batch 提高摘要生成效率；为每个文档分配唯一 id 确保关联正确
   - 摘要可以「与原文一起嵌入」也可以「代替原文嵌入」：
     一起嵌入召回面更广但存储翻倍，代替嵌入更省但丢失原文的字面匹配能力

11. 版本对照（课程基于 LangChain 0.x，本示例为 1.x）
   - 0.x：from langchain.retrievers import MultiVectorRetriever /
         from langchain.storage import LocalFileStore /
         from langchain_community.document_loaders import UnstructuredFileLoader
   - 1.x：from langchain_classic.retrievers import MultiVectorRetriever /
         from langchain_classic.storage import LocalFileStore /
         from langchain_unstructured import UnstructuredLoader
   - MultiVectorRetriever 的三个核心入参（vectorstore / byte_store / id_key）两版一致

===================================================================================
"""
import uuid

import dotenv
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量配置
dotenv.load_dotenv()

# 1.创建加载器、文本分割器并处理文档
# UnstructuredLoader：通用文档加载器，支持多种格式
loader = UnstructuredLoader("./电商产品数据.txt")

# RecursiveCharacterTextSplitter：递归字符文本分割器
# - chunk_size：每个文本块的目标大小（字符数）
# - chunk_overlap：相邻块之间的重叠部分，避免语义断裂
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# load_and_split：加载文档并立即分割
docs = loader.load_and_split(text_splitter)

# 2.定义摘要生成链
# 使用 LCEL 表达式构建摘要生成流程
summary_chain = (
        # lambda x: x.page_content：提取 Document 对象的文本内容
        {"doc": lambda x: x.page_content}
        # ChatPromptTemplate.from_template：创建摘要提示模板
        | ChatPromptTemplate.from_template("请总结以下文档的内容：\n\n{doc}")
        # ChatOpenAI：调用大语言模型生成摘要
        | ChatOpenAI(model="deepseek-v4-pro", temperature=0)
        # StrOutputParser：将模型输出解析为字符串
        | StrOutputParser()
)

# 3.批量生成摘要与唯一标识
# batch：批量处理多个文档，提高效率
# - max_concurrency：最大并发数，控制同时处理的请求数量
summaries = summary_chain.batch(docs, {"max_concurrency": 5})

# 为每个摘要生成唯一的 UUID 作为文档 ID
# 这个 ID 用于关联向量数据库中的摘要和文档数据库中的原文
doc_ids = [str(uuid.uuid4()) for _ in summaries]

# 4.构建摘要文档
# 将摘要包装成 Document 对象，并在 metadata 中存储 doc_id
summary_docs = [
    Document(page_content=summary, metadata={"doc_id": doc_ids[idx]})
    for idx, summary in enumerate(summaries)
]

# 5.构建文档数据库与向量数据库
# LocalFileStore：本地文件存储，用于存储原始文档
# 这是一个键值存储，键是 doc_id，值是序列化的 Document 对象
byte_store = LocalFileStore("./multy-vector")

# FAISS.from_documents：创建 FAISS 向量数据库并添加文档
# 注意：这里添加的是摘要文档，不是原始文档
# - summary_docs：包含摘要的文档列表
# - embedding：用于生成嵌入向量的模型
db = FAISS.from_documents(
    summary_docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# 6.构建多向量检索器
# MultiVectorRetriever：多向量检索器的核心类
# - vectorstore：存储摘要向量的向量数据库
# - byte_store：存储原始文档的文档数据库
# - id_key：用于关联摘要和原文的元数据字段名
retriever = MultiVectorRetriever(
    vectorstore=db,
    byte_store=byte_store,
    id_key="doc_id",
)

# 7.将摘要文档和原文档存储到数据库中
# docstore.mset：批量设置键值对
# zip(doc_ids, docs)：将 doc_id 和原始文档配对
# 这样就建立了 doc_id -> 原始文档的映射关系
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 8.执行检索
# 检索流程：
# 1. 将查询转换为嵌入向量
# 2. 在向量数据库中匹配最相似的摘要
# 3. 从摘要的 metadata 中提取 doc_id
# 4. 根据 doc_id 从文档数据库中获取原始文档
# 5. 返回原始文档而非摘要
search_docs = retriever.invoke("推荐一些潮州特产?")
print(search_docs)
print(len(search_docs))

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
