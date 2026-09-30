#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/6 12:53
@Author  : thezehui@gmail.com
@File    : 1.父文档检索器示例.py

===================================================================================
知识点讲解：多级文档分割与父文档检索（小文档块检索大文档块）
===================================================================================

1. 何时需要多级分割
   - 上一例把「父文档」设为完整原始文档，但有时完整文档太大，并不想按原样返回
   - 此时希望：先把原始文档切成较大的块（如 1000~2000 字符/token），再切成小块索引，检索时返回较大块而非原文档
   - 这样在检索精度和返回长度之间取得更好的平衡，也节省 token

2. 核心概念：三级结构
   - 原始文档 → 父文档块（parent_splitter 切出的大块）→ 子文档块（child_splitter 切出的小块）
   - parent_splitter：生成较大的文档块，作为最终返回内容
   - child_splitter：生成较小的文档块，用于生成嵌入并检索
   - 分层存储：向量库存小块，文档库存中块（父文档块）

3. 组件定位与复用
   - 与基本用法共用同一个 ParentDocumentRetriever，无需改动检索流程
   - 唯一区别：多传一个 parent_splitter 参数，框架据此把「父文档」从「完整原文档」换成「大块」
   - 其余（vectorstore、byte_store、child_splitter）完全不变，体现了对开闭原则的践行

4. 本示例配置（小块检索，中块返回）
   - parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)：父文档块约 2000 字符，存入文档数据库
   - child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)：子文档块约 500 字符，用于检索
   - 检索时用小块匹配查询（精确），命中后返回对应的 2000 字符大块（足够上下文）

5. 与基本用法的对比
   - 基本用法：不设置 parent_splitter，返回完整原始文档
   - 高级用法：设置 parent_splitter，返回中等大小的文档块
   - 取舍：高级用法避免返回过长文档（节省 token），同时保持足够上下文，适合超长文档

6. 优势与适用场景
   - 更灵活的粒度控制；避免返回过长文档（省 token）；保持足够上下文；适合超长文档
   - 适用：书籍、长篇文章等超长文档；需精确控制返回内容长度；多章节文档的分段检索

7. 工作原理（五个步骤）
   - 用 parent_splitter 将原始文档切成大块（如 2000 字符）
   - 用 child_splitter 将每个大块进一步切成小块（如 500 字符）
   - 小块生成嵌入向量存入向量数据库
   - 大块（父文档）存入文档数据库
   - 检索用小块匹配（精确），返回对应大块（足够上下文）

8. 最佳实践
   - parent_splitter 大小建议 1500~3000 字符（提供足够上下文）
   - child_splitter 大小建议 300~600 字符（保证检索精度）
   - 保持合理的 chunk_overlap 避免语义断裂
   - 根据具体文档类型调整分割策略

===================================================================================
"""
import dotenv
import weaviate
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载环境变量配置
dotenv.load_dotenv()

# 1.创建加载器与文档列表，并加载文档
# 加载多个文档用于演示多级分割
loaders = [
    UnstructuredLoader("./电商产品数据.txt"),
    UnstructuredLoader("./项目API文档.md"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 2.创建文本分割器
# 关键：这里创建了两个分割器，实现多级分割

# parent_splitter：父文档分割器，生成较大的文档块
# - chunk_size=2000：每个父文档块约 2000 字符
# 这些块将被存储到文档数据库，作为最终返回的内容
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# child_splitter：子文档分割器，生成较小的文档块
# - chunk_size=500：每个子文档块约 500 字符
# - chunk_overlap=50：50 字符重叠
# 这些块将被转换为向量，用于检索匹配
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 3.创建向量数据库与文档数据库
# WeaviateVectorStore：用于存储子文档（小块）的嵌入向量
vector_store = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://mbakeruerziae6psyex7ng.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("ZltPVa9ZSOxUcfafelsggGyyH6tnTYQYJvBx"),
    ),
    index_name="ParentDocument",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# LocalFileStore：用于存储父文档（大块）的完整内容
store = LocalFileStore("./parent-document")

# 4.创建父文档检索器
# ParentDocumentRetriever：配置多级分割的检索器
retriever = ParentDocumentRetriever(
    # vectorstore：存储子文档向量
    vectorstore=vector_store,
    # byte_store：存储父文档内容
    byte_store=store,
    # parent_splitter：父文档分割器（生成大块）
    parent_splitter=parent_splitter,
    # child_splitter：子文档分割器（生成小块）
    child_splitter=child_splitter,
)

# 5.添加文档
# add_documents：执行多级分割和存储
# 处理流程：
# 1. 使用 parent_splitter 将原始文档分割成父文档块
# 2. 为每个父文档块分配唯一 ID
# 3. 使用 child_splitter 将每个父文档块进一步分割成子文档块
# 4. 为每个子文档块生成嵌入向量并存储到 vectorstore
# 5. 将父文档块存储到 byte_store
# 6. 建立子文档和父文档的关联关系
retriever.add_documents(docs, ids=None)

# 6.检索并返回内容
# invoke：执行检索
# 检索流程：
# 1. 将查询转换为嵌入向量
# 2. 在向量数据库中匹配最相似的子文档块（小块）
# 3. 从子文档的 metadata 中获取父文档 ID
# 4. 根据父文档 ID 从文档数据库中获取父文档块（大块）
# 5. 返回父文档块（约 2000 字符）而不是子文档块（约 500 字符）
search_docs = retriever.invoke("分享关于LLMOps的一些应用配置")

# 输出结果
print(search_docs)
print(len(search_docs))

# 结果分析：
# - 如果只用 child_splitter（500 字符），返回的上下文可能不足
# - 如果只用 parent_splitter（2000 字符），检索精度可能降低
# - 使用两级分割：用 500 字符的小块检索（精确），返回 2000 字符的大块（完整）

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
