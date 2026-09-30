#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/6 12:53
@Author  : thezehui@gmail.com
@File    : 1.父文档检索器示例.py

===================================================================================
知识点讲解：父文档检索器（ParentDocumentRetriever）
===================================================================================

1. 设计动机：拆分与检索的冲突
   - RAG 中「文档拆分」与「文档检索」的愿望相互冲突：
     * 希望文档小：嵌入向量才能最准确地反映其语义，太长则嵌入记录不了太多文本特征
     * 又希望文档长：才能保留每个块的上下文
   - 矛盾难以两全，于是引入「拆分子文档块、检索父文档块」的平衡策略

2. 父文档检索器的核心思想
   - 检索时先获取小块（子文档），再依据小块元数据中存储的 id，查找并返回其所属的更大文档（父文档）
   - 「父文档」指小块来源的文档，可以是整个原始文档，也可以是切割后较大的文档块
   - 该策略适合不太能拆分、或上下文关联性很强的文档（如技术文档、法律文件）

3. 组件定位
   - LangChain 在 MultiVectorRetriever 之上封装了 ParentDocumentRetriever，使用更便捷
   - 只需传入 向量数据库、文档数据库、子文档分割器 三要素即可工作
   - 子文档→父文档的运行流程与多向量检索器（MultiVectorRetriever）完全一致

4. 双层存储架构
   - 向量数据库（vectorstore）：存储小块的嵌入向量，负责精确检索
   - 文档数据库（byte_store，如 LocalFileStore）：以 文档ID→序列化父文档 的键值形式存储完整父文档
   - 自动关联：通过内部 ID 机制自动把子文档和它的父文档关联起来，无需手工维护

5. 工作原理（五个步骤）
   - 将原始文档用 child_splitter 分割成小块（子文档）
   - 为每个小块生成嵌入向量并存入向量数据库
   - 将完整原始文档（父文档）存入文档数据库
   - 检索时用小块匹配查询（语义聚焦、精确）
   - 命中后按 ID 回查文档数据库，返回完整的父文档（上下文完整）

6. 本示例配置（小块检索，完整原始文档返回）
   - 仅创建 child_splitter（chunk_size=500, chunk_overlap=50），不设置 parent_splitter
   - 此时「父文档」就是原始完整文档，检索返回的是整篇而非碎片
   - 子文档小而精确保证命中率，父文档完整保证上下文，两全其美

7. 优势与适用场景
   - 优势：检索精确（小块语义聚焦）+ 上下文完整（返回父文档）+ 关联自动管理 + 粒度可配
   - 适用：长文档问答（技术文档、法律文件）、需要完整上下文、精确检索与完整返回需平衡的任务

8. 最佳实践
   - 子文档（child_splitter）应小而精确，如 500 字符
   - 不设置 parent_splitter 时父文档即原始完整文档
   - 合理设置 chunk_overlap 避免语义断裂
   - 文档存储建议使用持久化方案（LocalFileStore），保证重启后父文档仍在

9. 典型输出示例
   - 检索返回的是完整文档片段，而非拆分后的小块（向量库里存的却仍是分割后的小块）
   - 例如 query="分享关于LLMOps的一些应用配置" 会返回带 metadata source 的完整 Document，len(search_docs) 通常为 1
   - 观察点：page_content 是整段原文，印证「小块检索、大块返回」

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
# UnstructuredLoader：通用文档加载器，支持多种格式
# 这里加载两个不同的文档用于演示
loaders = [
    UnstructuredLoader("./电商产品数据.txt"),
    UnstructuredLoader("./项目API文档.md"),
]

# 遍历加载器，加载所有文档
docs = []
for loader in loaders:
    # load()：加载文档，返回 Document 对象列表
    docs.extend(loader.load())

# 2.创建文本分割器
# 这里只创建子文档分割器，父文档就是原始完整文档
# RecursiveCharacterTextSplitter：递归字符文本分割器
# - chunk_size=500：每个子文档约 500 字符，保证检索精度
# - chunk_overlap=50：50 字符重叠，避免语义断裂
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# 3.创建向量数据库与文档数据库
# WeaviateVectorStore：Weaviate 向量数据库实例
# 用于存储子文档的嵌入向量
vector_store = WeaviateVectorStore(
    # connect_to_wcs：连接到 Weaviate Cloud Services
    client=weaviate.connect_to_wcs(
        cluster_url="https://mbakeruerziae6psyex7ng.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("ZltPVa9ZSOxUcfafelsggGyyH6tnTYQYJvBx"),
    ),
    # index_name：索引名称（Weaviate 中的 Class 名）
    index_name="ParentDocument",
    # text_key：文档内容在 Weaviate 中的属性名
    text_key="text",
    # embedding：用于生成嵌入向量的模型
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# LocalFileStore：本地文件存储，用于存储父文档
# 这是一个键值存储，键是文档 ID，值是序列化的完整文档
byte_store = LocalFileStore("./parent-document")

# 4.创建父文档检索器
# ParentDocumentRetriever：父文档检索器的核心类
retriever = ParentDocumentRetriever(
    # vectorstore：存储子文档向量的向量数据库
    vectorstore=vector_store,
    # byte_store：存储父文档的文档数据库
    byte_store=byte_store,
    # child_splitter：用于分割子文档的文本分割器
    child_splitter=text_splitter,
    # 注意：没有设置 parent_splitter，意味着父文档就是完整的原始文档
)

# 5.添加文档
# add_documents：将文档添加到检索器
# 该方法会自动：
# 1. 使用 child_splitter 将文档分割成子文档
# 2. 为每个子文档生成嵌入向量并存储到 vectorstore
# 3. 将完整的父文档存储到 byte_store
# 4. 建立子文档和父文档的关联关系（通过内部 ID）
# retriever.add_documents(docs, ids=None)

# 6.检索并返回内容
# 方式一：直接在向量数据库上检索（仅返回匹配的子文档）
search_docs = retriever.vectorstore.similarity_search("分享关于LLMOps的一些应用配置")

# 方式二：使用父文档检索器检索（返回匹配子文档对应的完整父文档）
# search_docs = retriever.invoke("分享关于LLMOps的一些应用配置")

# 输出结果
print(search_docs)
print(len(search_docs))

# 对比说明：
# - vectorstore.similarity_search()：返回匹配的小块子文档
# - retriever.invoke()：返回小块对应的完整父文档
# 实际应用中应该使用 retriever.invoke() 以获得完整上下文

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
