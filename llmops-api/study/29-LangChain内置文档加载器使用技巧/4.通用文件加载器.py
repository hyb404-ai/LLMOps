#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/1 23:30
@Author  : thezehui@gmail.com
@File    : 4.通用文件加载器.py

===================================================================================
知识点讲解：UnstructuredLoader 通用文件加载器
===================================================================================

1. 为什么需要通用加载器
   - 实际开发中数据种类无穷，无法为每种数据单独配一个加载器
   - 对于无法判断类型或想做通用加载的文件，统一用非结构化文件加载器 UnstructuredLoader 实现加载
   - 它是所有 UnstructuredXxxLoader 文档类的基类，核心是把文档划分为「元素」

2. 支持的文件类型
   - 一个通用加载器可加载：文本文件、PowerPoint、HTML、PDF、图像、Markdown、Excel、Word 等
   - 当传入一个文件时，库会读取文档、分割为多个部分、对各部分分类、提取文本，
     再依据策略决定是否合并（single / paged / elements）

3. LangChain 1.x 的包变化
   - 0.x 版本: langchain_community.document_loaders.UnstructuredFileLoader
   - 1.x 版本: 改用独立集成包 langchain-unstructured 的 UnstructuredLoader
   - 需单独安装: pip install langchain-unstructured

4. chunking_strategy 参数（替代旧 mode）
   - 默认值（按元素 element 返回）：每个结构元素是一个 Document，保留更多文档结构信息
   - "basic"   : 将整个文件合并为一个 Document
   - "by_title": 按标题分块，适合长文档
   - 旧版 mode 参数已废弃，由 chunking_strategy 代替

5. 元素类型（element types）
   - Title        : 标题
   - NarrativeText: 叙述文本
   - ListItem     : 列表项
   - Table        : 表格
   - metadata 中可通过 category 字段查看元素类型

6. 元数据偏少与选型建议
   - 通用加载器提取的 metadata 通常只记录 source（数据来源），信息相对较少
   - 因此若明确文件类型，或属于高频文件，应尽可能使用更精确的专用加载器，记录内容更丰富
   - 实践中常按扩展名分派：匹配到的用专用加载器，没匹配到的再退回通用加载器，例如：
       if file_extension in [".xlsx", ".xls"]:
           loader = UnstructuredExcelLoader(file_path)
       elif file_extension == ".pdf":
           loader = UnstructuredPDFLoader(file_path)
       elif file_extension in [".md", ".markdown"]:
           loader = UnstructuredMarkdownLoader(file_path)
       ...
       else:
           loader = UnstructuredFileLoader(file_path)

7. 性能考虑
   - unstructured 库功能强大但解析速度较慢
   - 若明确文件类型，使用专用加载器可能更快
   - 适合离线处理，不适合实时高并发场景

8. 使用场景
   - 需要处理多种文件格式的应用
   - 不确定输入文件类型的场景
   - 需要保留文档结构信息的知识库构建

===================================================================================
"""
from langchain_unstructured import UnstructuredLoader

# UnstructuredLoader(): 通用文件加载器
#   参数:
#     - file_path: 文件路径，支持多种格式
#     - chunking_strategy: 分块策略
#       * None/默认: 按元素（element）返回，保留文档结构
#       * "basic": 整个文件作为一个文档
#       * "by_title": 按标题分块
#     - mode: 加载模式（已废弃，使用 chunking_strategy 代替）
#   返回: UnstructuredLoader 实例
#   作用: 自动识别文件类型并解析内容
loader = UnstructuredLoader("./项目API资料.md")

# load(): 执行文件加载和解析
#   返回: List[Document]
#   作用: 根据文件类型选择合适的解析器，提取结构化内容
#   注意: 默认按元素返回，可能返回多个 Document 对象
documents = loader.load()

# 输出文档对象列表
print(documents)

# 输出文档数量（按元素返回时，数量通常大于 1）
print(len(documents))

# 输出第一个文档的元数据
# 包含 source、category（元素类型）、page_number 等
print(documents[0].metadata)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
