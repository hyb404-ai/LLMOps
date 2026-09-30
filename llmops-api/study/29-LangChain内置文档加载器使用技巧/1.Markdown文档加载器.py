#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/1 18:35
@Author  : thezehui@gmail.com
@File    : 1.Markdown文档加载器.py

===================================================================================
知识点讲解：Markdown 文档加载器的使用
===================================================================================

1. 文档加载器的统一使用范式
   - LangChain 内部封装了上百种文档加载器，涵盖 CSV、目录数据、HTML 网页、JSON、Markdown、PDF、Office 等
   - 所有加载器整体流程高度一致，固定两步：
     * 传递参数（文件路径、加载器配置等）创建加载器实例
     * 调用 .load() 得到 List[Document]
   - 差异只体现在「实例化参数」和「Document 记录的信息」上，例如：
     * CSVLoader      : 可额外传解析指定列、数据来源列、分隔符符号
     * DirectoryLoader: 传目录路径 + 需解析的文件列表 / glob
     * JSONLoader     : 传文件路径 + 提取 json 数据的 jq 结构表达式
   - 集成清单：https://imooc-langchain.shortvar.com/docs/integrations/document_loaders/

2. Markdown 与 UnstructuredMarkdownLoader
   - Markdown 是一种轻量级标记语言，可用纯文本编辑器创建格式化文本（课程电子书即用此格式）
   - LangChain 封装的 UnstructuredMarkdownLoader 专门用于加载 .md 文件
   - 基于 unstructured 库，能识别 Markdown 的结构化元素（标题、列表、代码块等）
   - 默认把整篇 Markdown 解析为纯文本，去除语法标记

3. unstructured 库：加载器的底层核心
   - 一款开源的非结构化数据预处理工具，旨在简化结构/非结构化文档的预处理
   - 内置读取和预处理图像与文本文档（PDF、HTML、Word 等）的开源组件
   - 也是 LangChain 文档加载器的核心：绝大部分加载器都基于 unstructured 封装
   - 使用前必须安装：pip install unstructured

4. mode 参数与「元素(element)」机制（重点）
   - 默认（不传 mode）UnstructuredMarkdownLoader 把整个文件装进「一个」Document，
     列表只有一个元素，page_content 是整篇 Markdown 的全部内容
   - 但幕后 unstructured 其实已为不同文本块创建了不同的「元素」，默认只是全部合并
   - 传 mode="elements" 可让所有元素分离：
       loader = UnstructuredMarkdownLoader("./项目API资料.md", mode="elements")
     一份文档可能被拆成几十个 Document（课程示例拆出 72 个）

5. 典型输出示例（mode="elements" 时）
   文档数量: 72
   page_content='LLMOps 项目 API 文档'
   metadata={'source': './LLMOps 项目 API 文档（资料）.md',
             'last_modified': '2024-07-05T10:41:07', 'page_number': 1,
             'languages': ['eng'], 'filetype': 'text/markdown',
             'file_directory': '.', 'filename': 'LLMOps 项目 API 文档（资料）.md',
             'category': 'Title'}
   page_content='应用 API 接口统一以 JSON 格式返回...'
   metadata={..., 'parent_id': 'b7210d8e5b8b15feccc935fd705f763b',
             'category': 'NarrativeText'}
   - metadata 字段含义：
     * source        : 数据来源路径
     * last_modified : 文件最后修改时间
     * page_number   : 所在页码
     * languages     : unstructured 识别出的语言
     * filetype      : MIME 类型
     * file_directory / filename : 目录与文件名
     * category      : 元素类型（Title 标题 / NarrativeText 叙述文本 等）
     * parent_id     : 父元素 id，体现元素的层级归属

6. 工程建议：不要在加载阶段做分割
   - 虽然 mode="elements" 能拆分，但一般加载文件为文档时很少这么做
   - 原因：在文档加载器中执行分割没法保证操作的一致性——
     没法确保所有传递文档的分割统一，分割出来的文档块大小不一，使用不便
   - 正确做法：加载阶段只负责「读全」，分割交给下游专门的 TextSplitter 统一处理

7. 使用场景
   - 加载技术文档、API 文档、README 文件
   - 构建知识库、文档问答系统
   - 需要保留文档结构信息的场景

===================================================================================
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# UnstructuredMarkdownLoader(): Markdown 文档加载器
#   参数:
#     - file_path: Markdown 文件路径
#     - mode: 加载模式，默认 "single" 将整个文件作为一个文档
#            "elements" 模式会将文档按元素（标题、段落等）分割
#   返回: UnstructuredMarkdownLoader 实例
#   作用: 解析 Markdown 文件，提取结构化内容
loader = UnstructuredMarkdownLoader("./项目API资料.md")

# load(): 执行文档加载
#   返回: List[Document]
#   作用: 解析 Markdown 语法，将内容转换为纯文本格式
documents = loader.load()

# 输出加载的文档对象列表
print(documents)

# 输出文档数量
print(len(documents))

# 输出元数据信息
# 通常包含 source（文件路径）、category（元素类型）等
print(documents[0].metadata)
