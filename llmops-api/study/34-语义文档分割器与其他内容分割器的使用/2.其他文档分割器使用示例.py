#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 14:55
@Author  : thezehui@gmail.com
@File    : 2.其他文档分割器使用示例.py

===================================================================================
知识点讲解：HTML 标题文本分割器（HTMLHeaderTextSplitter）
===================================================================================

1. 组件定位：HTML 标题分割器（HTMLHeaderTextSplitter）
   - 专门处理 HTML 文档，基于标题标签（h1-h6）层级分割
   - 与字符分割器不同，它保留文档的逻辑层级结构，并把标题信息写入 metadata
   - 同类还有 HTMLSectionSplitter（只向上找最近的副标题即停止）与 MarkdownHeaderTextSplitter（Markdown 标题同理）

2. 为什么需要按标题分割
   - HTML 通过 h1-h6 组织内容层次：h1 是主标题，h2-h6 为子标题
   - 按标题切分能保持内容逻辑结构，避免把同一小节内容拆散
   - 关键认知：层级关系是「目录导航」视角——内容落在哪个标题下就视为被嵌套到哪个标题下，与实际 DOM 嵌套无关

3. headers_to_split_on 参数
   - 定义要识别的标题级别与对应 metadata 键名，格式 List[Tuple[str, str]]
   - 第一个元素是 HTML 标签名（如 "h1"），第二个是 metadata 键名（如 "一级标题"）
   - 本例配置 h1/h2/h3 三级，分别存为「一级标题 / 二级标题 / 三级标题」

4. 元数据层级与查询逻辑
   - 每个块的 metadata 包含所属各级标题，例如 {"一级标题": "标题1", "二级标题": "子标题1"}
   - HTMLHeaderTextSplitter 会「顺序往上逐层查找」所有嵌套层级标题，把祖先标题都带进 metadata
   - 这便于理解文档结构与上下文，是它区别于 HTMLSectionSplitter 的核心点

5. 典型输出示例与观察点
   - 对示例 HTML 分割得到：
       Document(metadata={'一级标题': '标题1'}, page_content='关于标题1的一些介绍文本。')
       Document(metadata={'一级标题': '标题1', '二级标题': '子标题1', '三级标题': '子子标题1'}, page_content='关于子子标题1的一些文本。')
   - 观察点 1：纯标题文本（如 '标题1'）也会成为独立块，且其 metadata 只带自身级标题
   - 观察点 2：同一内容会随所在层级携带不同深度的 metadata，层级越深祖先越多

6. split_text() vs split_documents()
   - split_text()：接收 HTML 字符串，返回 Document 列表（HTML 场景常用）
   - split_documents()：接收 Document 列表，HTML 场景下不常用

7. 使用场景与同类工具
   - 场景：网页内容（博客、文档站）、在线帮助中心等需要保留结构的索引
   - 同类：HTMLSectionSplitter（按 section 标签、近副标题即停）、MarkdownHeaderTextSplitter（Markdown 标题）、RecursiveCharacterTextSplitter.from_language(Language.HTML)

===================================================================================
"""
from langchain_text_splitters import HTMLHeaderTextSplitter

# 1.构建文本与分割标题
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>标题1</h1>
        <p>关于标题1的一些介绍文本。</p>
        <div>
            <h2>子标题1</h2>
            <p>关于子标题1的一些介绍文本。</p>
            <h3>子子标题1</h3>
            <p>关于子子标题1的一些文本。</p>
            <h3>子子标题2</h3>
            <p>关于子子标题2的一些文本。</p>
        </div>
        <div>
            <h3>子标题2</h2>
            <p>关于子标题2的一些文本。</p>
        </div>
        <br>
        <p>关于标题1的一些结束文本。</p>
    </div>
</body>
</html>
"""

# 定义要识别的标题级别和对应的 metadata 键名
# headers_to_split_on: 标题分割配置列表
#   每个元组包含:
#     - HTML 标签名（如 "h1", "h2", "h3"）
#     - metadata 中的键名（用于存储该级别标题内容）
#   作用: 告诉分割器要识别哪些标题级别，以及如何命名
headers_to_split_on = [
    ("h1", "一级标题"),  # h1 标签的内容存储为 "一级标题"
    ("h2", "二级标题"),  # h2 标签的内容存储为 "二级标题"
    ("h3", "三级标题"),  # h3 标签的内容存储为 "三级标题"
]

# 2.创建分割器并分割
# HTMLHeaderTextSplitter(): HTML 标题文本分割器
#   参数:
#     - headers_to_split_on: 标题分割配置列表
#     - return_each_element: 是否为每个 HTML 元素创建独立 Document（默认 False）
#   返回: HTMLHeaderTextSplitter 实例
#   作用:
#     - 解析 HTML 文档
#     - 识别指定的标题标签（h1-h6）
#     - 根据标题层级分割内容
text_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

# split_text(): 分割 HTML 文本
#   参数:
#     - text: HTML 字符串
#   返回: List[Document]
#   工作流程:
#     1. 解析 HTML 结构
#     2. 识别 h1, h2, h3 标签
#     3. 在标题边界处分割内容
#     4. 将标题信息存储到 metadata 中
#     5. 每个块包含标题到下一个同级/上级标题之间的内容
chunks = text_splitter.split_text(html_string)

# 3.输出分割内容
# 每个块的 metadata 包含其所属的各级标题
for chunk in chunks:
    print(chunk)
