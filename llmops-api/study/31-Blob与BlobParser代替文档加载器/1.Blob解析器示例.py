#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 10:51
@Author  : thezehui@gmail.com
@File    : 1.Blob解析器示例.py

===================================================================================
知识点讲解：Blob 与 BlobParser 的使用
===================================================================================

1. 设计动机：为什么要把「加载」与「解析」拆开
   - 许多文档加载器都涉及解析文件，这类加载器之间的差异通常源于「文件解析方式」，
     而不是「文件加载方式」
   - 例如你可以用 open() 读取 PDF 或 Markdown 的二进制内容，
     但需要不同的解析逻辑把二进制数据转换为文本
   - 由此可得出一个关键等式：
       文档加载器 = 二进制数据读取 + 解析逻辑
   - 拆开的好处：读取步骤高度同质（文本文件、PDF、Doc 都能读成二进制），
     解析步骤各不相同；分离后解析逻辑与加载逻辑各自可复用、易维护

2. Blob 方案的定位与灵感来源
   - LangChain 提供的 Blob 方案，灵感来源于前端浏览器中的 Blob WebAPI 规范
     （参考 https://developer.mozilla.org/en-US/docs/Web/API/Blob）
   - 该方案由三个类构成：
     * Blob：LangChain 封装的数据对象，通过引用或值表示原始数据，
       提供一个接口以表示不同形式具体化的二进制数据，
       使用该类有助于「将数据加载器的开发与解析器解耦」
     * BlobLoader：Blob 数据加载器，类似 DocumentLoader，
       不过被设计成可以加载任何数据（未来规划）
     * BlobParser：Blob 数据解析器，用于将传入的 Blob 数据转换成文档列表
   - 引入 Blob 后，文档加载器的运行流程变为：
       BlobLoader 加载 → Blob（原始二进制）→ BaseBlobParser 解析 → Document 列表

3. Blob 的核心概念
   - Blob（Binary Large Object）表示二进制数据对象，是文档加载流程中的中间层
   - 包含数据本身与元数据（路径、MIME 类型、编码等）
   - 关系：Blob → Parser → Document（原始数据 → 结构化文档）

4. Blob 的常用属性与方法
   - 属性：
     * data     ：原始数据，支持存储字节或字符串
     * mimetype ：文件的 mimetype 类型
     * encoding ：文件编码，默认 utf-8
     * path     ：文件原始路径，支持字符串路径或 Path 对象
     * metadata ：存储的元数据，一般都有 source 字段
   - 方法：
     * source()      ：只读属性/函数，返回数据来源
     * as_string()   ：将数据转换成字符串
     * as_bytes()    ：将数据转换成字节数据
     * as_bytes_io() ：将数据转换成缓冲流字节数据
     * from_path()   ：从路径加载 Blob 数据（文件），最常用
     * from_data()   ：从原始数据加载 Blob 数据（非文件）
   - 两种构造写法：
       blob = Blob.from_path("./喵喵.txt")
       blob = Blob(data="喵喵🐱\r\n喵喵🐱\r\n喵😻😻")   # 直接从内存加载，无需文件

5. Blob vs Document
   - Blob：原始二进制数据的容器，未经解析，偏「数据源」
   - Document：解析后的结构化文档对象，偏「可检索内容」
   - 二者是流水线上的上下游，不是替代关系

6. BlobParser 的作用与自定义写法
   - 负责将 Blob 解析为 Document 对象，实现「数据读取」与「解析逻辑」的解耦
   - 自定义解析器只需继承 BaseBlobParser 并实现 lazy_parse(blob) 方法：
       class CustomParser(BaseBlobParser):
           def lazy_parse(self, blob: Blob) -> Iterator[Document]:
               line_number = 0
               with blob.as_bytes_io() as f:
                   for line in f:
                       yield Document(
                           page_content=line,
                           metadata={"source": blob.source, "line_number": line_number}
                       )
                       line_number += 1
   - 使用方式：
       blob = Blob.from_path("./喵喵.txt")
       documents = list(parser.lazy_parse(blob))

7. BaseBlobParser 基类与惰性解析
   - lazy_parse(blob) -> Iterator[Document]：核心方法，接收 Blob 返回 Document 生成器
   - parse()：调用 lazy_parse() 并返回列表
   - 使用生成器模式，支持大文件惰性处理，避免一次性载入内存

8. 典型输出示例与观察点
   - 逐行解析的输出（每行一个 Document）：
       [Document(page_content='喵喵🐱\r\n', metadata={'source': './喵喵.txt', 'line_number': 0}),
        Document(page_content='喵喵🐱\r\n', metadata={'source': './喵喵.txt', 'line_number': 1}),
        Document(page_content='喵😻😻', metadata={'source': './喵喵.txt', 'line_number': 2})]
       3
       {'source': './喵喵.txt', 'line_number': 0}
   - 观察点 1：metadata 中的 line_number 由解析器自己塞入，
     Blob 本身只提供 source，行号这类业务语义归解析层负责
   - 观察点 2：page_content 保留了行尾的 \r\n，说明解析器不做清洗；
     若需要干净文本，应在解析后再接文档转换器（见 36-非分割类型的文档转换器使用技巧）

9. 设计模式优势
   - 分离关注点：Blob 负责数据源，Parser 负责解析逻辑
   - 灵活组合：不同的 BlobLoader + BlobParser 可任意搭配
   - 易于测试：可单独构造 Blob 测试解析逻辑，不依赖真实文件系统

10. 现状评估（选型参考）
    - Blob 方案目前 LangChain 封装与集成得非常少，内置加载器只有 FileSystemBlobLoader，
      内置解析器也相当有限
    - 现阶段若要使用 Blob 形式加载文件，仍需大量自行编写加载与解析逻辑，效率偏低
    - 随着官方封装的解析逻辑增多，该方案未来会逐步替代 DocumentLoader
    - 当前版本的定位：知道有这套机制、理解其解耦思想即可，
      生产环境仍以成熟的 DocumentLoader 为主

===================================================================================

===================================================================================
"""
from typing import Iterator

from langchain_core.document_loaders import Blob
from langchain_core.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document


class CustomParser(BaseBlobParser):
    """自定义解析器，用于将传入的文本二进制数据的每一行解析成Document组件"""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        惰性解析方法，逐行解析 Blob 数据

        参数:
            blob: Blob 对象，包含要解析的二进制数据

        返回:
            Iterator[Document]: Document 对象的生成器

        作用:
            - 将 Blob 的二进制数据按行解析为 Document
            - 每一行生成一个独立的 Document 对象
        """
        line_number = 0
        # as_bytes_io(): 将 Blob 数据转换为 BytesIO 对象
        # 返回类似文件对象的接口，支持迭代读取
        with blob.as_bytes_io() as f:
            # 逐行读取二进制数据
            for line in f:
                # 使用 yield 返回 Document 对象，实现惰性加载
                yield Document(
                    page_content=line,
                    # blob.source 是 Blob 的来源路径
                    metadata={"source": blob.source, "line_number": line_number}
                )
                line_number += 1


# 1.加载blob数据
# Blob.from_path(): 从文件路径创建 Blob 对象
#   参数:
#     - path: 文件路径
#   返回: Blob 实例，包含文件的二进制数据和元数据
#   作用: 读取文件内容到内存，封装为 Blob 对象
blob = Blob.from_path("./喵喵.txt")

# 创建自定义解析器实例
parser = CustomParser()

# 2.解析得到文档数据
# lazy_parse(): 返回 Document 生成器
# list(): 将生成器转换为列表，一次性获取所有 Document
documents = list(parser.lazy_parse(blob))

# 3.输出相应的信息
# 输出所有文档对象
print(documents)

# 输出文档数量（等于文件行数）
print(len(documents))

# 输出第一个文档的元数据
# 包含 source（来自 blob）和 line_number
print(documents[0].metadata)
