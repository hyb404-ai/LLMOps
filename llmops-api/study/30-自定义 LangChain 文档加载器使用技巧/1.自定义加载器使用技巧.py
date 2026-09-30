#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 8:32
@Author  : thezehui@gmail.com
@File    : 1.自定义加载器使用技巧.py

===================================================================================
知识点讲解：自定义文档加载器的实现
===================================================================================

1. 自定义加载器的必要性
   - 企业内部的数据库、API 接口等数据定制化非常强，用通用文档加载器提取，
     虽然能拿到记录，但加载出来的数据格式或样式大概率无法满足需求
   - 典型例子：用 WebBaseLoader 加载慕课网首页，会提取到大量空白数据（空格、换行、Tab），
     这类脏数据分割后存进向量数据库，会极大降低检索与生成的效率和正确性
   - 因此当通用加载器不够用时，应继承 BaseLoader 实现自己的解析逻辑

2. BaseLoader 基类与必须实现的方法
   - langchain_core.document_loaders.BaseLoader 是所有加载器的抽象基类，定义统一接口
   - 在 LangChain 中实现自定义文档加载器非常简单，只需两步：
     * 继承 BaseLoader 基类
     * 实现 lazy_load() 方法
   - 如果该加载器有异步使用场景，还需要额外实现 alazy_load() 方法
   - load() 已在基类中实现，内部调用 lazy_load() 并把结果收集成列表，无需自己写

3. 实现示例：逐行读取文本，每行为一个 Document
   - 同步版本：
       class CustomDocumentLoader(BaseLoader):
           def __init__(self, file_path: str) -> None:
               self.file_path = file_path

           def lazy_load(self) -> Iterator[Document]:
               with open(self.file_path, encoding="utf-8") as f:
                   line_number = 0
                   for line in f:
                       yield Document(
                           page_content=line,
                           metadata={"line_number": line_number, "source": self.file_path}
                       )
                       line_number += 1
   - 异步版本（不实现则委托 lazy_load()，即同步降级）：
       async def alazy_load(self) -> AsyncIterator[Document]:
           import aiofiles
           async with aiofiles.open(self.file_path, encoding="utf-8") as f:
               line_number = 0
               async for line in f:
                   yield Document(
                       page_content=line,
                       metadata={"line_number": line_number, "source": self.file_path}
                   )
                   line_number += 1
   - 依赖说明：异步版本需要 aiofiles 库，用 import 放在函数内可避免成为硬依赖

4. lazy_load() 与 load() 的区别
   - lazy_load()：返回生成器（Iterator[Document]），延迟加载，内存友好
   - load()：返回列表（List[Document]），一次性把所有文档载入内存
   - 大文件场景优先实现 lazy_load()，框架自动提供 load()；小文件两者皆可

5. yield 生成器的使用
   - 用 yield 逐条返回 Document 对象，而不是一次性 return 列表
   - 逐行 / 逐条处理，避免一次性把大文件读进内存
   - 适合流式处理和内存受限场景

6. alazy_load() 异步加载
   - 使用 async/await 语法实现异步 IO，需 aiofiles 等异步库支持
   - 适合网络请求、数据库查询等 IO 密集型场景
   - 不实现时，基类会委托 lazy_load() 实现（同步降级）

7. metadata 的设计
   - 应包含足够上下文信息（来源、行号、时间戳等），用于文档追溯与检索过滤
   - 保持字段结构一致，便于后续处理
   - 课程示例为每行 Document 写入 line_number 与 source（来源路径）两个字段，
     检索时即可按来源过滤或定位原文

8. 典型输出示例与观察点
   - 用 CustomDocumentLoader("./喵喵.txt") 加载一个 3 行的文本文件：
       [Document(page_content='喵喵🐱\n', metadata={'line_number': 0, 'source': './喵喵.txt'}),
        Document(page_content='喵喵🐱\n', metadata={'line_number': 1, 'source': './喵喵.txt'}),
        Document(page_content='喵😻😻',   metadata={'line_number': 2, 'source': './喵喵.txt'})]
       3
       {'line_number': 0, 'source': './喵喵.txt'}
   - 观察点 1：文件有几行就生成几个 Document，一一对应
   - 观察点 2：每个 Document 的 line_number 从 0 递增，source 记录来源路径
   - 观察点 3：load() 返回的列表长度等于文件行数

9. 扩展思考：文档加载器 = 二进制读取 + 解析逻辑
   - 观察 lazy_load() 的两个核心步骤：读取文件数据、把文件数据解析成 Document，
     绝大部分文档加载器都有这两个步骤，而且「读取文件数据」这步大家都大差不差
   - *.md、*.txt、*.py 等文本文件，乃至 *.pdf、*.doc 等非文本文件，
     都可以使用同一个「读取文件数据」步骤把文件读成二进制内容，
     再使用不同的解析逻辑来解析对应的二进制内容
   - 由此可得抽象：文档加载器 = 二进制数据读取 + 解析逻辑
   - 工程建议：如果项目中大量配置自定义文档解析器，把解析逻辑与加载逻辑分离，
     维护起来更容易，也更容易复用（具体取决于开发取舍）

10. 与 Blob 方案的关系
    - 基于上述拆分，所有 DocumentLoader 可以共用 Blob（数据读取），
      每个加载器内部只实现不同的 parse 即可
    - 这正是 LangChain 正在设计的新方案，也是下一章 Blob 与 BlobParser 的由来
    - 选择建议：需求简单、来源单一时直接继承 BaseLoader 即可；
      解析逻辑多、来源杂时再考虑 Blob 方案（见 31-Blob与BlobParser代替文档加载器）

===================================================================================

===================================================================================
"""
from typing import Iterator, AsyncIterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomDocumentLoader(BaseLoader):
    """自定义文档加载器，将文本文件的每一行都解析成Document"""

    def __init__(self, file_path: str) -> None:
        """
        构造函数

        参数:
            file_path: 要加载的文件路径
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载方法，逐行读取文件并生成 Document 对象

        返回:
            Iterator[Document]: Document 对象的生成器

        作用:
            - 按行读取文件，每一行生成一个独立的 Document
            - 使用生成器模式，内存占用小，适合大文件
        """
        # 1.读取对应的文件
        # 使用 with 语句确保文件正确关闭
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            # 2.提取文件的每一行
            for line in f:
                # 3.将每一行生成一个Document实例并通过yield返回
                # yield 使这个方法成为生成器，支持惰性加载
                yield Document(
                    page_content=line,
                    # metadata 包含来源和行号信息，便于追溯
                    metadata={"score": self.file_path, "line_number": line_number}
                )
                line_number += 1

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        异步惰性加载方法，适用于异步 IO 场景

        返回:
            AsyncIterator[Document]: Document 对象的异步生成器

        作用:
            - 使用异步文件读取，提高 IO 效率
            - 适合处理多个文件或网络 IO 场景
        """
        # aiofiles 提供异步文件操作支持
        import aiofiles
        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            # async for 异步迭代文件行
            async for line in f:
                # 使用 yield 返回 Document 对象
                yield Document(
                    page_content=line,
                    metadata={"score": self.file_path, "line_number": line_number}
                )
                line_number += 1


# 创建自定义加载器实例
loader = CustomDocumentLoader("./喵喵.txt")

# load(): 调用 lazy_load() 并将所有 Document 收集到列表中
#   返回: List[Document]
#   作用: 一次性加载所有行到内存
documents = loader.load()

# 输出所有文档对象
print(documents)

# 输出文档数量（等于文件行数）
print(len(documents))

# 输出第一个文档的元数据
print(documents[0].metadata)
