#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 11:10
@Author  : thezehui@gmail.com
@File    : 2.FileSystemBlobLoader示例.py

===================================================================================
知识点讲解：FileSystemBlobLoader 文件系统 Blob 加载器
===================================================================================

1. 组件定位：BlobLoader 这一层在做什么
   - 解析器封装了「将二进制数据解析为 Document」的逻辑，
     而 Blob 加载器封装的是「从给定存储位置加载 Blob」的逻辑
   - 二者分工明确：加载器只负责把数据读成 Blob，不负责解析
   - 目前在 LangChain 中只集成了一个实现：FileSystemBlobLoader（文件系统二进制数据加载器）

2. FileSystemBlobLoader 的作用
   - 从文件系统批量加载文件为 Blob 对象
   - 支持目录遍历与文件过滤（glob / exclude）
   - 只加载不解析，需配合 BlobParser 才能得到 Document

3. 核心参数
   - path：目标目录路径（"." 表示当前目录）
   - glob：文件匹配模式（如 "*.txt"、"**/*.py"）
   - exclude：排除的文件模式
   - show_progress：是否显示加载进度条

4. glob 模式语法
   - "*"：匹配任意字符（不含目录分隔符）
   - "**"：递归匹配所有子目录
   - "?"：匹配单个字符
   - "[abc]"：匹配字符集中的任意一个

5. 典型用法与输出示例
   - 代码：
       from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
       loader = FileSystemBlobLoader(".", show_progress=True)
       for blob in loader.yield_blobs():
           print(blob.source)
   - 输出：
       100%|██████████| 3/3 [00:00<00:00, 58.15it/s]
       1.Blob解析器示例.py
       2.FileSystemBlobLoader示例.py
       喵喵.txt
   - 观察点：进度条由 show_progress=True 触发；source 打印的是相对路径，
     与构造时传入的 path 保持一致

6. BlobLoader 抽象基类与自定义加载器（源码要点）
   - 自定义加载器只需继承 BlobLoader 并实现 yield_blobs()：
       class BlobLoader(ABC):
           # Abstract interface for blob loaders implementation.
           @abstractmethod
           def yield_blobs(self) -> Iterable[Blob]:
               # A lazy loader for raw data represented by LangChain's Blob object.
               ...
   - 基类文档要求实现者能「根据某些条件从存储系统加载原始内容，
     并以流的形式惰性返回 Blob」，因此自定义实现也应返回生成器而非列表
   - 目前官方仅内置 FileSystemBlobLoader 一个实现，
     数据库、对象存储等来源需要自行实现

7. yield_blobs() vs load()
   - yield_blobs()：返回 Blob 生成器，惰性加载，适合大量文件批处理
   - 配合 BlobParser 可实现流式「加载 → 解析」，内存占用恒定
   - 若需要一次性拿到全部结果，用 list() 包一层即可

8. as_string() 方法
   - 将 Blob 二进制数据转为字符串（自动检测或指定编码）
   - 适合文本文件直接读取，无需再写 open()

9. 典型使用场景
   - 批量加载目录下所有文档
   - 配合 GenericLoader 实现自动解析
   - 构建文档索引系统

10. 注意事项
    - 加载器只产出 Blob，不含解析结果；直接打印 blob 看不到文本内容，
      需要 as_string() 或交给 Parser
    - 大目录建议始终用 yield_blobs() 惰性迭代，避免一次性构造大量 Blob 对象
    - 官方实现较少，特殊存储来源需自行继承 BlobLoader 扩展

===================================================================================

===================================================================================
"""
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader

# FileSystemBlobLoader(): 文件系统 Blob 加载器
#   参数:
#     - path: 要扫描的目录路径，"." 表示当前目录
#     - glob: 文件匹配模式，默认 "*" 匹配所有文件
#     - exclude: 排除的文件模式列表（可选）
#     - show_progress: 是否显示进度条，默认 False
#   返回: FileSystemBlobLoader 实例
#   作用: 扫描指定目录，加载匹配的文件为 Blob 对象
loader = FileSystemBlobLoader(".", show_progress=True)

# yield_blobs(): 生成 Blob 对象的迭代器
#   返回: Iterator[Blob]
#   作用: 惰性加载目录中的文件，逐个返回 Blob 对象
#
# 使用 for 循环迭代所有 Blob 对象
for blob in loader.yield_blobs():
    # as_string(): 将 Blob 的二进制数据转换为字符串
    #   参数:
    #     - encoding: 编码格式，默认自动检测
    #   返回: str
    #   作用: 读取并解码文件内容为文本
    print(blob.as_string())
