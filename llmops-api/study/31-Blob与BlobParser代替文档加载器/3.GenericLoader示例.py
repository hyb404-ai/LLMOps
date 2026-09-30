#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 11:16
@Author  : thezehui@gmail.com
@File    : 3.GenericLoader示例.py

===================================================================================
知识点讲解：GenericLoader 通用加载器
===================================================================================

1. 设计动机：为什么需要 GenericLoader
   - 在前面的示例中，Blob 加载器与解析器是分开使用的：
     先 yield_blobs() 拿到 Blob，再手工喂给 Parser
   - GenericLoader 是 LangChain 封装的一个「由 BlobLoader 与 BaseBlobParser 组合而成」的类，
     旨在提供标准化的方法，让 BlobLoader 使用更简单
   - 它把「加载 → 解析」两步收敛成一个对象，调用方只面对一个 load()

2. 核心思想
   - 组合 BlobLoader + BlobParser 实现完整的加载流程
   - 提供统一加载接口，简化复杂场景
   - 实现「加载 → 解析」两阶段模式的对外透明化

3. from_filesystem() 工厂方法
   - 快速创建基于文件系统的加载器：
       loader = GenericLoader.from_filesystem(".", glob="*.txt", show_progress=True)
   - 内部组合 FileSystemBlobLoader + 指定的解析器
   - 支持 glob 模式与进度显示
   - 注意：当前 LangChain 仅集成了 FileSystemBlobLoader，
     因此 GenericLoader 目前也只支持文件系统这一种来源

4. 典型用法与输出示例
   - 代码：
       loader = GenericLoader.from_filesystem(".", glob="*.txt", show_progress=True)
       for idx, doc in enumerate(loader.lazy_load()):
           print(f"当前加载第{idx + 1}个文件，文件信息:{doc.metadata}")
   - 输出：
       100%|██████████| 1/1 [00:00<00:00, 19.76it/s]
       当前加载第1个文件，文件信息:{'source': '喵喵.txt'}
   - 观察点：glob="*.txt" 生效，目录下只有喵喵.txt 命中；
     metadata 直接带上了 source，无需解析器手工塞入

5. lazy_load() 的重要性
   - 返回 Document 生成器，按需加载
   - 避免大量文件同时载入内存
   - 适合处理 GB 级文档集合
   - 与 load() 的区别：load() 一次性返回 list，lazy_load() 惰性产出

6. GenericLoader 的优势
   - 自动处理 Blob → Document 转换，省去手工串联
   - 支持批量文件处理
   - 提供 lazy_load() 实现内存友好的流式处理

7. 使用场景
   - 批量导入文档到向量数据库
   - 构建知识库索引
   - 大规模文档预处理

8. 性能优化建议
   - 处理大量文件优先用 lazy_load() 而非 load()
   - 配合 glob / exclude 过滤减少不必要的加载
   - 考虑异步版本处理 IO 密集型任务

9. 现状评估
   - Blob 解决方案目前 LangChain 封装与集成得非常少，GenericLoader 也只支持
     FileSystemBlobLoader 一种组合
   - 现阶段使用 Blob 形式加载文件仍需大量自行编写逻辑，效率较低
   - 随着官方封装的 Blob 解析逻辑增多，GenericLoader 这类标准化入口会逐渐成熟
   - 当前建议：理解其组合思想即可，生产环境优先使用成熟的 DocumentLoader

===================================================================================

===================================================================================
"""
from langchain_community.document_loaders.generic import GenericLoader

# GenericLoader.from_filesystem(): 从文件系统创建通用加载器
#   参数:
#     - path: 目录路径，"." 表示当前目录
#     - glob: 文件匹配模式，"*.txt" 匹配所有 txt 文件
#     - suffixes: 文件后缀过滤（可选，与 glob 二选一）
#     - exclude: 排除的文件模式（可选）
#     - show_progress: 是否显示加载进度条
#     - parser: 自定义 BlobParser（可选），默认使用通用解析器
#   返回: GenericLoader 实例
#   作用: 组合 FileSystemBlobLoader 和解析器，提供完整的加载功能
loader = GenericLoader.from_filesystem(".", glob="*.txt", show_progress=True)

# lazy_load(): 惰性加载文档
#   返回: Iterator[Document]
#   作用: 按需加载文件并解析为 Document，内存占用小
#
# 使用 enumerate() 遍历文档并获取索引
for idx, doc in enumerate(loader.lazy_load()):
    # 输出加载进度和文件信息
    # metadata['source'] 包含文件路径
    print(f"当前正在加载第{idx}个文件,文件名:{doc.metadata['source']}")
