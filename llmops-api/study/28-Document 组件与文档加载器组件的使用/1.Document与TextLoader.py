#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/1 15:27
@Author  : thezehui@gmail.com
@File    : 1.Document与TextLoader.py

===================================================================================
知识点讲解：LangChain Document 组件与文档加载器基础
===================================================================================

1. Document 组件的核心概念
   - Document 是 LangChain 中表示文档的基础数据结构
   - 包含两个核心属性：
     * page_content: 文档的文本内容（字符串类型）
     * metadata: 文档的元数据信息（字典类型），如来源、页码、作者等
   - Document 是所有文档加载器的统一输出格式

2. 文档加载器（Document Loaders）的作用
   - 负责从各种数据源加载数据并转换为 Document 对象
   - LangChain 提供了丰富的内置加载器：文本文件、PDF、Word、网页等
   - 所有加载器都继承自 BaseLoader，提供统一的 load() 和 lazy_load() 接口

3. TextLoader 文本加载器
   - 用于加载纯文本文件（.txt）
   - 默认将整个文件作为一个 Document 对象
   - 需要指定正确的编码格式（如 utf-8）避免乱码

4. 文档加载器的两种加载方式
   - load(): 一次性加载所有文档到内存，返回 List[Document]
   - lazy_load(): 惰性加载，返回生成器，适合处理大文件

5. metadata 元数据的重要性
   - 用于追溯文档来源（source 字段）
   - 在向量检索、文档分块时保留上下文信息
   - 可以用于文档过滤和分类

6. Document 在 RAG 组件链路中的定位（关键）
   - 一句话公式：Document = page_content(页面内容) + metadata(元数据)
   - Document 是 文档加载器 → 文档分割器 → 向量数据库 → 检索器 这四个组件
     之间「交互传递的状态数据」，相当于 RAG 流水线上的统一数据载体
   - 正因为有了这个统一结构，上游换任何加载器、下游换任何向量库都不影响链路
   - 版本差异：LangChain 旧版本的 Document 还支持 lookup 检索功能，
     新版本中 Document 只保留最基础的「记录信息」职责

7. 为什么 RAG 里几乎不手动录入数据
   - 前面课时为了演示用手动输入构造 Document，但真实 RAG 开发中
     数据来源是 本地 markdown / HTML 网页 / PDF / DOC / URL 链接 等
   - 标准流程：读取数据 → 切割数据 → 文本嵌入 → 存储到向量数据库
   - 这个流程非常耗时（例如上传一个 30M 的文档，要跑完加载/切割/嵌入全过程），
     所以工程上通常把它放在 RAG 应用「外部」，用 队列 / 异步任务 来处理，
     而不是放在用户请求的同步链路里
   - 文档加载器在这条链路里的职责：从各式各样的数据中提取信息并转换成标准
     Document，从而「屏蔽不同类型文件的读取差异」

8. BaseLoader 基类封装的 5 个统一方法
   - load()          : 加载文档，返回 List[Document]
   - aload()         : load 的异步版本
   - load_and_split(): 传入一个分割器，加载并按该分割器切割，返回切割后的文档列表
   - lazy_load()     : 懒加载，返回迭代器。适用于数据源包含多份文档的情况
                       （例如文件夹加载器，可以边加载边消费，不必等全部加载完）
   - alazy_load()    : lazy_load 的异步版本
   - LangChain 内置了上百种文档加载器，几乎所有常见文件都不需要自己封装
   - 集成清单：https://imooc-langchain.shortvar.com/docs/integrations/document_loaders/

9. TextLoader 的适用边界与源码实现
   - 适用：源码、markdown、text 等以「文本结构」存储的文件
   - 不适用：DOC/PDF 等二进制格式（它们不是文本文件，需要专用加载器）
   - 行为：把整个文件内容读入「一个」Document，并为 metadata 写入 source 字段
   - 核心源码（langchain_community/document_loaders/text.py -> TextLoader::lazy_load）：
       def lazy_load(self) -> Iterator[Document]:
           text = ""
           try:
               with open(self.file_path, encoding=self.encoding) as f:
                   text = f.read()
           except UnicodeDecodeError as e:
               if self.autodetect_encoding:
                   detected_encodings = detect_file_encodings(self.file_path)
                   for encoding in detected_encodings:
                       try:
                           with open(self.file_path, encoding=encoding.encoding) as f:
                               text = f.read()
                           break
                       except UnicodeDecodeError:
                           continue
               else:
                   raise RuntimeError(f"Error loading {self.file_path}") from e
           except Exception as e:
               raise RuntimeError(f"Error loading {self.file_path}") from e

           metadata = {"source": str(self.file_path)}
           yield Document(page_content=text, metadata=metadata)
   - 可以看到本质就是：open 读取 + 包一层 Document，并把路径写进 metadata
   - autodetect_encoding=True 时，遇到 UnicodeDecodeError 会尝试自动探测编码重读

10. 典型输出示例（对应下方三个 print）
    [Document(page_content='xxx', metadata={'source': './电商产品数据.txt'})]
    1
    {'source': './电商产品数据.txt'}
    - 说明：TextLoader 只产出 1 个 Document，metadata 只有 source 一个字段

11. 举一反三
    - 以 TextLoader 为样板，LangChain 其他所有加载器的使用范式完全一致：
      实例化时传递对应信息（文件路径 / 网址 / 目录等）→ 调用 load() 一键加载
    - 差异只在于「构造参数」和「metadata 记录的字段丰富程度」

===================================================================================
"""
from langchain_community.document_loaders import TextLoader

# 1.构建加载器
# TextLoader(): 文本文件加载器
#   参数:
#     - file_path: 文件路径（相对或绝对路径）
#     - encoding: 文件编码格式，默认 None 会自动检测，建议显式指定避免乱码
#   返回: TextLoader 实例
loader = TextLoader("./电商产品数据.txt", encoding="utf-8")

# 2.加载数据
# load(): 加载文件内容并解析为 Document 对象列表
#   参数: 无
#   返回: List[Document]，通常文本文件只返回一个 Document
#   作用: 读取整个文件内容，将其封装为 Document 对象
documents = loader.load()

# 输出完整的 Document 对象列表
print(documents)

# 输出加载的文档数量（TextLoader 默认将整个文件作为一个文档）
print(len(documents))

# 输出第一个文档的元数据信息
# metadata 通常包含 source（文件路径）等信息
print(documents[0].metadata)
