#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 8:21
@Author  : thezehui@gmail.com
@File    : 2.程序代码递归分割示例.py

===================================================================================
知识点讲解：针对程序代码的递归分割（from_language）
===================================================================================

1. 代码分割的特殊需求
   - 保持函数、类的完整性，不能随意在代码体中切断
   - 需识别语言特定语法结构（类定义、函数定义、缩进方法）
   - 核心是「按语法边界切分」，而不是按自然文本断句

2. 衍生代码分割器的来源
   - RecursiveCharacterTextSplitter 的关键在于可传入不同优先级的分隔符列表
   - 其内部已为多种编程语言预置了分隔符列表，开箱即用
   - 这正是「递归字符分割器」能衍生出「代码分割器」的原因

3. Language 枚举支持的语言
   - 编程语言类型定义在 langchain_text_splitters.Language 枚举中
   - 涵盖约 25 种：CPP、GO、JAVA、KOTLIN、JS、TS、PHP、PYTHON、RUBY、RUST、SCALA、SWIFT、MARKDOWN、HTML、SOL、CSHARP、COBOL、C、LUA、PERL、HASKELL 等
   - 每种语言对应一份预置的分隔符优先级列表

4. get_separators_for_language 查看语言分隔符
   - 通过 RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON) 可打印该语言的分隔符
   - Python 返回：['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
   - 含义：优先切出所有类，再切函数，再切类方法、模块语句，最后才是行/词/字符

5. from_language 工厂方法
   - 推荐用 from_language(language=Language.PYTHON, ...) 构造，自动加载该语言最优分隔符
   - 内部自动设置 is_separator_regex=True（这些分隔符本身按正则语义编写）
   - 无需手动维护 separators 列表，避免手写出错

6. Python 分隔符优先级（示例）
   - "\nclass "：类定义（优先级最高）
   - "\ndef "：函数定义
   - "\n\tdef "：缩进的方法定义
   - "\n\n"：空行（段落）
   - "\n"：单行
   - " "：空格
   - ""：字符（最低优先级）

7. 典型输出示例与观察点
   - 对 demo.py 按 Language.PYTHON 分割，块大小同样逼近 chunk_size：
       块大小:151, 元数据:{'source': './demo.py'}
       块大小:439, 元数据:{'source': './demo.py'}
       块大小:499, 元数据:{'source': './demo.py'}
       ...（约 30 块，大小在 100~500 间）
   - 打印 chunks[2].page_content 可见其恰好切在 split_text 方法体内，印证「优先按函数/类边界切分」
   - 观察点：代码块因结构自带上下文，chunk_overlap 可设得比自然文本稍小

8. 代码分割最佳实践
   - 优先用 from_language() 而非手动配置分隔符
   - chunk_size 设大些（代码通常需要更大上下文，如 500~1000）
   - 与通用递归分割共用同一套预分割/递归/合并流程，区别只在分隔符列表

===================================================================================
"""
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 加载 Python 代码文件
loader = UnstructuredLoader("./demo.py")
documents = loader.load()

# RecursiveCharacterTextSplitter.from_language(): 创建语言特定的分割器
#   参数:
#     - language: Language 枚举值，指定编程语言类型
#       Language.PYTHON 表示 Python 语言
#     - chunk_size: 块大小，代码文件建议设置较大值（如 500-1000）
#     - chunk_overlap: 块重叠大小
#     - add_start_index: 是否添加起始位置索引
#   返回: RecursiveCharacterTextSplitter 实例
#   作用:
#     - 自动配置该语言的最优分隔符列表
#     - 优先在类定义、函数定义等语法边界分割
#     - 保持代码结构的完整性
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# split_documents(): 执行代码分割
#   返回: List[Document]
#   工作流程:
#     1. 优先在 class 定义处分割
#     2. 其次在 def 函数定义处分割
#     3. 然后在空行处分割
#     4. 最后在行、词、字符处分割
chunks = text_splitter.split_documents(documents)

# 输出分块信息
for chunk in chunks:
    print(f"块大小: {len(chunk.page_content)}, 元数据: {chunk.metadata}")

# 输出第3个块的具体内容，查看分割效果
print(chunks[2].page_content)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
