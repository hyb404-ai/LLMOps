#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 14:39
@Author  : thezehui@gmail.com
@File    : 1.递归字符文本分割器示例.py

===================================================================================
知识点讲解：递归字符文本分割器（RecursiveCharacterTextSplitter）
===================================================================================

1. 设计动机：为什么需要递归字符分割器
   - 朴素 CharacterTextSplitter 只能用单一分隔符，会出现两类问题使 RAG 不可控：
     * 块过大：极端时单块超过 LLM 上下文上限，永远不会被检索引用，相当于存了却丢了数据
     * 块过小：信息密度过低，填进 Prompt 后 LLM 也提不出有用信息
   - 理想方案：按分隔符初切后检测块大小，过大则按备选分隔符二次切，过小则合并前后块，让所有块尽量逼近指定长度

2. 组件定位：RecursiveCharacterTextSplitter
   - LangChain 提供的「递归字符串分割」组件，核心能力 = 一组分隔符 + 块大小约束
   - 运行三阶段（RAG 高频考点）：预分割 → 大文档块递归分割 → 小文档块合并
   - 最终文档块大小并不完全相同，但会逼近指定长度（仍有小概率远小于目标）

3. 递归工作原理
   - 维护一个「语义粒度从粗到细」的分隔符优先级列表，默认 ["\n\n", "\n", " ", ""] 对应 段落 → 行 → 词 → 字符
   - 优先用最粗粒度切开以保留语义；若某块仍超 chunk_size，就对这一块递归换用更细分隔符
   - 降级到 "" 即按单字符强制切分，保证一定能满足大小约束——这是「递归」二字来源

4. 默认分隔符优先级详解
   - "\n\n"：段落边界（优先级最高）
   - "\n"：行边界
   - " "：词边界
   - ""：字符边界（最后手段，会把中文切成单字、英文切成单字母，几乎丢失语义）

5. 核心参数
   - separators：分隔符列表（可选），默认 ["\n\n", "\n", " ", ""]，按优先级从高到低尝试
   - chunk_size：单块最大字符数（示例 500）
   - chunk_overlap：相邻块重叠大小（示例 50），避免语义在边界断裂
   - add_start_index：是否记录原文起始位置索引（体现在 metadata.start_index 中）

6. 与 CharacterTextSplitter 的对比
   - CharacterTextSplitter：单一分隔符，可能破坏语义或产生超大/超碎块，不推荐通用使用
   - RecursiveCharacterTextSplitter：智能选择分割点、保持语义完整，是通用首选

7. 典型输出示例与观察点
   - 对一份 Markdown API 文档按 chunk_size=500 分割，得到大小接近但略有差异的块：
       块大小:251, 块元数据:{'source': './项目API文档.md', 'start_index': 0}
       块大小:451, 块元数据:{'source': './项目API文档.md', 'start_index': 246}
       块大小:490, 块元数据:{'source': './项目API文档.md', 'start_index': 699}
       ...（后续块大小在 90~500 间波动）
   - 观察点 1：各块长度都 ≤ 500 但彼此不同，印证「逼近而非相等」
   - 观察点 2：metadata 带有 start_index，可用于回溯原文位置

8. 适用场景
   - 通用文本分割（文章、文档、书籍）
   - 需要保持语义完整性的 RAG 知识库预处理
   - 代码分割（配合语言特定分隔符，见 2.程序代码递归分割示例.py）

===================================================================================
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 Markdown 文档
loader = UnstructuredMarkdownLoader("./项目API文档.md")
documents = loader.load()

# RecursiveCharacterTextSplitter(): 递归字符文本分割器
#   参数:
#     - separators: 分隔符列表（可选），默认 ["\n\n", "\n", " ", ""]
#       按优先级从高到低尝试，找到最合适的分割点
#     - chunk_size: 每块最大字符数，500 表示每块不超过 500 字符
#     - chunk_overlap: 块之间的重叠大小，50 表示相邻块有 50 字符重叠
#     - add_start_index: 是否添加原文起始位置索引
#   返回: RecursiveCharacterTextSplitter 实例
#   作用:
#     - 智能选择分割点，优先保持段落、句子完整性
#     - 递归处理超大块，确保所有块都满足大小限制
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# split_documents(): 执行递归分割
#   参数:
#     - documents: Document 对象列表
#   返回: List[Document]，分割后的文档块
#   工作流程:
#     1. 尝试用 "\n\n" 分割
#     2. 如果块仍然太大，递归用 "\n" 分割
#     3. 继续递归用 " " 和 "" 分割
#     4. 确保所有块满足 chunk_size 限制
chunks = text_splitter.split_documents(documents)

# 遍历分块，输出每块的大小和元数据
for chunk in chunks:
    print(f"块大小: {len(chunk.page_content)}, 元数据: {chunk.metadata}")
