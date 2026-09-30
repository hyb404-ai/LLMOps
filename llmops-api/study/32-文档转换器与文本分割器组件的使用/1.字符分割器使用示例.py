#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 13:16
@Author  : thezehui@gmail.com
@File    : 1.字符分割器使用示例.py

===================================================================================
知识点讲解：文档转换器与字符文本分割器
===================================================================================

1. 为什么需要文档转换与切割
   - 使用文档加载器加载得到的文档，一般存在几个问题：
     原始文档太大、原始文档的数据格式不符合需求（需要英文但只有中文）、
     原始文档的信息没有经过提炼
   - 如果把这类数据直接转换成向量存储到数据库，会导致执行相似性搜索和 RAG 时错误率大大提升
   - 所以在 LLM 应用开发中，加载完数据后一般会多执行一步「转换」：
     把加载得到的文档列表转换成符合需求的文档列表
   - 转换涵盖的操作非常多：文档切割、文档属性提取、文档翻译、HTML 转文本、
     重排、元数据标记等都属于转换

2. 文档转换器基类与分类
   - LangChain 针对文档转换统一封装了基类 BaseDocumentTransformer，
     所有涉及文档转换的类都是该类的子类（文档分割器也是它的子类实现）
   - 基类封装两个方法：
     * transform_documents()：抽象方法，传递文档列表，返回转换后的文档列表
     * atransform_documents()：异步实现，如果没有实现，则委托 transform_documents() 实现
   - 文档转换组件分两类：
     * 文档分割器：使用频率高，已单独拆分到 langchain-text-splitters 包
     * 文档处理转换器：使用频率低，老版本写法
   - 安装（分割器已独立成包，可脱离 LangChain 单独使用）：
       pip install -qU langchain-text-splitters
   - 文本分割器除继承 BaseDocumentTransformer 外，还单独设置了 TextSplitter 基类，
     以实现更丰富的功能

3. 为什么需要文本分割
   - LLM 有最大 token 限制（4K / 8K / 128K 等）
   - 向量检索需要合适粒度的文本块以提高精度与相关性
   - 控制成本与响应速度

4. CharacterTextSplitter 字符分割器
   - 最简单的分割器：基于给定的字符串进行分割，默认为 \n\n，
     并且在分割时会尽可能保证数据的连续性
   - 分割出来的每一块长度通过「字符数」衡量
   - 完整参数与默认值：
     * separator           ：分隔符，默认 \n\n
     * is_separator_regex  ：是否按正则解析分隔符，默认 False
     * chunk_size          ：每块文档的内容大小，默认 4000
     * chunk_overlap       ：块与块之间重叠的内容大小，默认 200
     * length_function     ：计算文本长度的函数，默认 len
     * keep_separator      ：是否将分隔符保留到分割的块中，默认 False
     * add_start_index     ：是否添加开始索引，默认 False，为 True 时会在元数据中
                             添加该切块的起点（start_index）
     * strip_whitespace    ：是否删除文档头尾的空白，默认 True

5. 核心参数详解
   - separator：分隔符（"\n\n" 按段落、"\n" 按行）
   - chunk_size：每块最大字符数
   - chunk_overlap：块间重叠字符数，保持上下文连贯，通常取 chunk_size 的 10%~20%
   - add_start_index：是否在 metadata 添加原文起始位置，便于回溯定位

6. 典型用法与输出示例
   - 将文档切割为不超过 500 字符、块间重叠 50 字符：
       loader = UnstructuredMarkdownLoader("./项目API文档.md")
       documents = loader.load()
       text_splitter = CharacterTextSplitter(
           separator="\n\n",
           chunk_size=500,
           chunk_overlap=50,
           add_start_index=True,
       )
       chunks = text_splitter.split_documents(documents)
   - 输出：
       Created a chunk of size 771, which is longer than the specified 500
       Created a chunk of size 980, which is longer than the specified 500
       Created a chunk of size 542, which is longer than the specified 500
       Created a chunk of size 835, which is longer than the specified 500
       块内容大小:251,元数据:{'source': './项目API文档.md', 'start_index': 0}
       块内容大小:451,元数据:{'source': './项目API文档.md', 'start_index': 246}
       块内容大小:771,元数据:{'source': './项目API文档.md', 'start_index': 699}
       块内容大小:435,元数据:{'source': './项目API文档.md', 'start_index': 1472}
       ...（共 16 块）
   - 观察点 1：即使设置 chunk_size=500，仍有 771 / 980 / 542 / 835 超过阈值，
     并触发告警——说明 chunk_size 是软约束而非硬截断
   - 观察点 2：add_start_index=True 让每块带上 start_index，
     与原始 metadata 的 source 合并保留，便于回溯原文位置

7. 底层分割原理（源码要点）
   - 核心流程（langchain_text_splitters/character）：
       def split_text(self, text: str) -> List[str]:
           # First we naively split the large input into a bunch of smaller ones.
           separator = (
               self._separator if self._is_separator_regex else re.escape(self._separator)
           )
           splits = _split_text_with_regex(text, separator, self._keep_separator)
           _separator = "" if self._keep_separator else self._separator
           return self._merge_splits(splits, _separator)
   - 即：先按分隔符（正则）把整篇文档拆成片段，再循环遍历片段列表逐个相加，
     直到最接近 chunk_size 窗口大小时完成一个 Document 的组装
   - 这就是为什么 chunk_size 无法被严格遵守：合并是以「分隔片段」为最小粒度的，
     若某片段本身已超过 chunk_size，则直接告警并单独成块
   - _split_text_with_regex 要点：
     * keep_separator=True 时用 re.split(f"({separator})", text)，
       通过捕获括号把分隔符保留在结果中
     * keep_separator=False 时直接 re.split(separator, text)
     * separator 为空字符串时退化为 list(text)，即逐字符拆分
     * 最后过滤掉空串

8. split_documents() vs split_text()
   - split_documents()：接收 Document 列表，返回分割后的 Document 列表，
     保留并合并原 metadata（source 等字段会继承到每个切块）
   - split_text()：接收纯文本字符串，返回文本字符串列表，不涉及 metadata

9. 注意事项与选型建议
   - 分隔符优先级比 chunk_size 更高：先切分后合并，chunk_size 只是合并的窗口上限
   - 若输入文档单段就远超 chunk_size，字符分割器无能为力，
     此时应选递归字符分割器（见 33-递归字符文本分割器的使用与运行流程）
   - 中文场景注意 length_function 默认按字符数计，与 token 数不等价，
     估算成本时需另行换算

===================================================================================

===================================================================================
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter

# 1.加载对应的文档
# UnstructuredMarkdownLoader: 加载 Markdown 文档
loader = UnstructuredMarkdownLoader("./项目API文档.md")

# load(): 返回 Document 对象列表
documents = loader.load()

# 2.创建文本分割器
# CharacterTextSplitter(): 字符文本分割器
#   参数:
#     - separator: 分隔符，用于确定分割边界
#       "\n\n" 表示按空行（段落）分割
#     - chunk_size: 每个文本块的最大字符数
#       500 表示每块不超过 500 个字符
#     - chunk_overlap: 相邻块之间的重叠字符数
#       50 表示相邻块有 50 个字符重叠，保持上下文连贯
#     - add_start_index: 是否在 metadata 中添加 start_index 字段
#       记录该块在原文中的起始位置，便于追溯
#   返回: CharacterTextSplitter 实例
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# 3.分割文本
# split_documents(): 分割文档列表
#   参数:
#     - documents: Document 对象列表
#   返回: List[Document]，分割后的文档块列表
#   作用:
#     - 根据 separator 和 chunk_size 将长文档分割为多个小块
#     - 保留原文档的 metadata，并添加 start_index
chunks = text_splitter.split_documents(documents)

# 遍历所有分块，输出每块的大小和元数据
for chunk in chunks:
    print(f"块大小:{len(chunk.page_content)}, 元数据:{chunk.metadata}")

# 输出总分块数量
print(len(chunks))
