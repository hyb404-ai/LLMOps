#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 22:38
@Author  : thezehui@gmail.com
@File    : 1.自定义分割器示例.py

===================================================================================
知识点讲解：自定义文档分割器的实现
===================================================================================

1. 为什么要自定义文档分割器
   - 内置分割器（CharacterTextSplitter、RecursiveCharacterTextSplitter、
     HTMLHeaderTextSplitter、SemanticChunker 等）覆盖大多数场景，
     但极少数特殊需求仍需自定义（官方文档也注明这种情况「一般极少」）
   - 典型动机：实现内置器无法完成的分割逻辑（如按业务规则切分）、
     在分割的同时做关键词提取 / 摘要生成、结合 jieba 等 NLP 技术、
     针对特定领域文档定制

2. 自定义分割器的实现方式（继承 TextSplitter）
   - 在 LangChain 中实现自定义文档分割器的方法非常简单，三步即可：
     * 继承文本分割器基类 TextSplitter
     * 在构造函数中传递相关参数（如分隔符、关键词数量），并调用 super().__init__(**kwargs)
     * 实现 split_text() 方法
   - 其余能力（如 split_documents）由父类提供默认实现，无需自己写

3. TextSplitter 基类与 split_text() 方法
   - TextSplitter 是所有文本分割器的抽象基类
   - split_text(text: str) -> List[str] 是抽象方法，必须在子类中实现，负责核心分割逻辑
   - split_documents(documents) 有默认实现，内部遍历每个 Document 并调用 split_text()，
     自动继承原文档的 metadata，子类无需重写

4. 实现示例：分割 + 关键词提取二合一
   - 需求：按传入的分隔符把文档划分成片段，并把每个片段转换成 N 个关键词
   - 代码：
       class CustomTextSplitter(TextSplitter):
           def __init__(self, seperator: str, top_k: int = 10, **kwargs) -> None:
               super().__init__(**kwargs)
               self._seperator = seperator
               self._top_k = top_k

           def split_text(self, text: str) -> List[str]:
               split_texts = text.split(self._seperator)
               text_keywords = []
               for split_text in split_texts:
                   text_keywords.append(
                       jieba.analyse.extract_tags(split_text, self._top_k)
                   )
               return [",".join(keywords) for keywords in text_keywords]
   - 调用：
       loader = UnstructuredFileLoader("./科幻短篇.txt")
       text_splitter = CustomTextSplitter("\n\n")
       documents = loader.load()
       chunks = text_splitter.split_documents(documents)
   - 关键点：必须调用 super().__init__(**kwargs)，否则父类的 chunk_size /
     chunk_overlap / length_function 等通用参数无法生效

5. jieba.analyse 关键词提取
   - jieba.analyse 是中文分词与关键词提取库，extract_tags() 可提取文本关键词
   - 基于 TF-IDF 或 TextRank 算法，按权重降序返回 topK 个关键词，适合中文文本处理
   - 本示例用它把每段文本压缩为关键词列表，便于后续的轻量索引与匹配

6. 本示例的创新点
   - 不是简单「切文本」，而是在切分的同时对每段提取关键词
   - 把每段文本转换为逗号分隔的关键词字符串，适合构建基于关键词的索引与检索系统
   - 输出可直接作为 page_content 入库，检索时按关键词命中而非全文语义

7. 构造函数参数设计
   - seperator：分隔符，控制如何按原文切分（例如用段落之间的空行来切分）
   - top_k：每段提取的关键词数量，默认 10
   - **kwargs：透传给父类 TextSplitter 的其他参数（如 chunk_size / chunk_overlap）

8. 典型输出示例与观察点
   - 用 CustomTextSplitter("\n\n") 处理一篇科幻短篇，每段被压成关键词串：
       文档库块内容:星际,穿越,标题
       文档库块内容:概要
       文档库块内容:星系,李昂,文明,这个,高度发达,宇航员,星际,灭绝,一个,穿越
       文档库块内容:迷航,星际,第一章
       文档库块内容:飞船,李昂,观察窗,浩瀚,星空,宇航员,经验丰富,星系,与众不同,穿越
       ...
       文档库块内容:慕课
       文档库块内容:梦工厂,程序员
   - 观察点 1：输出不再是原文片段，而是关键词，说明 split_text() 的返回值拥有
     完全的自由——它只要求 List[str]，不要求保留原文
   - 观察点 2：章节标题这类极短段落（如「概要」「第五章,危机,希望」）提取出的
     关键词很少甚至就是标题本身，短文本是关键词提取的弱项
   - 观察点 3：这种「关键词化」的分块会丢失原文语序，不适合需要原文作答的 RAG，
     更适合标签化检索与聚类

9. RAG 的 4 种分块（Chunk）策略
   - 固定大小分块：最常见的分块方法，通过设定块的大小和是否有重叠来决定分块；
     简单直接、不需要任何 NLP 库，计算成本低且易于使用，
     例如 CharacterTextSplitter，或直接循环遍历固定大小拆分
   - 基于结构的分块：常见的 HTML、Markdown 格式，或其他有明确结构的文档；
     借助「结构感知」充分利用文本以外的信息，如 HTMLHeaderTextSplitter
   - 基于语义的分块：确保每个分块包含尽可能多的语义独立信息，
     可用标点符号、自然段落、NLTK / spaCy 等工具，或 Embedding-based 方法，
     如 SemanticChunker
   - 递归分块：使用一组分隔符，以分层和迭代的方式把输入文本划分成更小的块；
     若首次分割未得到所需大小的块，则继续递归分割直到满足条件，
     如 RecursiveCharacterTextSplitter
   - 这 4 种策略各有优势与适用场景，到目前为止还没有公认的最优策略，
     这也是很难有一个产品一统天下的原因；策略可以组合使用，
     并非一类文档只能用一种策略

10. 分块在 RAG 流程中的定位
    - 一个 RAG 场景可拆为四个主要阶段：预检索、检索、后检索、生成
    - 分块（Chunk）属于「预检索」阶段的策略
    - 重要判断：如果在分块阶段尝试上述 4 种策略均没有很好的效果，
      或许就不应该继续采用 RAG，而是改用微调，让这部分知识成为模型的永久记忆，
      效果可能会更好

11. 使用场景与最佳实践
    - 本自定义分割器适用于：构建关键词索引、文档标签生成、快速关键词预览、
      结合 NLP 的高级文档处理
    - 自定义 split_text 要保持纯函数风格（相同输入相同输出），避免引入网络 / IO 依赖
    - 生产环境注意 jieba 首次加载词典有开销，建议在进程启动时预热
    - 若只需常规分块，优先用内置 RecursiveCharacterTextSplitter，
      自定义是「极少数」场景的兜底

===================================================================================

===================================================================================
"""
from typing import List

import jieba.analyse
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TextSplitter


class CustomTextSplitter(TextSplitter):
    """自定义文本分割器"""

    def __init__(self, seperator: str, top_k: int = 10, **kwargs):
        """
        构造函数，传递分割器还有需要提取的关键词数，默认为10

        参数:
            seperator: 分隔符，用于分割原文（如 "\n\n" 按段落分）
            top_k: 每段文本提取的关键词数量，默认 10
            **kwargs: 其他参数，传递给父类 TextSplitter
        """
        # 调用父类构造函数
        super().__init__(**kwargs)
        self._seperator = seperator
        self._top_k = top_k

    def split_text(self, text: str) -> List[str]:
        """
        传递对应的文本执行分割并提取分割数据的关键词，组成文档列表返回

        参数:
            text: 要分割和处理的文本字符串

        返回:
            List[str]: 关键词字符串列表，每个元素是一段的关键词（逗号分隔）

        作用:
            - 先按分隔符分割文本
            - 对每段提取关键词
            - 将关键词列表转换为逗号分隔的字符串
        """
        # 1.根据传递的分隔符分割传入的文本
        # split(): 字符串的内置方法，按指定分隔符分割
        split_texts = text.split(self._seperator)

        # 2.提取分割出来的每一段文本的关键词，数量为self._top_k个
        text_keywords = []
        for split_text in split_texts:
            # jieba.analyse.extract_tags(): 提取关键词
            #   参数:
            #     - sentence: 要分析的文本
            #     - topK: 返回前 K 个关键词
            #   返回: List[str]，关键词列表（按权重降序）
            #   作用: 使用 TF-IDF 算法提取文本的核心关键词
            text_keywords.append(jieba.analyse.extract_tags(split_text, self._top_k))

        # 3.将关键词使用逗号进行拼接组成字符串列表并返回
        # ",".join(): 将列表元素用逗号连接成字符串
        # 列表推导式: 对每个关键词列表执行 join 操作
        return [",".join(keywords) for keywords in text_keywords]


# 1.创建加载器与分割器
loader = UnstructuredLoader("./科幻短篇.txt")

# CustomTextSplitter(): 创建自定义分割器实例
#   参数:
#     - seperator: 分隔符，"\n\n" 表示按段落分割
#     - top_k: 每段提取 10 个关键词
#   返回: CustomTextSplitter 实例
text_splitter = CustomTextSplitter("\n\n", 10)

# 2.加载文档并分割
documents = loader.load()

# split_documents(): 分割文档（继承自 TextSplitter）
#   内部调用 split_text() 方法
#   返回: List[Document]
#   每个 Document 的 page_content 是关键词字符串（逗号分隔）
chunks = text_splitter.split_documents(documents)

# 3.循环遍历文档信息
for chunk in chunks:
    # 输出每段的关键词（不是原文，而是提取的关键词）
    print(chunk.page_content)

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
