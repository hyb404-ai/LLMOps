#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 14:33
@Author  : thezehui@gmail.com
@File    : 1.语义分割器使用示例.py

===================================================================================
知识点讲解：语义文档分割器（SemanticChunker）
===================================================================================

1. 组件定位：语义分割器（SemanticChunker）解决什么问题
   - 传统字符分割器（如 RecursiveCharacterTextSplitter）只按字符数 / 分隔符等表面特征切分，不关心句子间的语义关联
   - SemanticChunker 基于文本语义相似度切分，能智能识别主题边界，让每个块尽量保持语义连贯
   - 适用场景：长文本需要切成「语义相关」的块、问答 / 知识库等对语义完整性要求高的任务

2. 实验性定位与包归属（重要）
   - 该类目前位于 langchain_experimental 包，安装命令：pip install -Uqq langchain_experimental
   - 实验性意味着类与方法未来大概率会发生变更，需谨慎用于生产
   - 与多数分割器不同，SemanticChunker 并没有继承 TextSplitter，因此 API 形态略有差异

3. 工作原理（五步）
   - 第一步：按 sentence_split_regex 把文档拆成独立的句子
   - 第二步：根据 buffer_size（默认 1）前后拼接相邻句子，形成待编码的小段
   - 第三步：用传入的 Embedding 模型对每段计算向量
   - 第四步：按传入的「分块数 + 断点类型」计算一个相似度阈值
   - 第五步：把相似度高于阈值的相邻段合并，低于阈值的作为切块边界

4. 核心参数详解
   - embeddings：文本嵌入模型，底层用向量余弦相似度识别句子相似性（必填）
   - buffer_size：缓冲区大小，默认 1，计算相似度时该句叠加前后各 1 条文本，首尾不足则不叠加
   - add_start_index：是否添加起点索引，默认 False（本例设为 True 便于定位）
   - breakpoint_threshold_type：断点阈值类型，默认 percentile（百分位）
   - breakpoint_threshold_amount：断点阈值的具体得分 / 金额
   - number_of_chunks：期望生成的块数量，默认 None（本例设为 10）
   - sentence_split_regex：句子切割正则，默认 (?<=[.?!])\s+；本例用 r"(?<=[。？！.?!])" 适配中英文句末标点

5. 底层断点检测算法
   - 目前支持 4 种阈值判定方法：百分位（默认）、标准差、四分位数、梯度
   - 不同方法会影响切块粒度，百分位适合大多数分布较均匀的文本

6. 典型输出示例与观察点
   - 按本例参数切 10 块后输出类似：
       块大小: 201, 元数据: {'source': './科幻短篇.txt'}
       块大小: 25,  元数据: {'source': './科幻短篇.txt'}
       块大小: 31,  元数据: {'source': './科幻短篇.txt'}
       块大小: 466, 元数据: {'source': './科幻短篇.txt'}
       块大小: 0,   元数据: {'source': './科幻短篇.txt'}
   - 观察点 1：各块大小差异很大，因为切块由语义断点而非固定字符数决定
   - 观察点 2：可能出现大小为 0 的空块（语义断点恰好落在句边界），生产可过滤

7. 同类语义分割器补充
   - 除基于 Embedding 的 SemanticChunker，LangChain 还提供 3 种基于 NLP 的语义分割器：
     * NLTKTextSplitter：基于 NLTK 自然语言处理库
     * SpacyTextSplitter：基于 spaCy 高级 NLP 库
     * SentenceTransformersTokenTextSplitter：按句子转换器模型的 token 窗口切分

8. 优势、局限与注意事项
   - 优势：保持主题完整性、适合高语义连贯性场景
   - 局限：要调用 Embedding API 成本高、速度慢、依赖 Embedding 模型质量、大文档产生大量 API 调用
   - 注意事项：需要网络调用 Embedding API；建议配合 langchain_experimental 使用；生产环境注意空块过滤与 API 限流

===================================================================================
"""
import dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 加载环境变量（包含 API 密钥等配置）
dotenv.load_dotenv()

# 1.构建加载器和文本分割器
loader = UnstructuredLoader("./科幻短篇.txt")

# SemanticChunker(): 语义文档分割器
#   参数:
#     - embeddings: Embedding 模型实例
#       OpenAIEmbeddings(model="text-embedding-3-small") 使用 OpenAI 的小型嵌入模型
#       用于计算句子的向量表示
#     - number_of_chunks: 期望生成的块数量（可选）
#       10 表示目标生成约 10 个语义块
#     - add_start_index: 是否在 metadata 中添加起始位置索引
#     - sentence_split_regex: 句子分割正则表达式
#       r"(?<=[。？！.?!])" 表示在中英文句末标点后分割
#       (?<=...) 是正向后瞻断言，不消耗字符
#   返回: SemanticChunker 实例
#   作用:
#     - 先将文本按句子分割
#     - 计算每个句子的 Embedding 向量
#     - 分析相邻句子的语义相似度
#     - 在语义差异较大的边界处分割
text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    number_of_chunks=10,
    add_start_index=True,
    sentence_split_regex=r"(?<=[。？！.?!])"
)

# 2.加载文本与分割
documents = loader.load()

# split_documents(): 执行语义分割
#   返回: List[Document]
#   工作流程:
#     1. 将文档按 sentence_split_regex 分割为句子
#     2. 对每个句子调用 embeddings.embed_query() 获取向量
#     3. 计算相邻句子的余弦相似度
#     4. 在相似度低的位置进行分割
#     5. 确保生成约 number_of_chunks 个块
chunks = text_splitter.split_documents(documents)

# 3.循环打印
for chunk in chunks:
    print(f"块大小: {len(chunk.page_content)}, 元数据: {chunk.metadata}")

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
