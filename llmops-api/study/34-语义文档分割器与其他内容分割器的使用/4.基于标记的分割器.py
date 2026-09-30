#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 22:13
@Author  : thezehui@gmail.com
@File    : 4.基于标记的分割器.py

===================================================================================
知识点讲解：基于 Token 的文本分割器
===================================================================================

1. 为什么按 Token 而非字符分割
   - LLM 上下文窗口与 API 计费都以 Token 为单位，而非字符数 len()
   - 在 GPT 模型中，一个汉字约 1.5 个 Token，一个英文单词约 1 个 Token，用 len() 估算误差很大
   - 精确控制输入大小才能避免超窗、准确估算成本

2. tiktoken 工具库
   - OpenAI 官方的 Token 计数库，安装：pip install -U tiktoken
   - 不同模型 tokenizer 不同，但 tiktoken 可大致计算、误差小
   - encoding_for_model()：获取指定模型的编码器

3. calculate_token_count() 长度函数
   - encoding = tiktoken.encoding_for_model(model_name)
   - encoding.encode(text)：将文本编码为 Token ID 列表
   - len(encoding.encode(text)) 即为该文本的 Token 数量
   - 函数签名 (str) -> int，正好匹配分割器的 length_function 参数

4. 在 RecursiveCharacterTextSplitter 中使用
   - 默认 length_function 用 len() 算字符数；传入自定义函数即可改用 Token 数
   - 本例传入 calculate_token_count，配合 separators（中英文标点）、chunk_size=500、chunk_overlap=50
   - 这样每块控制在约 500 Token（重叠 50），与模型实际处理方式对齐

5. 典型输出示例与观察点
   - 用 tiktoken 方式切分「科幻短篇.txt」输出类似：
       块大小: 334, 元数据: {'source': './科幻短篇.txt'}
       块大小: 409, 元数据: {'source': './科幻短篇.txt'}
       块大小: 372, 元数据: {'source': './科幻短篇.txt'}
   - 观察点：块大小相比字符分割更接近真实 Token 消耗，便于成本与窗口管理

6. from_tiktoken_encoder() 快捷方式
   - 除传 length_function，还可直接调用类方法 from_tiktoken_encoder(model_name=..., chunk_size=..., chunk_overlap=..., separators=...) 快速创建基于 tiktoken 的分割器
   - 注意：分词器使用的模型要与开发的 LLM 保持一致，否则 Token 数仍有偏差

7. 使用场景
   - 需要精确控制 Token 数、成本敏感、接近上下文窗口的长文档、多模型（不同 tokenizer）支持

===================================================================================
"""
import tiktoken
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def calculate_token_count(query: str) -> int:
    """
    计算传入文本的token数

    参数:
        query: 要计算 Token 数的文本字符串

    返回:
        int: Token 数量

    作用:
        - 使用 tiktoken 库获取指定模型的编码器
        - 将文本编码为 Token ID 列表
        - 返回列表长度即为 Token 数
    """
    # tiktoken.encoding_for_model(): 获取指定模型的编码器
    #   参数:
    #     - model_name: 模型名称，如 "text-embedding-3-large"
    #   返回: Encoding 对象，包含该模型的 tokenizer
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")

    # encoding.encode(): 将文本编码为 Token ID 列表
    #   参数:
    #     - text: 要编码的文本
    #   返回: List[int]，Token ID 列表
    # len(): 计算列表长度，即 Token 数量
    return len(encoding.encode(query))


# 1.定义加载器和文本分割器
loader = UnstructuredLoader("./科幻短篇.txt")

# RecursiveCharacterTextSplitter(): 递归字符文本分割器
#   参数:
#     - separators: 自定义分隔符列表
#       包含中英文标点符号，适配混合文本
#     - is_separator_regex: 启用正则表达式模式
#     - chunk_size: 每块最大 Token 数（使用 length_function 计算）
#       500 表示每块不超过 500 个 Token
#     - chunk_overlap: 块重叠大小（Token 数）
#       50 表示相邻块有 50 个 Token 重叠
#     - length_function: 自定义长度计算函数
#       传入 calculate_token_count 函数，使用 Token 数而非字符数
#   返回: RecursiveCharacterTextSplitter 实例
#   作用:
#     - 基于 Token 数而非字符数进行分割
#     - 精确控制每块的 Token 数量
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。|！|？",
        r"\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
        r"；|;\s",
        r"，|,\s",
        " ",
        ""
    ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    length_function=calculate_token_count,  # 使用 Token 计数函数
)

# 2.加载文档并执行分割
documents = loader.load()

# split_documents(): 执行分割
#   工作流程:
#     1. 使用 calculate_token_count 计算每个候选块的 Token 数
#     2. 确保每块的 Token 数不超过 chunk_size（500）
#     3. 相邻块有 chunk_overlap（50）个 Token 重叠
chunks = text_splitter.split_documents(documents)

# 3.循环打印分块内容
for chunk in chunks:
    # 注意: len(chunk.page_content) 是字符数，不是 Token 数
    # 实际 Token 数由 calculate_token_count() 控制
    print(f"块大小: {len(chunk.page_content)}, 元数据: {chunk.metadata}")

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("./xxx.txt")
# 1.x 改用独立集成包 langchain-unstructured 的 UnstructuredLoader，
# 默认按元素(element)返回文档，需要整篇合并时传 chunking_strategy="basic"
