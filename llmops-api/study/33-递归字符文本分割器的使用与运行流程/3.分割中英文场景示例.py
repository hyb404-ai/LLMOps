#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@Time    : 2024/7/2 10:53
@Author  : thezehui@gmail.com
@File    : 3.分割中英文场景示例.py

===================================================================================
知识点讲解：自定义分隔符处理中英文混合文本
===================================================================================

1. 中英文混合文本的挑战
   - 中文使用全角标点（。！？），英文使用半角标点（.!?）且后常跟空格
   - 默认分隔符 ["\n\n", "\n", " ", ""] 按英文习惯设计，对中文句子边界不敏感
   - 想更好切中英文文档，需重设分隔符列表（或继承类重写）

2. 自定义 separators 参数
   - 可覆盖默认分隔符列表，按优先级从高到低排列
   - 配合 is_separator_regex=True 启用正则模式，分隔符被当成正则直接使用

3. 中英文分隔符设计与优先级
   - "\n\n"：段落边界（最高优先级）
   - "\n"：行边界
   - "。|！|？"：中文句末标点，句尾一般可切断
   - r"\.\s|\!\s|\?\s"：英文句末标点+空格（标准英文写法符号后通常加空格）
   - r"；|;\s"：中英文分号
   - r"，|,\s"：中英文逗号（逗号表示语义未完，一般不切，除非块仍超大）
   - " "：空格
   - ""：字符（最低优先级，会把中文切单字、英文切单字母，几乎丢失语义）

4. 各符号优先级含义
   - 逗号/空格优先级低，因为「两个词才有意义」，过早切会丢语义
   - 空字符串是最后兜底，一旦用上语义损失最严重
   - 整体策略：越能代表句子结束的符号越优先切

5. 正则表达式要点
   - is_separator_regex=True：不调用 re.escape，分隔符按正则解释，"|" 表示或
   - 使用原始字符串 r"" 避免 \s 等被 Python 转义
   - \s 匹配空白字符（空格、制表符等），英文标点后常需 \s 才匹配得到

6. 常见错误
   - 想用正则却忘记设 is_separator_regex=True：分隔符会被 re.escape 转义，"。" 等字面量不匹配，模式失效
   - 误把中文标点写成英文半角（反之同理），导致对应语言句子切不动

7. 典型输出与观察点
   - 用中英文分隔符对同一份 Markdown 分割，块大小分布与默认配置接近（251、451、490…）
   - 观察点：中文句末标点处更易形成干净的句子级切块，块内语义更连贯
   - 在 LLMOps 项目中，分隔符/块大小/重叠大小均由创建知识库用户外部传入，再生成分割器

8. 使用场景
   - 技术文档、中英文混排的 API 文档
   - 新闻、博客等需要精确控制句子边界的场景

===================================================================================
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.创建加载器和文本分割器
loader = UnstructuredMarkdownLoader("./项目API文档.md")

# 自定义分隔符列表，适配中英文混合场景
separators = [
    "\n\n",                    # 段落边界（最高优先级）
    "\n",                      # 行边界
    "。|！|？",                # 中文句末标点符号
    r"\.\s|\!\s|\?\s",         # 英文句末标点+空格（r 前缀表示原始字符串）
    r"；|;\s",                 # 中英文分号
    r"，|,\s",                 # 中英文逗号
    " ",                       # 空格（词边界）
    ""                         # 字符（最低优先级，最后手段）
]

# RecursiveCharacterTextSplitter(): 递归字符文本分割器
#   参数:
#     - separators: 自定义分隔符列表
#       按优先级排列，从粗粒度到细粒度
#     - is_separator_regex: 是否将分隔符作为正则表达式处理
#       True 表示启用正则模式，支持 "|" 等正则语法
#     - chunk_size: 块大小限制
#     - chunk_overlap: 块重叠大小
#     - add_start_index: 是否添加起始位置索引
#   返回: RecursiveCharacterTextSplitter 实例
#   作用:
#     - 根据自定义的中英文分隔符智能分割
#     - 优先在句子边界分割，保持语义完整
text_splitter = RecursiveCharacterTextSplitter(
    separators=separators,
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# 2.加载文档与分割
documents = loader.load()

# split_documents(): 执行分割
#   工作流程:
#     1. 优先在段落边界（\n\n）分割
#     2. 然后在句子边界（中英文标点）分割
#     3. 最后在词和字符边界分割
chunks = text_splitter.split_documents(documents)

# 输出每个块的大小和元数据
for chunk in chunks:
    print(f"块大小: {len(chunk.page_content)}, 元数据: {chunk.metadata}")

# 输出第3个块的内容，查看中英文分割效果
print(chunks[2].page_content)
