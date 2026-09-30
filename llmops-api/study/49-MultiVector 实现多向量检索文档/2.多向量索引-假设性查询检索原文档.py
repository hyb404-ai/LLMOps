#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/6 11:09
@Author  : thezehui@gmail.com
@File    : 2.多向量索引-假设性查询检索原文档.py

===================================================================================
知识点讲解：假设性查询（HyDE）检索策略
===================================================================================

1. 设计动机：缩小语义鸿沟
   - 常规检索是「用户问题」直接匹配「文档描述」，二者表述差异大，存在语义鸿沟
   - 假设性查询（Hypothetical Document Embeddings, HyDE）让 LLM 为每个文档块
     生成它可能回答的假设性问题，用「问题」去检索
   - 问题-问题匹配比文档-问题匹配更贴近用户真实意图，提升召回率与准确率

2. 在多向量索引中的定位
   - 它是多向量/多表征索引的第三种实现路径，与前一示例（摘要）共用同一套
     MultiVectorRetriever 检索骨架，差别只在往向量库里塞的是「假设性问题」
   - 检索命中假设问题后，仍按 doc_id 换回原始文档，保证回答上下文完整

3. 核心概念
   - HyDE 策略：为文档生成假设性问题，用问题检索文档
   - 问题-文档对齐：查询与假设问题在语义空间中更接近
   - 多向量索引：一个文档对应多个检索向量
   - 结构化输出：用 Pydantic 约束假设问题列表的格式

4. 工作原理
   - 对每个文档生成若干假设性问题（如「这个文档可以回答什么问题？」）
   - 将这些假设问题转为向量并索引
   - 用户提问时，直接用问题匹配假设问题
   - 返回匹配到的假设问题对应的原始文档

5. 本示例实现要点
   - HypotheticalQuestions(BaseModel)：用 Pydantic 定义输出结构，确保返回字符串列表
   - Prompt 要求模型为给定文档生成 3 个可能的用户问题
   - llm.with_structured_output(HypotheticalQuestions) 内部使用 Function Calling 保证结构正确
   - LCEL 链：{"doc": lambda x: x.page_content} | prompt | structured_llm
   - temperature=0 保证问题生成的稳定性

6. 完整落地流程（批量版）
   - 对文档库每个文档执行 chain.batch([docs])
   - 将生成的假设问题转嵌入向量
   - 存入向量数据库，metadata 记录原文档 id（doc_id）
   - 用户查询时直接匹配假设问题，再按 doc_id 返回原始文档

7. 课程输出示例
   - 对「我叫慕小课，我喜欢打篮球，游泳」生成 3 个假设性问题，例如：
     * 「如果你不能打篮球，你会选择什么运动？」
     * 「如果你不能游泳，你会选择什么运动？」
     * 「如果你不能进行任何体育运动，你会选择什么爱好？」
   - 体现「让模型站在用户视角反推可能的问题」这一 HyDE 思想

8. 优势与适用场景
   - 问题-问题匹配更精确、缩小语义鸿沟、提高召回率与准确率
   - 适合 FAQ 系统、技术文档检索、客服知识库、问答系统

9. 最佳实践与注意事项
   - 生成的假设问题应多样化；数量建议 3-5 个；temperature=0 确保稳定
   - 可结合真实用户问题不断迭代生成策略
   - 同样有构建期「每个块多一次 LLM 调用」的成本，属多向量索引的通病，适合离线构建

===================================================================================
"""
from typing import List

import dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 加载环境变量配置
dotenv.load_dotenv()


class HypotheticalQuestions(BaseModel):
    """
    假设性问题模型：为文档生成假设性问题

    该模型定义了大语言模型输出的结构，确保返回的是一个字符串列表，
    而不是自由格式的文本。
    """
    questions: List[str] = Field(
        description="假设性问题列表，类型为字符串列表",
    )


# 1.构建一个生成假设性问题的prompt
# 这个 Prompt 要求模型为给定文档生成 3 个可能的用户问题
# 这些问题应该是用户在想要获取文档中的信息时可能会问的
prompt = ChatPromptTemplate.from_template("生成一个包含3个假设性问题的列表，这些问题可以用于回答下面的文档:\n\n{doc}")

# 2.创建大语言模型，并绑定对应的规范化输出结构
# ChatOpenAI：创建聊天模型实例
# - model：指定使用的模型
# - temperature=0：使用确定性输出，保证生成问题的稳定性
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# with_structured_output：将模型输出绑定到 Pydantic 模型
# 这确保模型返回的是符合 HypotheticalQuestions 结构的对象
# 内部使用函数调用（Function Calling）实现
structured_llm = llm.with_structured_output(HypotheticalQuestions)

# 3.创建链应用
# 使用 LCEL 表达式构建处理链
chain = (
        # {"doc": lambda x: x.page_content}：从 Document 对象提取文本内容
        {"doc": lambda x: x.page_content}
        # | prompt：将文档内容填充到提示模板
        | prompt
        # | structured_llm：调用模型生成结构化的假设问题列表
        | structured_llm
)

# 执行链，为测试文档生成假设性问题
# 示例文档：关于一个名叫"慕小课"的人的兴趣爱好
# 预期生成的问题可能包括：
# - "慕小课的名字是什么？"
# - "慕小课喜欢什么运动？"
# - "慕小课的兴趣爱好有哪些？"
hypothetical_questions: HypotheticalQuestions = chain.invoke(
    Document(page_content="我叫慕小课，我喜欢打篮球，游泳")
)

# 输出生成的假设性问题列表
print(hypothetical_questions)

# 实际应用中的完整流程：
# 1. 对文档库中的每个文档执行 chain.batch([docs])
# 2. 将生成的假设问题转换为嵌入向量
# 3. 存储到向量数据库，metadata 中记录原文档的 ID
# 4. 用户查询时，直接用查询匹配假设问题
# 5. 根据匹配到的假设问题的 doc_id 返回原始文档
#
# 示例：
# for doc in docs:
#     questions = chain.invoke(doc)
#     for q in questions.questions:
#         question_doc = Document(
#             page_content=q,
#             metadata={"doc_id": doc.id}
#         )
#         # 将 question_doc 添加到向量数据库
#         # 使用 MultiVectorRetriever 关联问题和原文档
