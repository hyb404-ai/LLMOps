#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 14:41
@Author  : thezehui@gmail.com
@File    : 9.语义路由选择不同的Prompt模板.py

===================================================================================
知识点讲解：基于语义相似度的动态 Prompt 路由
===================================================================================

1. 语义路由要解决的痛点
   - 针对不同场景的问题，使用"特定化 prompt 模板"效果普遍好于通用模板
   - 以教培场景为例：同时教物理 + 数学的授课机器人，若写通用模板会臃肿难维护，
     若写简单又覆盖不住学科——两难
   - 解法：按提问学科动态挑选专用模板（数学问题走数学模板，物理问题走物理模板）

2. 核心思想：把向量相似性用于"模板路由"
   - 基于向量与文本嵌入模型可推断：数学提问（如"能介绍下余弦计算公式么？"）在向量空间
     中与"数学模板"更近；物理提问（如"黑洞是什么?"）与"物理模板"更近
   - 因此向量相似性搜索不仅用于向量数据库，也可直接用于"原始问题 <-> prompt 模板"比对，
     实现 prompt 模板的动态路由

3. 技术原理与运行流程
   - 把每个 prompt 模板文本用 embed_documents 转成嵌入向量（启动时算一次，常驻内存）
   - 把用户 query 用 embed_query 转成向量
   - 用 cosine_similarity 计算 query 与各模板向量的相似度，取 argmax 得到最相似模板
   - 用 ChatPromptTemplate.from_template 把选中的模板字符串包装成可运行模板，接入 LCEL

4. 与逻辑路由的成本对比
   - 逻辑路由（按规则 / LLM 判定走哪个检索器）：每次路由消耗 1 次 LLM 调用，贵且慢，
     但能表达复杂条件（"若 A 且 B 则走 C"）、可解释性强
   - 语义路由（本示例）：每次路由只消耗 1 次 embedding 调用，便宜且快，
     模板向量常驻内存，仅对 query 做一次 embed_query，开销极小
   - 代价：只能回答"哪个更像"，无法表达复杂条件逻辑

5. 在 RAG 流程中的定位
   - 这类动态路由属于 RAG 多阶段里的"路由"阶段（与逻辑路由同族，但路由对象不同：
     逻辑路由选"检索器 / 数据源"，本示例选"Prompt 模板"）

6. 典型输出示例与观察点
   - 查询 "黑洞是什么?" → 打印"使用物理模板"，返回黑洞成因、事件视界等物理讲解
   - 查询 "能介绍下余弦计算公式么？" → 打印"使用数学模板"，返回余弦定义、公式
     cos(θ)=邻边/斜边，并体现"分解成小步骤再整合"的风格
   - 观察点：数学模板输出明显带有模板里写入的"分解小步骤"要求，
     这正是 math_template 生效的证据，可据此验证路由是否命中预期模板

7. 应用场景
   - 多领域问答系统（物理 / 数学 / 历史等）
   - 多角色助手（教师 / 顾问 / 技术支持）
   - 多风格响应（正式 / 轻松 / 技术性）

8. 最佳实践
   - 为每个领域设计专门的 prompt 模板
   - 使用高质量嵌入模型（如 text-embedding-3-small）
   - 可加阈值判断：相似度过低时回退到默认模板，避免"强行选一个不像的"
   - 记录路由决策，用于持续优化模板设计
   - 模板文本要"有领域辨识度"：领域术语越明确，与该领域提问的余弦相似度越高，路由越准

9. 版本对照说明（示例基于 LangChain 1.x）
   - 0.x：from langchain.utils.math import cosine_similarity
   - 1.x：from langchain_classic.utils.math import cosine_similarity
     （LangChain 1.x 把原 langchain 包中的历史工具迁移到 langchain_classic 命名空间）
   - 其余 LCEL 写法两版一致

===================================================================================
"""
import dotenv
from langchain_classic.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 加载环境变量配置
dotenv.load_dotenv()

# 1.定义两份不同的prompt模板(物理模板、数学模板)
# 每个模板都针对特定领域设计，包含该领域的专业要求
physics_template = """你是一位非常聪明的物理教程。
你擅长以简洁易懂的方式回答物理问题。
当你不知道问题的答案时，你会坦率承认自己不知道。

这是一个问题：
{query}"""

math_template = """你是一位非常优秀的数学家。你擅长回答数学问题。
你之所以如此优秀，是因为你能将复杂的问题分解成多个小步骤。
并且回答这些小步骤，然后将它们整合在一起回来更广泛的问题。

这是一个问题：
{query}"""

# 2.创建文本嵌入模型，并执行嵌入
# OpenAIEmbeddings：创建 OpenAI 的文本嵌入模型
# - model：指定嵌入模型，text-embedding-3-small 性能好且成本低
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 将所有 Prompt 模板存储在列表中
prompt_templates = [physics_template, math_template]

# embed_documents：批量将模板文本转换为嵌入向量
# 返回的是一个嵌入向量列表，每个向量代表一个模板的语义特征
prompt_embeddings = embeddings.embed_documents(prompt_templates)


def prompt_router(input) -> ChatPromptTemplate:
    """
    Prompt 路由函数：根据传递的 query 计算返回不同的提示模板

    Args:
        input: 字典，包含 "query" 键，值为用户的问题

    Returns:
        ChatPromptTemplate: 与问题最相似的 Prompt 模板对象

    工作流程：
        1. 将用户问题转换为嵌入向量
        2. 计算问题向量与各模板向量的余弦相似度
        3. 选择相似度最高的模板
        4. 返回对应的 ChatPromptTemplate 对象
    """
    # 1.计算传入query的嵌入向量
    # embed_query：将单个文本转换为嵌入向量
    query_embedding = embeddings.embed_query(input["query"])

    # 2.计算相似性
    # cosine_similarity：计算两个向量集合之间的余弦相似度
    # 参数：([query_embedding], prompt_embeddings)
    # 返回：二维数组，[0] 取第一行，得到 query 与各模板的相似度列表
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]

    # argmax()：返回相似度最高的索引
    most_similar = prompt_templates[similarity.argmax()]

    # 输出路由决策，便于调试和监控
    print("使用数学模板" if most_similar == math_template else "使用物理模板")

    # 3.构建提示模板
    # from_template：将字符串模板转换为 ChatPromptTemplate 对象
    return ChatPromptTemplate.from_template(most_similar)


# 构建完整的处理链（使用 LCEL 表达式）
# {"query": RunnablePassthrough()}：将输入包装成字典
# | RunnableLambda(prompt_router)：调用路由函数选择模板
# | ChatOpenAI：使用选定的模板调用大语言模型
# | StrOutputParser()：将模型输出解析为字符串
chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | ChatOpenAI(model="deepseek-v4-pro")
        | StrOutputParser()
)

# 测试物理问题路由
# "黑洞"是物理学概念，应该路由到 physics_template
print(chain.invoke("黑洞是什么?"))
print("======================")

# 测试数学问题路由
# "余弦计算公式"是数学概念，应该路由到 math_template
print(chain.invoke("能介绍下余弦计算公式么？"))
