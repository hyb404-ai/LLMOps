#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 11:45
@Author  : thezehui@gmail.com
@File    : 8.基于逻辑和语义的路由分发.py

===================================================================================
知识点讲解：检索器的逻辑路由实现（路由分发）

1. 设计动机：多知识库场景的痛点
   - 当系统挂了多个知识库/集合（如 python_docs、js_docs、golang_docs）时，若每次提问都去检索全部向量库，会带来三个问题：
     * 成本高：N 个库就是 N 次向量检索
     * 噪声大：无关语言的文档混进上下文，挤占 Prompt 里靠前的高权重位置
     * 准确性低：跨库的相近表述互相干扰
   - 逻辑路由先判定「该去哪个库」，把检索范围提前缩减到一个库，同时降本、降噪、提准

2. 逻辑路由的整体方案
   - 设定对应的 Prompt，让 LLM 根据问题返回需要选择的检索器/向量数据库名称，再根据名称选择不同的检索器
   - 但普通 prompt 很难保证输出格式稳定，因此改用函数回调：设定一个虚假函数，让 LLM 强制调用并输出参数，从而保证输出统一
   - 本示例用 Prompt 工程 + 结构化输出 + LCEL 链式调用，把上述过程串成一条自动路由链

3. 在 RAG 6 阶段中的定位
   - RAG 优化划分为 6 个阶段：查询转换、路由、查询构建、索引、检索、生成
   - 本示例属于第 2 个阶段「路由」，对应的优化策略叫「动态路由」

4. 两类路由的区别
   - 逻辑路由（本示例）：用 LLM + 函数回调判定分类，靠「语义理解 + 结构化输出」；优点是可写复杂判定规则、能解释，缺点是每次都要一次 LLM 调用（有成本和延迟）
   - 语义路由：用 embedding 余弦相似度直接比对，不经过 LLM；优点是快且便宜，缺点是只能表达「哪个更像」，无法表达复杂条件

5. 为什么不用纯 Prompt 判定
   - LLM 的输出天生带随机性与「热情解释」，纯 prompt 很难保证格式稳定
   - 改用函数回调后，LLM 输出的是函数调用参数，格式由 schema 强约束，程序可直接消费

6. 路由链构建：LCEL 表达式与 choose_route
   - 核心链路：{"question": RunnablePassthrough()} | prompt | structured_llm | choose_route
     * RunnablePassthrough 将原始输入包装成 {"question": ...} 字典
     * prompt 生成完整消息列表，structured_llm 调用模型并解析为 RouteQuery 对象
     * choose_route 读取 RouteQuery.datasource，返回对应的检索器（本示例简化为返回标识符字符串）
   - choose_route 按 datasource 命中分支，生产环境应返回实际检索器实例而非字符串

7. 典型输出示例
   - 结构化中间结果：datasource='python_docs'
   - 路由函数返回值：chain for python_docs
   - 本示例只打印最终路由结果（即 choose_route 的返回值）

8. 应用场景
   - 多数据源检索：根据问题类型选择不同的知识库
   - 多模型调度：根据任务复杂度选择不同的模型
   - 多策略执行：根据问题特征选择不同的处理策略

9. 最佳实践
   - 在 System Prompt 中明确路由规则
   - 使用确定性输出（temperature=0）保证路由稳定性
   - 设计清晰的路由逻辑函数，可根据路由结果加载不同的检索器或处理链

10. 版本对照：原课程文档基于 LangChain 0.x，本示例为 1.x
    - 0.x：from langchain_core.pydantic_v1 import BaseModel, Field；路由链写法与本示例一致
    - 1.x：from pydantic import BaseModel, Field
    - LCEL 表达式 {"question": RunnablePassthrough()} | prompt | structured_llm | choose_route 在两个版本中写法完全一致

11. 延伸：LCEL 路由的能力边界
    - 本示例是「一次判定、单向流动」的线性路由，用 LCEL 表达式非常合适
    - 但当路由需要「判定 -> 执行 -> 评估 -> 不满意则回退重试」时，LCEL 力不从心，因为 LCEL 链只能从前往后流动、无法回退与循环
    - CRAG（纠正性 RAG）与 Self-RAG 正是这类带条件分支与循环的架构，需要改用 LangGraph 以循环图（有向有环图）作为骨架实现

===================================================================================
"""
from typing import Literal

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 加载环境变量配置
dotenv.load_dotenv()


class RouteQuery(BaseModel):
    """
    路由查询模型：将用户查询映射到最相关的数据源

    通过分析用户问题中的关键信息（如编程语言、技术栈等），
    决定应该从哪个专门的数据源检索答案。
    """
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        description="根据给定用户问题，选择哪个数据源最相关以回答他们的问题"
    )


def choose_route(result: RouteQuery) -> str:
    """
    路由选择函数：根据传递的路由结果选择不同的检索器

    Args:
        result: RouteQuery 对象，包含路由决策结果

    Returns:
        str: 对应数据源的标识符，实际应用中可以返回检索器对象

    说明：
        在生产环境中，这个函数应该返回实际的检索器实例，
        而不是字符串。这里为了演示简化，直接返回标识符。
    """
    if "python_docs" in result.datasource:
        # 实际应用中应该返回：return python_retriever
        return "chain in python_docs"
    elif "js_docs" in result.datasource:
        # 实际应用中应该返回：return js_retriever
        return "chain in js_docs"
    else:
        # 实际应用中应该返回：return golang_retriever
        return "golang_docs"


# 1.构建大语言模型并进行结构化输出
# ChatOpenAI：创建 OpenAI 兼容的聊天模型
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# with_structured_output：绑定输出结构到 RouteQuery 模型
# 确保模型输出符合预定义的数据源选项
structured_llm = llm.with_structured_output(RouteQuery)

# 2.创建路由逻辑链
# ChatPromptTemplate.from_messages：创建多消息提示模板
# - system 消息：定义助手的角色和路由规则
# - human 消息：用户的实际问题
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个擅长将用户问题路由到适当的数据源的专家。\n请根据问题涉及的编程语言，将其路由到相关数据源"),
    ("human", "{question}")
])

# 构建完整的路由链（使用 LCEL 表达式）
# {"question": RunnablePassthrough()}：将输入包装成字典，键为 "question"
# | prompt：将字典传入提示模板，生成完整的消息列表
# | structured_llm：调用模型并解析为 RouteQuery 对象
# | choose_route：根据路由结果选择具体的检索器
router = {"question": RunnablePassthrough()} | prompt | structured_llm | choose_route

# 3.执行相应的提问，检查映射的路由
# 这个问题包含 Python 代码（langchain_core），应该路由到 python_docs
question = """为什么下面的代码不工作了，请帮我检查下：

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("中文")"""

# 4.选择不同的数据库
# invoke：执行完整的路由链
# 输出应该是 "chain in python_docs"
print(router.invoke(question))
