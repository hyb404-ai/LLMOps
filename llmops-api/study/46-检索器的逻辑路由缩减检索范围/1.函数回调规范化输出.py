#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 13:08
@Author  : thezehui@gmail.com
@File    : 1.函数回调规范化输出.py

===================================================================================
知识点讲解：结构化输出与函数回调（路由基础）

1. 设计动机：为什么需要规范化输出
   - 在 Agent / 路由 / 查询构建等场景，程序需要 LLM 给出「可被直接消费」的结果（如一个数据源名称），而不是一大段自然语言解释
   - 大语言模型越热情，返回的数据越难处理；纯文本输出既不可靠也难解析，因此需要把输出「规范化」成固定结构

2. 函数回调（Function Calling）是什么
   - 给 LLM 一堆工具/函数的描述（函数名、参数描述、函数作用），让 LLM 自行识别在当前提问下最适合调用哪个函数（选一个、多个、不调用、或强制调用某个），然后把「要调用的函数的参数」作为 LLM 的输出内容
   - 拥有 function call 意味着两件事：LLM 可以智能选择不同工具，以及 LLM 可以规范化输出

3. 函数回调的完整运行过程（需要 LLM 与本地程序两方参与）
   - 本地程序把函数清单（名称 / 参数 schema / 用途描述）一起传给 LLM
   - LLM 只负责「选函数 + 生成调用参数」，它并不会真正执行本地函数
   - 本地程序解析 LLM 输出的参数，真正去调用本地函数
   - （可选）把函数执行结果再回传给 LLM，让它组织最终自然语言回复

4. 假函数技巧：用 with_structured_output 实现规范化输出
   - 既然可以让 LLM 强制调用某个函数并输出参数，那么「构造一个虚假函数并强制 LLM 调用它，让 LLM 返回该函数的参数信息」即可实现规范化输出
   - 把需要规范化的数据写成函数参数并配上解释，LLM 输出的参数天然就是结构化数据
   - 在 LangChain 中只需调用模型的 .with_structured_output() 并传入一个 BaseModel 子类，底层会自动把该 BaseModel 转换成函数回调并强制 LLM 调用

5. 反例：为什么不用纯 Prompt 约束输出
   - 若只写一段 prompt 要求 LLM「在 python_docs / js_docs / golang_docs 中选一个」，LLM 会返回一大段带语义场景的解释文本
   - 人类能从中读出答案，但程序要的只是 "python_docs" 这个字符串
   - 结论：用函数回调约束输出，比 Prompt 约束「可靠性更高，而且性能更佳」

6. 核心组件：Pydantic BaseModel / Literal / Field
   - BaseModel：定义数据结构和验证规则
   - Literal 类型：限定字段只能是指定的几个值之一（如数据源枚举），避免非法输出
   - Field 的 description：作为函数回调中「参数描述」的来源，写得越清晰 LLM 选得越准

7. with_structured_output 的底层原理
   - 把 RouteQuery 这个 BaseModel 自动转换成「函数回调」的参数 schema：字段名 -> 参数名，Field.description -> 参数描述，Literal -> enum 枚举约束
   - 告知 LLM 强制调用这个「虚假函数」
   - LLM 不会真的执行任何函数，它只输出这个函数的调用参数
   - LangChain 解析这段参数并实例化成 RouteQuery 对象返回
   - 因此 RouteQuery 本质上是「一个只为规范化输出而存在的假函数签名」

8. 在 RAG 6 阶段中的定位
   - RAG 优化可拆成 6 个阶段：查询转换、路由、查询构建、索引、检索、生成
   - 本示例的结构化输出是「路由」阶段（动态路由）与「查询构建」阶段（自检索）的共同技术底座，两者都依赖 LLM 输出可被程序直接消费的结构化结果

9. 典型输出示例（数据源名称随问题变化）
   - 得到的是一个 RouteQuery 对象（不是字符串），其字段为选定的数据源：
       datasource='python_docs'
   - 调用 res.datasource 即可拿到程序可直接使用的字符串

10. 最佳实践
    - 使用 temperature=0 确保输出的确定性，适合分类 / 路由任务
    - 在 Field 的 description 中提供清晰的字段说明
    - 使用 Literal 类型限定可选值，避免非法输出

11. 版本对照：原课程文档基于 LangChain 0.x，本示例为 1.x
    - 0.x：from langchain_core.pydantic_v1 import BaseModel, Field
    - 1.x：from pydantic import BaseModel, Field（LangChain 已移除 pydantic_v1 兼容层，直接用 pydantic v2）
    - with_structured_output 在两个版本语义一致，底层都是把 BaseModel 转成函数回调并强制 LLM 调用

12. 延伸：函数回调的更广泛应用
    - 函数回调不仅用于规范化输出，它还是「把 LLM 接到企业自有 API」的通用做法：让 LLM 按既定 schema 生成参数，本地程序解析参数后调用现成的、本身不具备智能的接口（例如 PPT 生成 API），即可把一个普通接口快速「智能化」
    - 更完整的工具调用 / Agent / LangGraph 内容见后续章节

===================================================================================
"""
from typing import Literal

import dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 加载环境变量配置（如 OPENAI_API_KEY）
dotenv.load_dotenv()


class RouteQuery(BaseModel):
    """
    路由查询模型：将用户查询映射到对应的数据源上

    该模型用于分类用户的问题，决定应该从哪个数据源检索信息。
    通过限定 datasource 字段的可选值，确保路由结果的合法性。
    """
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        description="根据用户的问题，选择哪个数据源最相关以回答用户的问题"
    )


# 1.创建绑定结构化输出的大语言模型
# ChatOpenAI：创建 OpenAI 聊天模型实例
# - model：指定使用的模型名称
# - temperature=0：设置为 0 使输出更加确定性，适合分类任务
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# with_structured_output：将模型输出绑定到 Pydantic 模型
# 这个方法会自动将大语言模型的 JSON 输出解析为 RouteQuery 对象
# 内部使用函数调用（Function Calling）机制实现结构化输出
#
# 底层做了什么(文档 3-9 第 01 节)：
# 1. 把 RouteQuery 这个 BaseModel 自动转换成一个"函数回调"的参数 schema
#    （字段名 -> 参数名，Field.description -> 参数描述，Literal -> enum 枚举约束）
# 2. 告知 LLM 强制调用这个"虚假函数"
# 3. LLM 不会真的执行任何函数，它只输出这个函数的调用参数
# 4. LangChain 解析这段参数并实例化成 RouteQuery 对象返回
# 所以这里的 RouteQuery 本质上是"一个只为规范化输出而存在的假函数签名"
structured_llm = llm.with_structured_output(RouteQuery)

# 2.构建一个问题
# 这个问题包含 JavaScript 代码，应该被路由到 js_docs 数据源
question = """为什么下面的代码不工作了，请帮我检查下：

var a = "123"
"""

# invoke：同步调用大语言模型
# 返回的 res 是一个 RouteQuery 对象，不是字符串
res: RouteQuery = structured_llm.invoke(question)

# 输出结果验证
print(res)  # 输出完整的 RouteQuery 对象
print(type(res))  # 验证返回类型是 RouteQuery
print(res.datasource)  # 访问路由结果，应该是 "js_docs"
