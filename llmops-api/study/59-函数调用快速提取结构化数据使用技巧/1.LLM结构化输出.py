#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/12 20:00
@Author  : thezehui@gmail.com
@File    : 1.LLM结构化输出.py

===================================================================================
知识点讲解：函数调用快速提取结构化数据
===================================================================================

1. with_structured_output() 方法简介
   - LangChain 提供的结构化输出快捷方法
   - 底层利用 LLM 的函数调用能力
   - 自动将输出解析为 Pydantic 对象
   - 比手动构造 JsonOutputParser 更简洁

2. with_structured_output() 参数详解
   - schema：Pydantic 模型、TypedDict 或 JSON Schema
   - method：输出方式
     * "function_calling"（默认）：使用函数调用能力
     * "json_mode"：使用 JSON 模式（需要模型支持）
     * "json_schema"：使用结构化输出 API
   - include_raw：是否同时返回原始响应（默认 False）

3. 三种 method 的区别
   - function_calling：将 schema 作为函数定义，LLM 生成函数调用
     * 兼容性最好，大多数模型支持
     * 可能受函数调用的限制
   - json_mode：强制 LLM 输出 JSON 格式
     * 需要提示词中说明 JSON 格式要求
     * 需要模型支持 response_format
   - json_schema：使用 OpenAI 的结构化输出功能
     * 保证 100% 符合 schema
     * 仅部分模型支持

4. Pydantic 模型的作用
   - 定义期望的数据结构
   - 类文档字符串作为函数描述
   - Field 的 description 描述字段含义
   - 提供类型验证和自动转换

5. json_mode 的注意事项
   - 必须在提示词中明确说明输出 JSON
   - 必须说明包含哪些字段
   - 否则可能输出不符合预期的结构

6. 返回值类型
   - 默认返回 Pydantic 模型实例（不是字典）
   - 可以通过属性访问：result.question、result.answer
   - include_raw=True 时返回字典：{"raw": ..., "parsed": ..., "parsing_error": ...}

7. 与其他方案的对比
   - with_structured_output()：最简洁，推荐首选
   - JsonOutputParser：更灵活，可自定义解析逻辑
   - PydanticOutputParser：功能类似，但依赖提示词
   - 手动函数调用：最灵活，但代码最复杂

8. 典型应用场景
   - 信息抽取：从文本中提取结构化字段
   - 数据清洗：将非结构化文本转换为结构化数据
   - 假设性问答生成：RAG 系统中生成 Q&A 对
   - 分类任务：输出分类标签和置信度
   - 实体识别：提取人名、地点、时间等

9. RAG 中的应用（本示例场景）
   - 从文档片段生成假设性问题
   - 用于 HyDE（Hypothetical Document Embeddings）
   - 用于 MultiVector 检索的问题向量化
   - 提升语义检索的准确性

10. 最佳实践
    - 优先使用 with_structured_output()
    - Pydantic 模型的 description 要清晰
    - json_mode 时提示词中明确字段要求
    - 复杂结构考虑拆分为多次调用
    - 添加异常处理应对解析失败

11. 底层源码解析（with_structured_output 的运行机制）
   - 方法根据传入的 method 走不同分支绑定运行时参数，核心逻辑（langchain_core ChatModel）等价如下：
       def with_structured_output(self, schema, *, method="function_calling", include_raw=False, **kwargs):
           if method == "function_calling":
               llm = self.bind_tools([schema], tool_choice="any")
               output_parser = (PydanticToolsParser(tools=[schema], first_tool_only=True)
                               if _is_pydantic_class(schema)
                               else JsonOutputKeyToolsParser(...))
           elif method == "json_mode":
               llm = self.bind(response_format={"type": "json_object"})
               output_parser = (PydanticOutputParser(pydantic_object=schema)
                               if _is_pydantic_class(schema) else JsonOutputParser())
           else:
               raise ValueError("Unrecognized method argument...")
           return llm | output_parser
   - 三条关键推论：
     * function_calling 本质是 bind_tools([schema], tool_choice="any")，即把 schema 当作一个强制调用的"虚假函数"，所以哪怕不用 LangChain，只要 bind 一个函数并设 tool_choice="any" 也能稳定拿到结构化输出
     * json_mode 本质是 bind(response_format={"type": "json_object"})，因此要求 prompt 中明确说明 JSON 格式与字段，否则易出错，且支持该模式的模型较少
     * 返回值就是 llm | output_parser 这条链：先让模型生成工具调用 / JSON，再由解析器转成 Pydantic 对象或字典
   - include_raw=True 时返回一个带 fallback 的结构：{raw, parsed, parsing_error}，解析失败也能拿到原始响应，便于排错
   - 结论：三种 method 中 function_calling 兼容性最好、最稳定，优先使用；json_mode 仅在模型不支持函数调用或确有需要时采用

===================================================================================
"""
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 定义结构化输出的 Pydantic 模型
# 类文档字符串会作为"函数描述"传递给 LLM
class QAExtra(BaseModel):
    """一个问答键值对工具，传递对应的假设性问题+答案"""
    # question 字段：假设性问题
    # Field 的 description 帮助 LLM 理解字段含义
    question: str = Field(description="假设性问题")
    # answer 字段：问题对应的答案
    answer: str = Field(description="假设性问题对应的答案")


# 创建大语言模型
llm = ChatOpenAI(model="deepseek-v4-pro")

# 使用 with_structured_output() 创建结构化输出的 LLM
# 参数说明：
#   - QAExtra：Pydantic 模型，定义输出结构
#   - method="json_mode"：使用 JSON 模式输出
#     * json_mode 要求提示词中明确说明 JSON 格式和字段
#     * 也可以使用 method="function_calling"（默认，兼容性更好）
# 返回值：调用后直接得到 QAExtra 实例，无需手动解析
structured_llm = llm.with_structured_output(QAExtra, method="json_mode")

# 构建提示词模板
# 关键点：使用 json_mode 时必须在提示词中：
#   1. 明确要求输出 JSON 格式
#   2. 明确说明包含哪些字段（question 和 answer）
prompt = ChatPromptTemplate.from_messages([
    ("system", "请从用户传递的query中提取出假设性的问题+答案。响应格式为JSON，并携带`question`和`answer`两个字段。"),
    ("human", "{query}")
])

# 构建 LCEL 链
# RunnablePassthrough() 将输入字符串传递给 query 键
# structured_llm 输出 QAExtra 实例（不是字符串或字典）
chain = {"query": RunnablePassthrough()} | prompt | structured_llm

# 执行链
# 输出是 QAExtra 实例，可以通过属性访问：
#   result = chain.invoke(...)
#   print(result.question)  # 访问问题
#   print(result.answer)    # 访问答案
print(chain.invoke("我叫慕小课，我喜欢打篮球，游泳。"))
