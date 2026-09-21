#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/8 22:11
@Author  : thezehui@gmail.com
@File    : 4.复用提示模板.py
"""
from langchain_core.prompts import PromptTemplate

full_template = PromptTemplate.from_template("""{instruction}

{example}

{start}""")

# 描述模板
instruction_prompt = PromptTemplate.from_template("你正在模拟{person}")

# 示例模板
example_prompt = PromptTemplate.from_template("""下面是一个交互例子：

Q: {example_q}
A: {example_a}""")

# 开始模板
start_prompt = PromptTemplate.from_template("""现在，你是一个真实的人，请回答用户的问题:

Q: {input}
A:""")

# PipelinePromptTemplate 已在 LangChain 1.x 中被移除，这里手动完成其“先格式化子模板、再填充到最终模板”的逻辑
pipeline_prompts = [
    ("instruction", instruction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

input_variables = {
    "person": "雷军",
    "example_q": "你最喜欢的汽车是什么?",
    "example_a": "小米su7",
    "input": "你最喜欢的手机是什么?",
}

# 逐个格式化子模板，得到 {instruction}、{example}、{start} 的占位内容
pipeline_values = {
    name: prompt.invoke(input_variables).to_string()
    for name, prompt in pipeline_prompts
}
print(pipeline_values)

# 将子模板内容填充到最终模板中
# print(full_template.invoke(pipeline_values).to_string())
