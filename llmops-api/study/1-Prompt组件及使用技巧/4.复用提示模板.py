#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/8 22:11
@Author  : thezehui@gmail.com
@File    : 4.复用提示模板.py

===================================================================================
知识点讲解：提示模板的复用与管道式组合

1. 复用提示模板的意义
   - 在复杂应用中，提示词通常由多个可复用的部分组成（指令、示例、输入等）
   - 通过模块化管理子模板，可以在不同场景下灵活组合，提高可维护性，减少重复代码

2. 管道式组合的思路（两阶段渲染）
   - 定义最终模板（full_template），包含多个占位符（如 instruction / example / start）
   - 定义多个子模板（instruction_prompt、example_prompt、start_prompt），先用输入变量各自格式化，得到子模板的输出字符串
   - 将子模板输出作为最终模板的输入，完成二次填充，即「把子模板的渲染结果当成最终模板的变量值」
   - 这就是「管道」一词的含义：最终模板的变量不由用户提供，而是由上一阶段的渲染结果提供

3. 版本演进：PipelinePromptTemplate 旧 API 与 1.x 替代方案
   - LangChain 0.x 提供 PipelinePromptTemplate 把上述两步封装起来，定位为「创建管道消息，可把提示模板当作变量快速复用」
   - 旧版写法：
       input_prompts = [("instruction", instruction_prompt),
                        ("example", example_prompt),
                        ("start", start_prompt)]
       pipeline_prompt = PipelinePromptTemplate(
           final_prompt=full_prompt, pipeline_prompts=input_prompts)
       pipeline_prompt.format(person="雷军", example_q="...", example_a="...", input="...")
   - LangChain 1.x 已移除 PipelinePromptTemplate，替代方案就是本文件的手动两阶段渲染写法
   - 两者执行语义完全一致，只是手写版本更透明，且能在中间步骤插入自定义处理逻辑
   - 若偏好声明式风格，1.x 中也可用 LCEL 表达同样的两阶段渲染，例如把 {"instruction": instruction_prompt | StrOutputParser(), ...} 这样的 RunnableParallel 接到 full_template 之前（参见 5-两个Runnable核心类的讲解与使用）

4. 变量传递机制要点
   - 各子模板只声明自己需要的变量（instruction_prompt 只要 person，example_prompt 要 example_q / example_a，start_prompt 要 input）
   - 本例把所有变量放在同一个字典里统一传给每个子模板，这依赖 PromptTemplate 会忽略多余的键；这也是 PipelinePromptTemplate 原本的行为：调用方只需提供一份「扁平的」变量集合
   - 最终模板 full_template 的变量（instruction / example / start）不由用户提供，而是由上一阶段的渲染结果提供

5. 典型输出示例与结构解读
   - 完成两阶段渲染后，拼装出的完整提示词如下（本例打印的是中间态字典，可直观看到三个子模板各自渲染出的文本片段）：
       你正在模拟雷军。

       下面是一个交互例子:

       Q: 你最喜欢的汽车是什么?
       A: 小米su7

       现在，你是一个真实的人，请回答用户的问题！

       Q: 你最喜欢的手机是什么?
       A:
   - 结构解读：这正是一个典型的 few-shot 提示词骨架 ——「角色指令（instruction）+ 示例（example）+ 待回答的问题（start）」，结尾留空的 "A:" 用于引导模型顺着格式续写答案

6. 应用场景
   - Few-shot learning：提供示例来引导模型输出
   - 角色扮演：通过指令和示例定义模型的角色行为
   - 复杂提示词的结构化管理

7. 实现细节
   - 使用字典推导式批量格式化子模板，invoke() 接收所有变量，to_string() 转换为字符串
   - 子模板的输出作为最终模板的占位符值，二次填充后即可得到完整提示词

===================================================================================
"""
from langchain_core.prompts import PromptTemplate

# 定义最终模板，包含三个占位符：instruction, example, start
# 这些占位符将由子模板的输出填充
full_template = PromptTemplate.from_template("""{instruction}

{example}

{start}""")

# 子模板 1：指令模板，定义角色扮演的身份
instruction_prompt = PromptTemplate.from_template("你正在模拟{person}")

# 子模板 2：示例模板，提供 Few-shot 示例引导模型输出风格
example_prompt = PromptTemplate.from_template("""下面是一个交互例子：

Q: {example_q}
A: {example_a}""")

# 子模板 3：开始模板，包含实际的用户输入
start_prompt = PromptTemplate.from_template("""现在，你是一个真实的人，请回答用户的问题:

Q: {input}
A:""")

# 定义管道式组合规则：(占位符名称, 对应的子模板)
# 这里模拟了 LangChain 0.x 中 PipelinePromptTemplate 的功能
pipeline_prompts = [
    ("instruction", instruction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

# 准备输入变量，包含所有子模板需要的变量
input_variables = {
    "person": "雷军",
    "example_q": "你最喜欢的汽车是什么?",
    "example_a": "小米su7",
    "input": "你最喜欢的手机是什么?",
}

# 第一步：逐个格式化子模板，得到每个占位符的实际内容
# 使用字典推导式，将每个子模板的输出转换为字符串
pipeline_values = {
    name: prompt.invoke(input_variables).to_string()
    for name, prompt in pipeline_prompts
}
print(pipeline_values)

# 第二步：将子模板的输出填充到最终模板中（已注释，取消注释可查看最终结果）
# print(full_template.invoke(pipeline_values).to_string())
