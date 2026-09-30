#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 18:51
@Author  : thezehui@gmail.com
@File    : 1.bind函数使用技巧.py

===================================================================================
知识点讲解：Runnable.bind() 动态绑定默认调用参数
===================================================================================

1. bind() 方法的核心作用
   - 为 Runnable 组件预先绑定（固定）一部分调用参数
   - 返回一个新的 RunnableBinding 对象，原组件保持不变（不可变设计）
   - 绑定的参数会在每次 invoke 时自动作为 kwargs 传入底层组件

2. 为什么需要 bind()：设计动机
   - LCEL 管道中组件通过 | 连接，调用时只能传递单一的数据流
   - 无法在管道中途为某个组件补充额外的调用参数
   - bind() 提前把这些参数「焊死」在组件上，解决参数传递断层问题
   - 典型场景：
     * 场景 1：构建一个通用 LLM，在不同链里才绑定不同停止词
     * 场景 2：同一 LLM 派生高确定性链（temperature≈0.7）与高创造性链（temperature≈1.2）
     * 场景 3：为链中某 LLM 固定绑定工具或函数调用能力
   - 灵活性优势：无需实例化多个 LLM 对象，从同一个基础 LLM 派生即可

3. bind() 在 LLM 场景的典型用途
   - 覆盖模型参数：model、temperature、max_tokens、top_p
   - 绑定停止词：llm.bind(stop="world")
   - 绑定工具/函数调用：llm.bind(tools=[...])、llm.bind(functions=[...])
   - 绑定响应格式：llm.bind(response_format={"type": "json_object"})

4. 典型输出示例：停止词绑定的效果
   - 构建 prompt | llm.bind(stop="world") | StrOutputParser()，传入 {"query": "Hello world"}
   - 模型在生成到 "world" 处即停止，最终输出：
       Hello
   - 观察点：stop 参数在调用时自动注入 LLM 请求，无需在 invoke 时手动传，链的正常调用方式完全不变

5. bind() 与构造参数的优先级关系
   - 构造时 ChatOpenAI(model="A")，bind 时 .bind(model="B")
   - 最终请求使用 B，因为 bind 的参数在调用时覆盖构造时的默认值
   - 这使得同一个 LLM 实例可以派生出多个不同配置的变体

6. bind() 的底层实现原理
   - bind() 本质上是往 Runnable 的 kwargs 属性添加对应的字段
   - 生成一个新的 RunnableBinding 对象，包装原 Runnable 和绑定的 kwargs
   - 当 Runnable 组件执行调用时（invoke、stream、batch、ainvoke 等），会自动将
     kwargs 字段里的所有参数合并并覆盖默认调用参数
   - 从而完成动态添加默认调用参数的效果

7. bind() 与 configurable_fields() 的区别
   - bind()：构建期固定参数，构建链时确定，运行时不可再改
   - configurable_fields()：声明可配置字段，运行时通过 config 动态传值
   - 选择原则：固定不变用 bind，需按请求切换用 configurable_fields

8. 不可变性与链复用
   - bind() 不修改原对象，因此可以从同一个 llm 派生多个变体
   - 例如：fast_llm = llm.bind(model="flash")、pro_llm = llm.bind(model="pro")
   - 这些变体可分别用于不同的链，互不干扰

9. bind() 的适用边界
   - 虽然 bind() 是所有 Runnable 共有的方法，但并非所有组件都支持绑定默认调用参数
   - 部分组件底层没有默认调用参数的概念，例如 PromptTemplate 底层的 invoke 方法：
       def invoke(self, input, config=None) -> PromptValue:
           config = ensure_config(config)
           if self.metadata: config["metadata"] = {...}
           if self.tags: config["tags"] = config["tags"] + self.tags
           return self._call_with_config(self._format_prompt_with_error_handling, input, config, run_type="prompt")
   - 可见其 invoke 直接处理 input，并不消费额外的 kwargs，使用前需确认目标组件的 invoke 是否支持接收额外 kwargs

10. 数据流转过程
   - {"query": str} → prompt → Messages
   - Messages → RunnableBinding(llm, kwargs={"model": "deepseek-v4-pro"})
     → 实际请求：llm.invoke(Messages, model="deepseek-v4-pro") → AIMessage
   - AIMessage → StrOutputParser → str

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量（API Key 等）
dotenv.load_dotenv()

# ChatPromptTemplate.from_messages(messages: list) -> ChatPromptTemplate
#   作用：创建仅含单条用户消息的提示词模板
#   参数：messages - [(role, template)] 形式的消息列表
#   返回：ChatPromptTemplate 实例
#   说明：{query} 为透传变量，invoke 时由调用方填入
prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}")
])

# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例，构造时指定默认模型为 deepseek-flash
#   参数：model - 默认使用的模型名称
#   返回：ChatOpenAI 实例
#   注意：这里的 deepseek-flash 会在下一步被 bind 覆盖
llm = ChatOpenAI(model="deepseek-flash")

# 构建链：关键在于 llm.bind(model="deepseek-v4-pro")
#
# Runnable.bind(**kwargs) -> RunnableBinding
#   作用：为 Runnable 绑定固定的调用参数，返回新的绑定对象
#   参数：**kwargs - 任意键值对，会在调用底层组件时作为额外参数传入
#         对 ChatOpenAI 而言，可传入 model / temperature / max_tokens / stop / tools 等
#   返回：RunnableBinding 实例，包装了原 Runnable 与绑定的 kwargs
#   关键特性：
#     1. 不修改原 llm 对象，llm 仍然是 deepseek-flash
#     2. 绑定的 model 参数优先级高于构造时的 model 参数
#     3. 返回对象仍实现 Runnable 协议，可继续参与 | 管道组合
#   实际效果：虽然 llm 构造时是 deepseek-flash，但请求时使用 deepseek-v4-pro
chain = prompt | llm.bind(model="deepseek-v4-pro") | StrOutputParser()

# chain.invoke(input: dict) -> str
#   作用：执行链并获取模型回复
#   参数：input - {"query": 用户问题}
#   返回：字符串格式的模型输出
#   数据流转：
#     {"query": "你是什么模型呢？"}
#     → prompt → [HumanMessage("你是什么模型呢？")]
#     → llm（携带 bind 的 model="deepseek-v4-pro" 参数）→ AIMessage
#     → StrOutputParser → str
#   验证点：输出内容应反映的是 deepseek-v4-pro 而非 deepseek-flash
content = chain.invoke({"query": "你是什么模型呢？"})

print(content)

# ==================== 最佳实践与扩展写法 ====================
# 1. 绑定多个模型参数（一次 bind 传入多个 kwargs）
# chain = prompt | llm.bind(
#     model="deepseek-v4-pro",
#     temperature=0,          # 输出更确定
#     max_tokens=512,         # 限制最大输出长度
# ) | StrOutputParser()
#
# 2. 绑定停止词（模型遇到该字符串即停止生成）
# chain = prompt | llm.bind(stop=["\n\n", "###"]) | StrOutputParser()
#
# 3. 绑定 JSON 输出格式（配合 JsonOutputParser 使用）
# from langchain_core.output_parsers import JsonOutputParser
# chain = prompt | llm.bind(response_format={"type": "json_object"}) | JsonOutputParser()
#
# 4. 从同一个 llm 派生多个变体（体现 bind 的不可变特性）
# fast_llm = llm.bind(model="deepseek-flash", temperature=0.9)
# pro_llm = llm.bind(model="deepseek-v4-pro", temperature=0)
# fast_chain = prompt | fast_llm | StrOutputParser()
# pro_chain = prompt | pro_llm | StrOutputParser()
#
# 5. 链式调用 bind（后者覆盖前者的同名参数）
# final_llm = llm.bind(temperature=0.5).bind(temperature=0)   # 最终 temperature=0
#
# 6. 需要运行时动态切换时，改用 configurable_fields（见第 16 章）
# llm_configurable = ChatOpenAI(model="deepseek-flash").configurable_fields(
#     temperature=ConfigurableField(id="llm_temperature")
# )
# chain.invoke({"query": "..."}, config={"configurable": {"llm_temperature": 0}})
