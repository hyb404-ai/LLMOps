#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 11:06
@Author  : thezehui@gmail.com
@File    : 2.LCEL表达式简化版本.py

===================================================================================
知识点讲解：LCEL 管道操作符的使用
===================================================================================

1. LCEL 是什么：声明式组合语法
   - LCEL（LangChain Expression Language）是 LangChain 1.x 引入的声明式组合语法
   - 用管道操作符 | 连接多个组件形成数据处理链，所有组件都遵循统一的 Runnable 接口
   - 相比上一个文件手写的 Chain，LCEL 更简洁，且天然补齐了批量 / 流式 / 异步等高级能力

2. 为什么需要统一的 Runnable 协议（标准接口全表）
   - 设计目的：为尽可能简化「创建自定义链」，LangChain 官方实现了 Runnable 协议，适用于绝大部分组件
   - 协议一次性补齐 7 个标准入口，对照手写 Chain 只有 invoke 一个方法：
     * stream      : 流式返回响应块，组件不支持流式则直接输出
     * invoke      : 调用组件并得到结果
     * batch       : 批量调用并得到结果
     * astream     : stream 的异步版本
     * ainvoke     : invoke 的异步版本
     * abatch      : batch 的异步版本
     * astream_log : 除流式返回最终响应块外，还流式返回「中间步骤」（过程可观测的官方方案之一，与回调互补）
   - 这正是「手写 Chain 功能过于简陋」的具体所指

3. 运算符重载：| 从哪来
   - Runnable 重写了 __or__ 与 __ror__ 两个魔术方法，这正是 Python 中 | 运算符的计算逻辑
   - 因此所有 Runnable 组件都可通过 | 或 pipe() 拼接成链
   - 为什么需要两个方法：
     * __or__  处理「左操作数是 Runnable」的情况，如 prompt | llm
     * __ror__ 处理「左操作数不是 Runnable」的反向情况，如 {"query": RunnablePassthrough()} | prompt
       此时左边是原生 dict，Python 回退调用右操作数的 __ror__，由 Runnable 把 dict 转成 RunnableParallel
   - 对比 Prompt 组件：Prompt 重载 __add__ 支撑 + 拼接，Runnable 重载 __or__/__ror__ 支撑 | 管道，两者层面不同

4. 等价写法：| 与 pipe() 以及显式 RunnableParallel
   - 本文件写法：chain = prompt | llm | parser
   - 文档等价写法（用 pipe 逐级追加，并在最前面显式包装输入）：
       composed_chain_with_pipe = (
           RunnableParallel({"query": RunnablePassthrough()})
           .pipe(prompt).pipe(llm).pipe(parser)
       )
   - 两点说明：
     * .pipe(x) 与 | x 完全等价，前者是方法形式，在动态 / 循环追加步骤时更方便
     * 上面多出的 RunnableParallel({"query": RunnablePassthrough()}) 是为允许直接传裸字符串；本文件传完整 dict，无需该包装

5. LCEL 与手写 Chain 的对比
   - 手写：Chain([prompt, llm, parser]).invoke(input)
   - LCEL：(prompt | llm | parser).invoke(input)
   - 二者本质同构：prompt | llm | parser 创建的是 RunnableSequence 对象，实现了 Runnable 协议
   - LCEL 额外带来类型兼容、可组合（嵌套 / 并行 / 条件分支）、统一接口等优势

6. 源码级实现：RunnableSequence.invoke 的底层机制
   - Runnable 底层的运行逻辑本质上也是将每个组件加入列表，再按顺序执行并返回最终结果 —— 与手写 Chain 同构
   - 核心源码（langchain_core/runnables/base.py）：
       def invoke(self, input, config=None):
           config = config_with_context(ensure_config(config), self.steps)
           callback_manager = get_callback_manager_for_config(config)
           run_manager = callback_manager.on_chain_start(dumpd(self), input, ...)
           # 调用所有步骤并逐个执行，输出作为下一个的输入
           for i, step in enumerate(self.steps):
               input = step.invoke(input, patch_config(config, callbacks=run_manager.get_child(f"seq:step:{i+1}")))
           run_manager.on_chain_end(input)
           return input
   - 观察点：每个 step 被标记为子 run（seq:step:i），这正是回调与 run_id 追踪能定位到每一步的基础

7. 典型输出与执行流程
   - 输入 dict → prompt 处理 → messages → llm 处理 → AIMessage → parser 处理 → str
   - 每一步自动将输出传给下一步，最终 chain.invoke 的返回值等于最后一个组件的输出

===================================================================================
"""

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()

# 1. 构建组件
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建提示词模板
#   参数：template - 模板字符串，"{query}" 表示一个变量占位符
#   返回：实现了 Runnable 协议的 ChatPromptTemplate 对象
prompt = ChatPromptTemplate.from_template("{query}")

# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例
#   参数：model - 指定使用的模型名称
#   返回：实现了 Runnable 协议的 ChatOpenAI 对象
llm = ChatOpenAI(model="deepseek-flash")

# StrOutputParser() -> StrOutputParser
#   作用：创建字符串解析器，提取 AIMessage 的 content 字段
#   返回：实现了 Runnable 协议的 StrOutputParser 对象
parser = StrOutputParser()

# 2. 创建链
# 管道操作符 | 的使用：prompt | llm | parser
#   等价于：RunnableSequence([prompt, llm, parser])
#   
#   工作原理：
#     - prompt | llm 返回一个新的 Runnable 对象（包含 prompt 和 llm）
#     - (prompt | llm) | parser 再次组合，形成最终的链
#   
#   类型推导：
#     - prompt: Runnable[dict, list[BaseMessage]]
#     - llm: Runnable[list[BaseMessage], AIMessage]
#     - parser: Runnable[AIMessage, str]
#     - chain: Runnable[dict, str]
#   
#   返回：RunnableSequence 对象，可以调用 invoke/batch/stream 方法
chain = prompt | llm | parser

# 3. 调用链得到结果
# chain.invoke(input: dict) -> str
#   
#   参数：
#     - input: dict - 输入字典，需要包含模板中的所有变量
#   
#   执行流程：
#     1. prompt.invoke({"query": "请讲一个程序员的冷笑话"})
#        → 返回消息列表 [HumanMessage(content="请讲一个程序员的冷笑话")]
#     
#     2. llm.invoke([HumanMessage(...)])
#        → 调用 DeepSeek API，返回 AIMessage(content="...")
#     
#     3. parser.invoke(AIMessage(...))
#        → 提取 content 字段，返回字符串
#   
#   返回：str - 最终的文本结果
print(chain.invoke({"query": "请讲一个程序员的冷笑话"}))

# 其他调用方式示例：
# 
# 批量调用：
# results = chain.batch([
#     {"query": "请讲一个程序员的冷笑话"},
#     {"query": "请讲一个产品经理的冷笑话"}
# ])
# 
# 流式调用：
# for chunk in chain.stream({"query": "请讲一个程序员的冷笑话"}):
#     print(chunk, end="", flush=True)
