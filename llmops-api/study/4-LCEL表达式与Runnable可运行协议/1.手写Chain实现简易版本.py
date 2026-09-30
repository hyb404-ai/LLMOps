#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 10:56
@Author  : thezehui@gmail.com
@File    : 1.手写Chain实现简易版本.py

===================================================================================
知识点讲解：LCEL 与 Runnable 协议基础（手写 Chain）
===================================================================================

1. 设计动机：多组件 invoke 嵌套写法的缺陷
   - 引入 LCEL 之前，组装应用只能靠 invoke 层层嵌套：
       content = parser.invoke(
           llm.invoke(
               prompt.invoke({"query": req.query.data})
           )
       )
   - 这种写法存在三个明确缺陷：
     * 嵌套式写法让维护性与可读性大大降低，修改某个组件异常困难
     * 没法得知每一步的具体结果与执行进度，出错时难以排查
     * 嵌套式写法没法集成大量组件，组件变多时代码会变成「一次性」代码
   - 文档用「前端嵌套 / 回调地狱」作类比，本质是同一类问题
   - 关键思考：能否把嵌套写法改成平级调用，屏蔽嵌套带来的缺陷 —— 这正是本文件手写 Chain 的出发点

2. 推导：从「各组件方法各异」到「共同的 invoke」
   - 观察 Prompt、Model、OutputParser 各自有独立的调用方式：
     * Prompt 组件  : format、invoke、to_string、to_messages
     * Model 组件   : generate、invoke、batch
     * OutputParser : parse、invoke
   - 关键发现：它们有一个「共同的调用方法 invoke」，且每个组件的输出恰好是下一个组件的输入
   - 由此推导方案：把所有组件装进一个列表，循环依次调用每个组件的 invoke，并把当前输出作为下一个输入
   - 本文件的 Chain 类就是这个推导结论的最小实现，把「嵌套调用」改写成「平级的 for 循环」

3. 手写 Chain 的实现与数据流转
   - Chain 类持有 steps 列表，invoke 时用 for 循环依次 step.invoke(output)，并打印每一步
   - 数据三段式流转：输入 dict → Prompt.invoke() → PromptValue/Messages → LLM.invoke() → AIMessage → Parser.invoke() → str
   - 每一步的输出是下一步的输入，揭示链式调用的本质：数据在组件间的顺序流动

4. 手写 Chain 的价值与局限
   - 价值：简化了组装过程，且因每一步都能打印，天然解决了「无法得知每一步结果与进度」的缺陷
   - 局限：功能过于简陋
     * 只实现了 invoke，没有 batch / stream / 异步版本
     * 没有回调、没有配置传递、没有并行与分支能力
     * 没有错误上下文（哪一步失败、失败前输入是什么）
   - 引出下一个问题：既然各组件都支持 invoke，LangChain 底层是否已统一了某种规范、支持更简单调用？
     —— 答案就是 Runnable 协议与 LCEL，见 2.LCEL表达式简化版本.py

5. 典型输出示例与观察点
   - 因每一步都打印 step 与中间结果，可清楚看到三段式数据形态演进：
     * 第 1 步（prompt）：input_variables=['query']，执行结果 messages=[HumanMessage(content='你好，你是?')]
     * 第 2 步（llm）：打印模型客户端对象与 AIMessage，执行结果 content='你好！我是...'
     * 第 3 步（parser）：StrOutputParser 结果降维为纯字符串：你好！我是...
   - 观察点 1：AIMessage 上有 id='run-xxx'，与后续回调机制中的 run_id 同源，是链路追踪基础（见 6-利用回调功能调试链应用）
   - 观察点 2：打印 llm 对象时 api_key 显示为 SecretStr('**********')，是 pydantic 防止密钥被日志泄露
   - 观察点 3：最后一行是 chain.invoke 的返回值，与第 3 步结果一致，印证「链的输出 = 最后一个组件的输出」

===================================================================================
"""
from typing import Any

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量（OPENAI_API_KEY, OPENAI_API_BASE）
dotenv.load_dotenv()

# 1. 构建组件
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：根据字符串模板创建提示词模板对象
#   参数：template - 包含变量占位符的模板字符串（如 "{query}"）
#   返回：ChatPromptTemplate 实例，实现了 Runnable 协议
#   invoke 方法：接收 dict，返回消息列表
prompt = ChatPromptTemplate.from_template("{query}")

# ChatOpenAI(model: str, ...) -> ChatOpenAI
#   作用：创建 OpenAI 兼容的聊天模型实例
#   参数：model - 模型名称（如 "deepseek-flash"）
#   返回：ChatOpenAI 实例，实现了 Runnable 协议
#   invoke 方法：接收消息列表，返回 AIMessage 对象
llm = ChatOpenAI(model="deepseek-flash")

# StrOutputParser() -> StrOutputParser
#   作用：创建字符串输出解析器，提取 AIMessage.content
#   返回：StrOutputParser 实例，实现了 Runnable 协议
#   invoke 方法：接收 AIMessage，返回 str
parser = StrOutputParser()


# 2. 定义一个简易的链类
# 目的：模拟 LCEL 管道的工作原理，手动实现组件的顺序调用
class Chain:
    steps: list = []

    def __init__(self, steps: list):
        """
        初始化链
        
        参数：
            steps: list - 包含多个 Runnable 组件的列表，按执行顺序排列
        """
        self.steps = steps

    def invoke(self, input: Any) -> Any:
        """
        执行链的核心方法
        
        工作流程：
            1. 遍历 steps 列表中的每个组件
            2. 调用当前组件的 invoke() 方法，传入上一步的输出
            3. 将当前组件的输出作为下一步的输入
            4. 返回最后一个组件的输出
        
        参数：
            input: Any - 链的初始输入，通常是 dict 类型
        
        返回：
            Any - 最后一个组件的输出结果
        """
        for step in self.steps:
            # 关键步骤：调用当前组件的 invoke 方法，将上一步的输出传入
            # 数据流转：input 被当前组件处理后，结果重新赋值给 input
            input = step.invoke(input)
            
            # 调试输出：显示当前步骤和中间结果
            print("步骤:", step)
            print("输出:", input)
            print("===============")
        
        # 返回最终结果（最后一个组件的输出）
        return input


# 3. 编排链
# 创建 Chain 实例，传入组件列表
# 执行顺序：prompt → llm → parser
# 数据流转：dict → messages → AIMessage → str
chain = Chain([prompt, llm, parser])

# 4. 执行链并获取结果
# chain.invoke(input: dict) -> str
#   输入：{"query": "你好，你是?"}
#   中间过程：
#     - prompt.invoke({"query": "你好，你是?"}) → [HumanMessage(content="你好，你是?")]
#     - llm.invoke([HumanMessage(...)]) → AIMessage(content="我是DeepSeek...")
#     - parser.invoke(AIMessage(...)) → "我是DeepSeek..."
#   输出：最终的字符串结果
print(chain.invoke({"query": "你好，你是?"}))

# 对比 LCEL 管道写法（等价实现）：
# chain = prompt | llm | parser
# result = chain.invoke({"query": "你好，你是?"})
# 
# LCEL 的优势：
#   - 语法更简洁，使用 | 操作符连接组件
#   - 自动处理数据传递和类型转换
#   - 支持更多高级特性（并行、重试、回退等）
