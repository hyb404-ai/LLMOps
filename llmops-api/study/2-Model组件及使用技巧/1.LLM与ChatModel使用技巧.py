#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/9 18:49
@Author  : thezehui@gmail.com
@File    : 1.LLM与ChatModel使用技巧.py

===================================================================================
知识点讲解：LangChain Model 组件基础
===================================================================================

1. 设计动机：LangChain 不实现模型，只提供标准接口
   - Model 是 LangChain 的核心组件，但 LangChain 本身不实现任何大模型
   - 它提供的是一个标准接口，用于封装不同类型的 LLM 并与之交互
   - 收益：业务代码面向统一接口编程，更换底层模型厂商（OpenAI / DeepSeek / 通义等）时改动最小

2. 模型两大分类与基类层次
   - LangChain 为两种类型的模型提供接口和集成：
     * LLM（Large Language Model）：使用「纯文本」作为输入和输出的文本补全类模型
     * Chat Model：使用「聊天消息列表」作为输入并返回聊天消息的聊天模型
   - 模型总基类 BaseLanguageModel 据此划分出两个子基类：
     BaseLLM（对应 LLM）/ BaseChatModel（对应 Chat Model）
   - 现代 API（如 OpenAI、DeepSeek）主要使用 ChatModel 接口

3. 关键屏蔽机制：输入类型的自动适配
   - 无论是 LLM 还是 Chat Model，都可以接受以下三种参数：
     * PromptValue（prompt.invoke() 的产物）
     * 字符串
     * 消息列表
   - 内部会根据模型的类型自动转换成字符串或消息列表
   - 这正是「屏蔽不同模型差异」的实现方式，也解释了为什么本例
     可以把 prompt.invoke(...) 的 PromptValue 直接丢给 llm.invoke(...)
   - 联动 Prompt 组件：Prompt 产出 PromptValue 这层中间态，
     Model 负责把 PromptValue 落到自己需要的形态，两端解耦

4. 调用大模型的三个标准方法（Runnable 协议）
   - invoke : 一次输入 -> 一次输出，最常用的同步调用
   - batch  : invoke 的批量版本，输入列表 -> 输出列表（见 2.Model批处理.py）
   - stream : invoke 的流式输出版本，每生成一个片段就返回一个片段（见 3.Model流式输出.py）
   - 这三个方法来自 Runnable 协议，Prompt / Model / OutputParser 共享同一套签名，
     这是后续 LCEL 能用 | 串起来的前提（见 4-LCEL表达式与Runnable可运行协议）

5. ChatOpenAI 组件
   - LangChain 对 OpenAI API 的封装，同时兼容 OpenAI 格式的其他服务（如 DeepSeek）
   - 通过 model 参数指定具体的模型名称
   - 自动从环境变量读取 API Key（OPENAI_API_KEY）和 Base URL（OPENAI_API_BASE）
   - 同一个 llm 实例可被多个并行 / 批量分支安全复用（内部无请求级可变状态）

6. Message 组件全景
   - Message 是所有消息的基类（BaseMessage），通用属性有三个：
     * type              : 消息类型
     * content           : 消息内容
     * response_metadata : 响应元数据
   - LangChain 封装的 Message 涵盖 5 种类型：
     * SystemMessage   : 系统消息，设定角色与约束
     * HumanMessage    : 人类消息，用户输入
     * AIMessage       : AI 消息，模型输出
     * FunctionMessage : 函数消息（旧版函数调用返回值）
     * ToolMessage     : 工具消息（工具调用返回值）

7. AIMessage 对象结构（ChatModel 的返回）
   - type: 消息类型（此处为 'ai'）
   - content: 模型生成的文本内容
   - response_metadata: 响应元数据（token 使用量、模型信息、结束原因等）
   - 本例读取的 ai_message.type / content / response_metadata
     正是上述三个通用属性，对任意 Message 子类都成立

8. invoke() 方法
   - 同步调用模型，接收 PromptValue 或消息列表（字符串在内部被自动包装）
   - 返回 AIMessage 对象
   - 是 Runnable 协议的标准方法，统一了所有组件的调用接口

9. 基础聊天应用的完整运行流程
   - 加入 Model 组件后，流程为：
     用户输入 -> Prompt 组装 -> Model 推理 -> AIMessage -> 输出解析 -> 返回
   - 本文件覆盖了前三步；输出解析见 3-OutputParser组件及使用技巧

10. 典型输出示例与观察点
    - type:
        ai
    - content:
        现在是18:12。好的，这是一个关于程序员的冷笑话：
        为什么程序员总是觉得寂寞？
        因为他们总是在和bug独处！
    - response_metadata:
        {'token_usage': {'completion_tokens': 53, 'prompt_tokens': 71,
                         'total_tokens': 124},
         'model_name': '...', 'system_fingerprint': '...',
         'finish_reason': 'stop', 'logprobs': None}
    - 观察点 1：response_metadata 里的 token_usage 是做成本核算 / 限流的关键数据，
      LLMOps 场景下通常需要落库统计
    - 观察点 2：finish_reason='stop' 表示正常结束；若为 'length'
      则说明被 max_tokens 截断，需要排查
    - 观察点 3：模型能答出「现在是几点」，靠的是 Prompt 里 partial 注入的 now，
      而不是模型自身具备时间感知能力

===================================================================================
"""
from datetime import datetime

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量（OPENAI_API_KEY, OPENAI_API_BASE）
dotenv.load_dotenv()

# 1. 编排 Prompt：构造多轮对话的提示词模板
prompt = ChatPromptTemplate.from_messages([
    # system 消息：定义 AI 的角色和行为约束
    ("system", "你是DeepSeek开发的聊天机器人，请回答用户的问题，现在的时间是{now}"),
    
    # human 消息：用户的实际输入
    ("human", "{query}"),
    
# partial() 提前绑定 now 参数为当前时间，避免每次调用都传入
]).partial(now=datetime.now())

# 2. 创建大语言模型实例
# ChatOpenAI 兼容 OpenAI API 格式，可用于 OpenAI、DeepSeek 等服务
# model 参数指定模型名称（deepseek-flash 是 DeepSeek 的快速模型）
llm = ChatOpenAI(model="deepseek-flash")

# 3. 执行推理：prompt.invoke() 生成消息列表，llm.invoke() 调用模型
# 两步 invoke 的组合可以简化为 LCEL 管道：prompt | llm
ai_message = llm.invoke(prompt.invoke({"query": "现在是几点，请讲一个程序员的冷笑话"}))

# 4. 解析 AIMessage 对象
# type 字段：消息类型，这里是 "ai"
print(ai_message.type)

# content 字段：模型生成的文本内容
print(ai_message.content)

# response_metadata 字段：响应元数据（token 消耗、模型版本、finish_reason 等）
print(ai_message.response_metadata)
