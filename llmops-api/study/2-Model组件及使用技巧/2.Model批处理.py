#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/9 18:59
@Author  : thezehui@gmail.com
@File    : 2.Model批处理.py

===================================================================================
知识点讲解：Model 批处理（Batch Processing）
===================================================================================

1. 批处理的应用场景
   - 需要同时处理多个独立的请求（如批量翻译、批量问答、批量摘要）
   - 提高吞吐量，减少网络往返次数
   - 适合对实时性要求不高、需要一次性拿到多份结果的场景

2. batch 在 Runnable 方法体系中的定位
   - 文档把调用大模型最常用的方法归纳为三个，batch 的定义是：
     「invoke 的批量版本，可以一次性生成多个内容」
   - 三者关系：
     * invoke : 一次输入 -> 一次输出
     * batch  : 输入列表 -> 输出列表（本文件）
     * stream : 一次输入 -> 增量输出流（见 3.Model流式输出.py）
   - batch 同属 Runnable 协议标准接口，因此不止 Model 有 batch，
     Prompt、OutputParser 以及整条 Chain 都有 batch

3. batch() 方法的特性
   - Runnable 协议的标准方法之一（invoke, batch, stream）
   - 接收输入列表，返回输出列表，且输入与输出保持顺序严格对应
   - 底层实现可能并发调用 API（取决于具体的模型实现），部分 API 提供原生批处理支持

4. batch() vs invoke()
   - invoke()：单次请求，适合实时交互场景
   - batch()：批量请求，适合离线处理或数据处理场景
   - batch() 的输入和输出都是列表，顺序一一对应，可用 zip(inputs, outputs) 配对

5. 输入元素依然享受「输入类型自动适配」
   - LLM 与 Chat Model 都能接受 PromptValue / 字符串 / 消息列表
   - 这一点在 batch 中同样成立：列表里的每个元素都可以是上述任一形态
   - 本例传入的是 prompt.invoke(...) 产出的 PromptValue 列表；
     等价地也可以直接传字符串列表，例如 llm.batch(["你好", "讲个笑话"])

6. 性能优化建议
   - 批处理可以减少 HTTP 连接开销
   - 部分 API 提供原生批处理支持（如 OpenAI Batch API）
   - 对于不支持批处理的 API，LangChain 会自动并发调用
   - 可通过 max_concurrency 参数控制并发数，避免触发厂商 QPS/RPM 限流

7. 错误处理
   - 批处理中某个请求失败不会影响其他请求（异常通常被包装进对应位置的结果）
   - 需要针对每个结果单独检查错误状态
   - 可以配合 max_concurrency 参数控制并发数

8. 典型输出示例与观察点
   - 文档示例用两个不同的 subject 批量提问，输出为两段互不相干的回答：
       当程序员去海滩度假时，他们把沙子编成了二进制。...
       为什么Python喜欢处理字符串？
       因为它不喜欢弄得太复杂，总是喜欢把问题拆成小块来处理！
   - 观察点：返回值是一个与输入等长、且顺序严格对应的 AIMessage 列表，
     因此可以用 zip(inputs, ai_messages) 把请求与响应配对
   - 每个元素仍是完整的 AIMessage，各自带有独立的 response_metadata
     （即 token 消耗是逐条统计的，需要累加才是本次批处理的总消耗）

===================================================================================
"""
from datetime import datetime

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()

# 1. 编排 Prompt：构造提示词模板
prompt = ChatPromptTemplate.from_messages([
    # system 消息：定义 AI 的角色
    ("system", "你是DeepSeek开发的聊天机器人，请回答用户的问题，现在的时间是{now}"),
    
    # human 消息：用户输入占位符
    ("human", "{query}"),
    
# partial() 提前绑定时间参数
]).partial(now=datetime.now())

# 2. 创建大语言模型实例
llm = ChatOpenAI(model="deepseek-flash")

# 3. 使用 batch() 方法批量调用模型
# 接收消息列表，返回 AIMessage 列表
# 每个 prompt.invoke() 生成一个消息列表，作为批处理的一个输入
ai_messages = llm.batch([
    prompt.invoke({"query": "你好，你是?"}),
    prompt.invoke({"query": "请讲一个关于程序员的冷笑话"}),
])

# 4. 遍历批处理结果
# ai_messages 是 AIMessage 对象的列表，顺序与输入对应
for ai_message in ai_messages:
    # 输出每个请求的响应内容
    print(ai_message.content)
    print("====================")
