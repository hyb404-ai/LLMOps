#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 23:53
@Author  : thezehui@gmail.com
@File    : 1.BaseChatMemory运行流程解析.py
"""
from langchain_classic.memory.chat_memory import BaseChatMemory


# BaseChatMemory 是抽象基类(ABC),memory_variables / load_memory_variables
# 由子类实现,所以这里定义一个最小实现,用来观察它的运行流程
class SimpleChatMemory(BaseChatMemory):
    @property
    def memory_variables(self) -> list:
        # 该记忆会向链的输入里注入哪些变量名
        return ["chat_history"]

    def load_memory_variables(self, inputs: dict) -> dict:
        # 从 chat_memory 读取历史,return_messages 决定返回消息对象还是纯文本
        return {self.memory_variables[0]: list(self.chat_memory.messages)}


memory = SimpleChatMemory(
    input_key="query",
    output_key="output",
    return_messages=True,
    # chat_history 假设
)

# 1. 还没写入任何上下文,此时历史为空
memory_variable = memory.load_memory_variables({})
print("保存前:", memory_variable)

# content = chain.invoke({"query": "你好，我是慕小课你是谁", "chat_history": memory_variable.get("chat_history")})
memory.save_context({"query": "你好，我是慕小课你是谁"}, {"output": "你好，我是ChatGPT,有什么可以帮到您的"})

# 2. save_context 写入后,再取一次就能拿到刚才那轮问答
memory_variable = memory.load_memory_variables({})
print("保存后:", memory_variable)
# content = chain.invoke({"query": "你好，我是慕小课你是谁", "chat_history": memory_variable.get("chat_history")})
