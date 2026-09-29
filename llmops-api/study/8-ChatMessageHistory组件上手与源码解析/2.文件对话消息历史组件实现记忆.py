#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/22 23:33
@Author  : thezehui@gmail.com
@File    : 2.文件对话消息历史组件实现记忆.py
"""
import os

import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

dotenv.load_dotenv()

# 1.创建客户端&记忆
client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"))
chat_history = FileChatMessageHistory("./memory.txt")

# 2.循环对话
while True:
    # 3.获取用户的输入
    query = input("Human: ")

    # 4.检测用户是否退出对话
    if query == "q":
        exit(0)

    # 5.发起聊天对话
    print("AI: ", flush=True, end="")
    system_prompt = (
        "你是OpenAI开发的ChatGPT聊天机器人，可以根据相应的上下文回复用户信息，上下文里存放的是人类与你对话的信息列表。\n\n"
        f"<context>{chat_history}</context>\n\n"
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model='deepseek-v4-pro',
        extra_body={"thinking": {"type": "disabled"}},
        messages=messages,
        stream=True,
    )
    ai_content = ""
    in_reasoning = False
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # 6.1 输出思考过程
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_reasoning:
                print("[思考] ", flush=True, end="")
                in_reasoning = True
            print(reasoning, flush=True, end="")

        # 6.2 输出正式回答，并累积到ai_content中用于记录记忆
        if delta.content:
            if in_reasoning:
                print("\n[回答] ", flush=True, end="")
                in_reasoning = False
            ai_content += delta.content
            print(delta.content, flush=True, end="")

    chat_history.add_user_message(query)
    chat_history.add_ai_message(ai_content)
    print("")
