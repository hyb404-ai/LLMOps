#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 18:51
@Author  : thezehui@gmail.com
@File    : 1.bind函数使用技巧.py
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}")
])
llm = ChatOpenAI(model="deepseek-flash")

chain = prompt | llm.bind(model="deepseek-v4-pro") | StrOutputParser()

content = chain.invoke({"query": "你是什么模型呢？"})

print(content)
