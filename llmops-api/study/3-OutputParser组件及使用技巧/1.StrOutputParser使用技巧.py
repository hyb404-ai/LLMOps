#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 1:24
@Author  : thezehui@gmail.com
@File    : 1.StrOutputParser使用技巧.py

===================================================================================
知识点讲解：StrOutputParser 字符串输出解析器
===================================================================================

1. 设计动机：为什么需要输出解析器
   - 朴素调用模型返回的都是「字符串」而非结构化数据，且夹带寒暄、编号、说明等多余内容，某些场合我们只需要对应的返回值
   - 朴素改进：把格式约束写进提示词，让模型先按规则输出再解析
   - 抽象结论：要让 LLM 理解你想要的格式，必须先向 LLM 告知输出结构，推理后再按约定解析——这就是「输出解析器」的由来
   - 一句话拆解：输出解析器 = 预设提示 + 解析功能（StrOutputParser 是其中「预设提示」为空的极简特例）

2. OutputParser 的三类解析场景
   - 将非结构化文本转为结构化数据（json / dict 等）
   - 截取部分文本：只要结果中的某一部分
   - 自定义处理：在提示中注入其他指令，解析器据此做对应解析

3. OutputParser 的两个抽象函数
   - get_format_instructions：约定输出格式并转换为描述文本
   - parse：解析 LLM 输出为约定格式
   - 实际使用中直接 invoke，底层仍调 parse；parser.invoke(ai_message) 与 parser.parse(文本) 同源，前者多做了「从 AIMessage 取出 content」这一步
   - 这两个函数也是「自定义输出解析器」需要实现的核心

4. StrOutputParser 的功能定位
   - 最简单的输出解析器，提取 AIMessage.content 转为纯字符串
   - 适用于只需文本内容的场景（不需要元数据）
   - 在 LCEL 管道中通常作为最后一步：prompt | llm | StrOutputParser()
   - 支持流式输出：逐块转换为字符串

5. 为什么需要 StrOutputParser
   - llm.invoke() 返回 AIMessage 对象，含 type、content、response_metadata 等字段
   - 大多数场景只需 content，StrOutputParser 简化了 ai_message.content 的提取
   - 实现了 Runnable 协议（invoke / batch / stream），可无缝集成 LCEL，支持链式组合

6. StrOutputParser 源码级实现
   - parse 源码极简，原样返回输入文本，不做任何字符串加工：
       def parse(self, text: str) -> str:
           # Returns the input text with no changes.
           return text
   - 其价值在于「类型归一」——把 AIMessage 降维成 str，使其在 LCEL 链条中充当标准收尾节点
   - 最小可运行示例（无需模型）：StrOutputParser().parse("程序员的梦工厂") -> "程序员的梦工厂"

7. 典型输出示例
   - 链末挂 StrOutputParser 后得到纯字符串而非 AIMessage
   - 若去掉 parser，content 会是 AIMessage 对象，下游需手动取 .content 才能当文本用

8. 其他思路：不用解析器的方案
   - 若模型支持 Function / Tool Calling，可直接定义函数并规定其参数结构，强制模型调用该函数，精准按约束输出
   - 对应 LangChain 1.x 的 llm.with_structured_output(...)
   - 延伸思考：解析失败可重试 / 修复 / 降级为纯文本 / 收紧提示词；小参数模型且无工具回调时，格式约束越弱越稳定（StrOutputParser 优于严格的 JSON / Pydantic 解析器）

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()

# 1. 编排提示模板
# 创建简单的单消息模板，包含 query 变量
prompt = ChatPromptTemplate.from_template("{query}")

# 2. 构建大语言模型
llm = ChatOpenAI(model="deepseek-flash")

# 3. 创建字符串输出解析器
# StrOutputParser 将 AIMessage 对象转换为纯字符串（提取 content 字段）
parser = StrOutputParser()

# 4. 调用大语言模型生成结果并解析
# 执行流程：
#   - prompt.invoke() 生成消息列表
#   - llm.invoke() 调用模型，返回 AIMessage 对象
#   - parser.invoke() 提取 AIMessage.content，返回字符串
content = parser.invoke(llm.invoke(prompt.invoke({"query": "你好，你是?"})))

# 输出纯字符串结果（而不是 AIMessage 对象）
print(content)

# 推荐的 LCEL 管道写法（等价于上面的嵌套调用）：
# chain = prompt | llm | parser
# content = chain.invoke({"query": "你好，你是?"})
