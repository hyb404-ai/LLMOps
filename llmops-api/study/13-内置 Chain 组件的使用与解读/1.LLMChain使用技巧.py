#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/21 2:46
@Author  : thezehui@gmail.com
@File    : 1.LLMChain使用技巧.py

===================================================================================
知识点讲解：LLMChain 的演进与 LCEL 管道替代
===================================================================================

1. Chain 的本质与两种类型
   - Chain 描述的是「一系列操作或函数按特定顺序依次执行，前一个操作的输出作为后一个操作的输入」，这种范式也叫管道（Pipeline）/ 链式调用（Chain Calling）
   - LangChain 中存在两类链：
     * [推荐] 使用 LCEL 构建的链（顺序可执行链），是 1.x 的官方推荐写法
     * [遗产] 通过 Chain 类子类构建的链，不使用 LCEL，是独立的类，官方文档称为「遗产链」

2. LLMChain 的历史定位
   - LangChain 0.x 中最基本的 Chain 实现，封装的任务是：用用户输入格式化提示模板 → 传给 LLM → 取回响应
   - 封装 Prompt + LLM + OutputParser 的组合，并提供 run()、apply()、generate()、predict() 等多达 5 种调用方法，极大增加了使用复杂度
   - LangChain 1.x 已将其标记为废弃（传统链从 0.3.0 起逐渐淘汰），推荐用 LCEL 管道替代

3. Chain 基类与 Runnable 化（源码要点）
   - 0.1.0 之前的 Chain 基类定义（课程原版）：
       class Chain(BaseModel, ABC):
           memory: BaseMemory
           callbacks: Callbacks
           def __call__(self, inputs, return_only_outputs=False, callbacks=None) -> Dict[str, Any]: ...
   - 0.1.0 之后 Chain 基类继承了 RunnableSerializable，本身也是 Runnable 的子类
   - 因此遗产链也能直接用 Runnable 通用方法 invoke() 运行，与 LCEL 链在使用上被屏蔽了差异

4. LCEL 管道语法的优势
   - 用管道操作符 | 连接组件，语义清晰，符合函数式编程思想
   - 统一 Runnable 协议（invoke / batch / stream），降低学习成本
   - 自动处理数据流转，无需手动管理中间状态
   - 支持并行、条件分支、重试、回退等更复杂的组合模式

5. Runnable 协议的核心方法
   - invoke(input)：同步执行，接收单个输入，返回单个输出
   - batch(inputs)：批量执行，接收输入列表，返回输出列表（部分模型支持批量推理，比循环 invoke 更高效）
   - stream(input)：流式执行，返回生成器，逐块产出结果，适合实时交互

6. LCEL 管道的数据流转
   - 输入 dict → Prompt.invoke() → Messages
   - Messages → LLM.invoke() → AIMessage
   - AIMessage → StrOutputParser.invoke() → str
   - 每个组件的输出自动作为下一个组件的输入

7. 从 LLMChain 迁移到 LCEL 的映射
   - LLMChain(prompt, llm) → prompt | llm | StrOutputParser()
   - chain.run() → chain.invoke()
   - chain.apply() → chain.batch()
   - chain.predict(subject=...) → chain.invoke({"subject": ...})
   - 旧方法名的多样性被统一为三个标准方法

8. 适用场景
   - 单轮问答：直接使用 prompt | llm | parser
   - 多轮对话：配合 RunnableWithMessageHistory 管理历史
   - 复杂链路：通过 | 操作符连接更多自定义组件

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：从项目根目录的 .env 文件加载环境变量
#   返回：加载成功返回 True
#   常用环境变量：OPENAI_API_KEY, OPENAI_API_BASE
dotenv.load_dotenv()

# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：根据字符串模板快速创建聊天提示词模板
#   参数：template - 包含变量占位符的模板字符串（如 "{subject}"）
#   返回：ChatPromptTemplate 实例，实现 Runnable 协议
#   内部行为：将模板转换为 [HumanMessage(content=template)]
prompt = ChatPromptTemplate.from_template("请讲一个关于{subject}主题的冷笑话")

# ChatOpenAI(model: str, temperature: float = 0.7, ...) -> ChatOpenAI
#   作用：创建 OpenAI 兼容的聊天模型实例
#   参数：model - 模型名称（如 "deepseek-v4-pro"）
#   返回：ChatOpenAI 实例，实现 Runnable 协议
#   调用方式：通过 invoke/batch/stream 方法与模型交互
llm = ChatOpenAI(model="deepseek-v4-pro")

# ==================== LangChain 1.x 推荐写法 ====================
# 使用 LCEL 管道操作符 | 连接组件
# 数据流转：dict → prompt → messages → llm → AIMessage → parser → str
chain = prompt | llm | StrOutputParser()

# chain.invoke(input: dict) -> str
#   作用：同步执行链，处理单个输入
#   参数：input - 包含模板变量的字典（如 {"subject": "程序员"}）
#   返回：字符串格式的 LLM 输出
#   内部流程：
#     1. prompt.invoke({"subject": "程序员"}) → [HumanMessage(content="请讲一个关于程序员主题的冷笑话")]
#     2. llm.invoke([HumanMessage(...)]) → AIMessage(content="...")
#     3. parser.invoke(AIMessage(...)) → str
print(chain.invoke({"subject": "程序员"}))

# chain.batch(inputs: list[dict]) -> list[str]
#   作用：批量执行链，处理多个输入
#   参数：inputs - 包含多个输入字典的列表
#   返回：字符串列表，每个元素对应一个输入的输出
#   优势：某些模型支持批量推理，比循环调用 invoke 更高效
print(chain.batch([{"subject": "程序员"}, {"subject": "产品经理"}]))

# chain.stream(input: dict) -> Iterator[str]
#   作用：流式执行链，逐块返回 LLM 生成的内容
#   参数：input - 包含模板变量的字典
#   返回：生成器，yield 每个生成的文本片段
#   适用场景：实时交互（如聊天界面），提升用户体验
for chunk in chain.stream({"subject": "测试工程师"}):
    # flush=True 确保立即输出，不缓冲
    # end="" 避免自动换行，实现连续输出效果
    print(chunk, end="", flush=True)
print("")

# ==================== 附：旧的使用方式(LangChain 0.x，已废弃，仅作对照) ====================
# LLMChain 提供了多种调用方法，但本质功能相同，增加学习成本
# from langchain_classic.chains.llm import LLMChain
#
# chain = LLMChain(prompt=prompt, llm=llm)
#
# # 以下方法功能相似，但返回格式略有差异
# print(chain("程序员"))                          # 返回 {"text": "...", "subject": "程序员"}
# print(chain.run("程序员"))                      # 返回 str
# print(chain.apply([{"subject": "程序员"}]))      # 返回 [{"text": "..."}]
# print(chain.generate([{"subject": "程序员"}]))   # 返回 LLMResult 对象
# print(chain.predict(subject="程序员"))          # 返回 str
# print(chain.invoke({"subject": "程序员"}))       # 返回 {"text": "..."}
#
# 迁移建议：
#   - 用 LCEL 管道替代 LLMChain
#   - 统一使用 invoke/batch/stream 三种标准方法
#   - 通过 StrOutputParser 确保输出为纯字符串
