#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/6 11:37
@Author  : thezehui@gmail.com
@File    : 2.Runnable回退机制.py

===================================================================================
知识点讲解：with_fallbacks() 回退机制提升系统可用性
===================================================================================

1. 组件定位：什么是回退（Fallback）机制
   - 在某些场合中，对于 Runnable 组件的出错，并不想执行重试方案，
     而是执行特定的备份/回退方案（例如 OpenAI 的 LLM 异常时，自动切换到文心一言）
   - LangChain 提供对应的回退机制——with_fallback()：当主组件失败时，自动切换到备用组件
   - 返回 RunnableWithFallbacks，仍实现 Runnable 协议，可参与 | 管道组合
   - 属于容错设计中的降级策略，目标是保证服务"可用"而非"最优"
   - 与重试的区别：重试是"再试同一个"，回退是"换一个试"

2. with_fallbacks() 的核心参数
   - fallbacks: Sequence[Runnable] - 备用组件列表，按顺序依次尝试，必填
   - exceptions_to_handle: tuple - 触发回退的异常类型，默认 (Exception,)
   - exception_key: str - 可选，将异常信息以该键传入备用组件的输入

3. 回退的执行顺序与失败传播
   - 先执行主组件；成功则直接返回，不触碰备用组件
   - 主组件失败 → 尝试 fallbacks[0] → 仍失败 → 尝试 fallbacks[1] → ...
   - 全部失败时，抛出最后一个组件的异常
   - 可注册多级回退，形成"主 → 备1 → 备2"的降级链条

4. 本示例的容错场景
   - 主模型故意写为不存在的 deepseek-v4-pro-18k，调用必然失败
   - 备用模型为百度文心一言 QianfanChatEndpoint
   - 实际效果：主模型报错后自动切换到文心一言，用户侧仍能得到正常回复

5. 底层实现原理
   - 运行流程简单直接：按照主组件 → fallbacks 列表的顺序依次尝试
   - 每个组件执行时捕获 exceptions_to_handle 中指定的异常类型
   - 一旦某个组件成功执行，立即返回结果，不再尝试后续组件
   - 所有组件都失败时，抛出最后一个组件的异常
   - 核心实现在 RunnableWithFallbacks 类的 invoke 方法中

6. 典型的回退设计模式
   - 跨厂商回退：OpenAI 故障时切换到文心一言/通义千问，规避单点风险
   - 降级回退：强模型不可用时降级到轻量模型，保证基本可用性
   - 兜底回退：所有模型都失败时返回预设的固定文案，避免直接报错
   - 配额回退：主账号限流时切换到备用 API Key

7. 与 configurable_alternatives 的对比
   - with_fallbacks：被动触发，仅在异常发生时自动切换，调用方无感
   - configurable_alternatives：主动选择，由调用方通过 config 显式指定
   - 两者正交，可叠加使用，分别解决"容错"与"路由"两类问题

8. 注意事项
   - 备用组件必须与主组件的输入输出类型兼容，否则回退后链会断裂
   - 建议用 exceptions_to_handle 精确限定异常类型，避免掩盖代码 bug
   - 回退会叠加延迟（主组件失败耗时 + 备用组件耗时），需评估超时预算
   - 回退发生时应记录日志与监控告警，否则故障会被静默吞掉

9. 数据流转过程
   - {"query": "你好，你是?"} → prompt → Messages
   - → RunnableWithFallbacks 主组件 ChatOpenAI("deepseek-v4-pro-18k") → 抛出异常
   - → 捕获异常，切换至 QianfanChatEndpoint() → AIMessage
   - → StrOutputParser → str

===================================================================================
"""
import dotenv
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量
#   注意：备用模型文心一言需要 QIANFAN_AK / QIANFAN_SK 环境变量
dotenv.load_dotenv()

# 1.构建prompt与LLM，并将model切换为不存在的deepseek-v4-pro-18k引发出错
#
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建透传用户输入的聊天提示词模板
#   参数：template - "{query}"
#   返回：ChatPromptTemplate 实例
prompt = ChatPromptTemplate.from_template("{query}")

# Runnable.with_fallbacks(
#     fallbacks: Sequence[Runnable],
#     exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
#     exception_key: str = None,
# ) -> RunnableWithFallbacks
#   作用：为 Runnable 注册备用组件，主组件失败时自动切换执行
#   参数：
#     fallbacks - 备用 Runnable 列表，按列表顺序依次尝试
#     exceptions_to_handle - 触发回退的异常类型元组，默认捕获所有 Exception
#     exception_key - 若指定，会把主组件的异常对象以该键注入备用组件的输入 dict
#   返回：RunnableWithFallbacks 实例，仍实现 Runnable 协议，可参与 | 组合
#   执行逻辑：
#     1. 调用主组件 ChatOpenAI(model="deepseek-v4-pro-18k")
#     2. 因模型名不存在，API 返回错误并抛出异常
#     3. 异常类型匹配 exceptions_to_handle，触发回退
#     4. 调用 fallbacks[0] 即 QianfanChatEndpoint()，成功返回 AIMessage
#   类型兼容性：两个模型都接收 Messages 返回 AIMessage，因此可安全互换
llm = ChatOpenAI(model="deepseek-v4-pro-18k").with_fallbacks([QianfanChatEndpoint()])

# 2.构建链应用
# LCEL 管道：dict → prompt → Messages → llm（含回退）→ AIMessage → parser → str
# 说明：回退逻辑被封装在 llm 内部，链的其他部分对此完全透明
chain = prompt | llm | StrOutputParser()

# 3.调用链并输出结果
# chain.invoke(input: dict) -> str
#   作用：执行链，主模型失败时自动由备用模型接管
#   参数：input - {"query": "你好，你是?"}
#   返回：最终成功的模型回复字符串
#   实际执行序列：
#     1. prompt.invoke({"query": "你好，你是?"}) → [HumanMessage("你好，你是?")]
#     2. 主模型 deepseek-v4-pro-18k 调用失败（模型不存在）
#     3. 自动回退到 QianfanChatEndpoint()，返回 AIMessage
#     4. StrOutputParser 提取 content → str
#   验证点：尽管主模型必然失败，程序仍能正常输出结果，体现回退的容错价值
content = chain.invoke({"query": "你好，你是?"})
print(content)

# ==================== 最佳实践与扩展写法 ====================
# 1. 多级回退（主 → 备1 → 备2，逐级降级）
# llm = ChatOpenAI(model="deepseek-v4-pro-18k").with_fallbacks([
#     ChatOpenAI(model="deepseek-v4-pro"),     # 第一备选：同厂商的正确模型
#     ChatOpenAI(model="deepseek-flash"),      # 第二备选：降级到轻量模型
#     QianfanChatEndpoint(),                   # 第三备选：切换厂商
# ])
#
# 2. 精确限定触发回退的异常类型（避免掩盖代码 bug）
# from openai import RateLimitError, APITimeoutError
# llm = ChatOpenAI(model="deepseek-v4-pro").with_fallbacks(
#     [QianfanChatEndpoint()],
#     exceptions_to_handle=(RateLimitError, APITimeoutError),
# )
#
# 3. 重试 + 回退组合（推荐的生产级容错配置）
# llm = (
#     ChatOpenAI(model="deepseek-v4-pro")
#     .with_retry(stop_after_attempt=3)          # 先对主模型重试 3 次
#     .with_fallbacks([QianfanChatEndpoint()])   # 仍失败才切换备用厂商
# )
#
# 4. 对整条链做回退（备用链可使用完全不同的 prompt 策略）
# simple_chain = ChatPromptTemplate.from_template("简要回答：{query}") | ChatOpenAI(
#     model="deepseek-flash") | StrOutputParser()
# robust_chain = chain.with_fallbacks([simple_chain])
#
# 5. 固定文案兜底（保证任何情况下都不向用户抛异常）
# from langchain_core.runnables import RunnableLambda
# default_answer = RunnableLambda(lambda x: "抱歉，服务暂时不可用，请稍后重试。")
# robust_chain = chain.with_fallbacks([default_answer])
#
# 6. 将异常信息传递给备用组件（便于备用组件感知失败原因）
# fallback_prompt = ChatPromptTemplate.from_template(
#     "上一次调用出现错误：{error}\n请回答用户问题：{query}"
# )
# fallback_chain = fallback_prompt | ChatOpenAI(model="deepseek-flash") | StrOutputParser()
# robust_chain = chain.with_fallbacks([fallback_chain], exception_key="error")
#
# 7. 回退与动态选择的区别总结
#   - with_fallbacks：被动容错，异常时自动切换，调用方无需干预
#   - configurable_alternatives：主动路由，调用方通过 config 显式指定组件
#   - 生产建议：两者叠加使用，先按业务规则路由，再由回退兜底
#
# 8. 可观测性建议
#   - 回退发生时应记录日志与埋点，否则主模型故障会被静默掩盖
#   - 建议对回退触发率设置监控告警，及时发现主链路问题
