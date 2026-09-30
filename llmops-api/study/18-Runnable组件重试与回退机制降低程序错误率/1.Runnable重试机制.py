#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/6 9:59
@Author  : thezehui@gmail.com
@File    : 1.Runnable重试机制.py

===================================================================================
知识点讲解：with_retry() 重试机制降低程序错误率
===================================================================================

1. 组件定位：什么是 with_retry 重试机制
   - 在 LangChain 中，针对 Runnable 抛出的异常提供了重试机制——with_retry()
   - 当 Runnable 组件出现异常时，支持针对特定的异常或所有异常，重试特定的次数，
     并且配置每次重试的时间指数增加
   - 返回一个新的 Runnable（RunnableRetry），仍实现 Runnable 协议，可参与 | 管道组合
   - 与回退机制（with_fallbacks）并列，是 LLM 应用容错的两条核心路径之一

2. 为什么需要重试机制
   - LLM 应用强依赖网络与第三方 API，存在不可避免的瞬时故障
   - 常见瞬时错误：网络超时、限流（429）、服务端 5xx、连接重置
   - 这类错误往往重试一次即可成功，重试机制能显著提升链的成功率

3. with_retry() 的核心参数
   - stop_after_attempt: int - 最大尝试次数（含首次），默认 3
   - retry_if_exception_type: tuple - 需要重试的异常类型元组，默认 (Exception,)
   - wait_exponential_jitter: bool - 是否启用指数退避加随机抖动，默认 True
   - exponential_jitter_params: dict - 退避参数（initial/max/jitter 等）

4. stop_after_attempt 的准确语义
   - 该值表示"总尝试次数"，而不是"失败后额外重试的次数"
   - stop_after_attempt=2 → 最多执行 2 次（首次 + 1 次重试）
   - 若全部尝试都失败，抛出最后一次的原始异常

5. 指数退避与随机抖动（Exponential Backoff with Jitter）
   - 指数退避：每次重试的等待时间按指数增长（如 1s、2s、4s）
   - 随机抖动：在等待时间上叠加随机量，避免多客户端同时重试造成雪崩
   - 这是分布式系统中处理限流与瞬时故障的标准实践
   - wait_exponential_jitter=True 时，每次重试时间指数增加并随机再增加 1 秒内的时间

6. 底层实现原理
   - 运行原理非常简单，通过构建一个新的 Runnable 组件
   - 在执行调用类的函数时，循环特定次数，直到组件能正常执行结束即暂停
   - 在每次循环的过程中，休眠特定的时间（指数退避）
   - 若达到最大尝试次数仍失败，则抛出最后一次捕获的异常

7. 本示例的执行过程分析
   - counter 初始为 -1，全局变量在多次调用间持续累加
   - 第 1 次尝试：counter 变为 0，执行 2 / 0 → 抛出 ZeroDivisionError
   - 框架捕获异常，等待退避时间后进行第 2 次尝试
   - 第 2 次尝试：counter 变为 1，执行 2 / 1 = 2.0 → 成功返回
   - 最终输出 2.0，说明重试机制成功挽救了首次失败

8. 数据流转过程
   - 2（int）→ RunnableRetry(RunnableLambda(func), stop_after_attempt=2)
   - → 尝试 1：func(2) 抛出 ZeroDivisionError → 捕获并等待
   - → 尝试 2：func(2) 返回 2.0 → 成功
   - → 输出 2.0

9. 重试的适用边界与注意事项
   - 只应对瞬时错误重试；参数错误、认证失败等确定性错误重试无意义
   - 应通过 retry_if_exception_type 精确限定重试的异常类型
   - 被重试的函数最好是幂等的，否则可能产生重复副作用（如重复下单）
   - 重试会放大延迟与 Token 成本，需在超时预算内合理设置次数

10. 重试与回退的配合使用
   - with_retry()：同一组件重试，解决瞬时故障
   - with_fallbacks()：切换到备用组件，解决持久故障
   - 推荐组合：llm.with_retry(...).with_fallbacks([backup_llm])
   - 语义为：先对主模型重试若干次，全部失败后再切换备用模型

===================================================================================
"""
from langchain_core.runnables import RunnableLambda

# counter: int
#   作用：全局计数器，用于模拟"首次调用失败、重试后成功"的场景
#   初始值 -1 的设计意图：第一次自增后变为 0，使除法运算抛出 ZeroDivisionError
counter = -1


def func(x):
    """模拟一个不稳定的函数：首次调用必然失败，重试后成功

    参数：
        x - 被除数，由 invoke 传入（本例为 2）

    返回：
        float - x 除以 counter 的结果

    异常：
        ZeroDivisionError - 当 counter 为 0 时（即首次调用）抛出

    设计说明：
        通过全局变量 counter 记录调用次数，模拟真实场景中的瞬时故障
        （如首次请求超时、重试后恢复正常）
    """
    global counter

    # 每次调用自增，第 1 次调用后 counter=0，第 2 次调用后 counter=1
    counter += 1
    print(f"当前的值为 {counter=}")

    # 关键点：counter=0 时触发 ZeroDivisionError，counter=1 时正常返回 2.0
    return x / counter


# 构建带重试能力的 Runnable
#
# RunnableLambda(func: Callable) -> RunnableLambda
#   作用：将普通 Python 函数包装为 Runnable 组件
#   参数：func - 被包装的可调用对象
#   返回：RunnableLambda 实例
#
# Runnable.with_retry(
#     retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
#     wait_exponential_jitter: bool = True,
#     exponential_jitter_params: dict = None,
#     stop_after_attempt: int = 3,
# ) -> RunnableRetry
#   作用：为 Runnable 附加自动重试能力，失败时按退避策略重新执行
#   参数：
#     retry_if_exception_type - 触发重试的异常类型元组，默认捕获所有 Exception
#     wait_exponential_jitter - 是否启用指数退避+随机抖动，默认 True
#     exponential_jitter_params - 退避细节参数（如 initial、max、jitter）
#     stop_after_attempt - 最大尝试总次数（含首次），此处为 2
#   返回：RunnableRetry 实例，仍实现 Runnable 协议，可参与 | 管道组合
#   行为说明：
#     - 尝试次数用尽后仍失败，则向上抛出最后一次捕获的异常
#     - 每次重试之间会有退避等待，避免瞬间高频重试
chain = RunnableLambda(func).with_retry(stop_after_attempt=2)

# chain.invoke(input: int) -> float
#   作用：执行带重试的 Runnable
#   参数：input - 2，作为 func 的参数 x
#   返回：最终成功的返回值 2.0
#   实际执行序列：
#     1. 第 1 次尝试：counter 变为 0 → func(2) 执行 2/0 → ZeroDivisionError
#     2. 框架捕获异常，等待退避时间
#     3. 第 2 次尝试：counter 变为 1 → func(2) 执行 2/1 → 返回 2.0
#     4. 成功，返回 2.0（若第 2 次也失败，则抛出 ZeroDivisionError）
resp = chain.invoke(2)

print(resp)

# ==================== 最佳实践与扩展写法 ====================
# 1. 只对特定异常类型重试（推荐做法，避免对确定性错误做无意义重试）
# chain = RunnableLambda(func).with_retry(
#     retry_if_exception_type=(ZeroDivisionError, TimeoutError),
#     stop_after_attempt=3,
# )
#
# 2. 自定义退避参数（控制等待时间上下界）
# chain = RunnableLambda(func).with_retry(
#     stop_after_attempt=5,
#     wait_exponential_jitter=True,
#     exponential_jitter_params={"initial": 1, "max": 10, "jitter": 1},
# )
#
# 3. 关闭退避等待（仅适合本地快速失败的场景，生产不建议）
# chain = RunnableLambda(func).with_retry(
#     stop_after_attempt=3,
#     wait_exponential_jitter=False,
# )
#
# 4. 为 LLM 调用添加重试（最常见的实际用途）
# import dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# dotenv.load_dotenv()
# prompt = ChatPromptTemplate.from_template("{query}")
# llm = ChatOpenAI(model="deepseek-flash").with_retry(stop_after_attempt=3)
# chain = prompt | llm | StrOutputParser()
#
# 5. 重试 + 回退组合（先重试主模型，彻底失败后切换备用模型）
# from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
# llm = (
#     ChatOpenAI(model="deepseek-flash")
#     .with_retry(stop_after_attempt=3)
#     .with_fallbacks([QianfanChatEndpoint()])
# )
#
# 6. 对整条链而非单个组件重试（注意会重复执行链上全部步骤）
# chain = (prompt | llm | StrOutputParser()).with_retry(stop_after_attempt=2)
#
# 7. 注意事项
#   - 被重试函数应尽量幂等，避免重试产生重复副作用
#   - 重试次数与退避时间之和不应超过上游的超时预算
#   - 重试会增加 Token 消耗与响应延迟，需结合业务 SLA 权衡
