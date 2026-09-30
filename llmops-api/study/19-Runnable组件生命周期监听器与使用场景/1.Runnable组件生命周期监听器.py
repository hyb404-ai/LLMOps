#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/5 22:41
@Author  : thezehui@gmail.com
@File    : 1.Runnable组件生命周期监听器.py

===================================================================================
知识点讲解：with_listeners() 生命周期监听器与使用场景
===================================================================================

1. 什么是生命周期监听器（Lifecycle Listener）
   - 在 Runnable 执行的关键节点挂载回调函数，实现执行过程的观测与干预
   - 属于面向切面编程（AOP）思想：不侵入业务逻辑即可插入横切关注点
   - 三个核心钩子：on_start（开始前）、on_end（成功结束后）、on_error（异常时）
   - with_listeners 是 Callbacks 机制的简约替代，方法更简洁、统一

2. 三个钩子的触发时机与互斥关系
   - on_start：组件即将执行前触发，此时 run_obj.outputs 为空
   - on_end：组件执行成功后触发，run_obj.outputs 已填充
   - on_error：组件执行抛异常时触发，run_obj.error 记录错误
   - 关键点：on_end 与 on_error 互斥，一次执行只触发其中之一

3. 回调函数的签名规范
   - 必须接收两个参数：(run_obj: Run, config: RunnableConfig)
   - 返回值被忽略，通常声明为 None
   - 回调自身抛异常会影响主流程，内部应做异常保护

4. Run 对象承载的关键信息
   - id：UUID，本次运行唯一标识，可串联分布式链路追踪
   - name / run_type：组件名称与运行类型（chain / llm / tool / retriever…）
   - start_time / end_time：起止时间，用于计算耗时
   - inputs / outputs：输入与输出数据；error：异常信息（仅 on_error 有值）
   - tags / metadata：附加标签与元数据

5. RunnableConfig 承载的运行时配置
   - configurable：运行时配置项（本例传入了自定义 name）
   - callbacks / tags / metadata / run_name / max_concurrency 等
   - 监听器可读取 configurable 中的业务上下文（user_id、trace_id 等）

6. 典型输出示例与观察点（课程示例）
   - on_start 阶段 run_obj.inputs 为 {'input': 2}，config 含 {'configurable': {'name': '慕小课'}}
   - on_end 阶段可读取完整 Run：id=UUID(...)、name='RunnableLambda'、run_type='chain'、inputs={'input':2}、outputs={'output': None}
   - 观察点：on_start 时 outputs 为空，on_end 时 outputs 才有值，印证钩子触发时机

7. 典型应用场景
   - 性能监控：用 end_time - start_time 统计各环节耗时
   - 日志审计：记录每次调用输入输出，满足合规与排障
   - 成本统计：on_end 累计 Token 消耗与费用
   - 异常告警：on_error 上报监控平台
   - 链路追踪：把 run_obj.id 写入 APM 系统

8. 与 Callbacks 机制的关系
   - with_listeners 是 Callbacks 的轻量语法糖，只暴露三个粗粒度钩子
   - 完整 BaseCallbackHandler 支持更细粒度事件（on_llm_new_token 等）
   - 简单监控用 with_listeners，复杂需求实现自定义 CallbackHandler

9. 底层实现原理（源码要点）
   - with_listeners() 将 on_start / on_end / on_error 合并进 config 的 callbacks，本质用 CallbackHandler 逻辑实现监听
   - 具体：创建一个 RootListenersTracer 对象包装三个回调，通过 RunnableBinding 的 config_factories 注入执行流程
   - _merge_configs 会把 config_factories 产出的回调与基础 config 合并

10. 数据流转与执行时序
    - chain.invoke(2, config={...})
    - → 触发 on_start(run_obj, config)，打印运行元信息
    - → 执行 lambda x: time.sleep(x)，阻塞 2 秒，返回 None
    - → 触发 on_end(run_obj, config)，此时可读取 outputs 与 end_time
    - → 若抛异常则改为触发 on_error，不会触发 on_end

===================================================================================
"""
import time

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run


def on_start(run_obj: Run, config: RunnableConfig) -> None:
    """组件开始执行前的回调钩子

    触发时机：
        Runnable 即将执行业务逻辑之前，此时尚未产生输出

    参数：
        run_obj: Run - 本次运行的追踪对象，可读取的关键字段：
            - id: UUID - 运行唯一标识，可用于链路追踪
            - name: str - 组件名称（此处为 RunnableLambda）
            - run_type: str - 运行类型（此处为 chain）
            - start_time: datetime - 开始时间戳
            - inputs: dict - 输入数据（此处为 {"input": 2}）
            - outputs: dict - 此阶段为空或 None（尚未执行完成）
        config: RunnableConfig - 运行时配置字典，可读取：
            - configurable: dict - 自定义配置（此处含 {"name": "慕小课"}）
            - tags / metadata / callbacks 等

    返回：
        None - 返回值被框架忽略

    典型用途：
        记录请求开始日志、启动计时器、初始化追踪 span、校验输入合法性
    """
    print("on_start")
    print("run_obj:", run_obj)
    print("config:", config)
    print("============")


def on_end(run_obj: Run, config: RunnableConfig) -> None:
    """组件执行成功后的回调钩子

    触发时机：
        Runnable 执行完成且未抛出异常，与 on_error 互斥

    参数：
        run_obj: Run - 本次运行的追踪对象，此阶段新增可用字段：
            - end_time: datetime - 结束时间戳，配合 start_time 可算出耗时
            - outputs: dict - 输出数据（此处 lambda 返回 None）
        config: RunnableConfig - 与 on_start 收到的是同一份配置

    返回：
        None

    典型用途：
        统计耗时（end_time - start_time）、记录成功日志、
        累计 Token 消耗与成本、上报监控指标、结束追踪 span
    """
    print("on_end")
    print("run_obj:", run_obj)
    print("config:", config)
    print("============")


def on_error(run_obj: Run, config: RunnableConfig) -> None:
    """组件执行抛出异常时的回调钩子

    触发时机：
        Runnable 执行过程中抛出异常，与 on_end 互斥

    参数：
        run_obj: Run - 本次运行的追踪对象，此阶段关键字段：
            - error: str - 异常的字符串描述，用于定位问题原因
            - end_time: datetime - 异常发生的时间戳
            - inputs: dict - 导致失败的输入，便于复现问题
        config: RunnableConfig - 运行时配置

    返回：
        None

    注意：
        该回调不会阻止异常向上传播，异常仍会抛给调用方
        若需要吞掉异常，应使用 with_fallbacks 而非监听器

    典型用途：
        异常日志记录、上报监控告警平台、失败率统计、故障现场快照保存
    """
    print("on_error")
    print("run_obj:", run_obj)
    print("config:", config)
    print("============")


# 1.创建RunnableLambda与链
#
# RunnableLambda(func: Callable) -> RunnableLambda
#   作用：将 lambda 函数包装为 Runnable 组件
#   参数：func - lambda x: time.sleep(x)，接收秒数并阻塞等待
#   返回：RunnableLambda 实例
#   说明：time.sleep 返回 None，因此该组件的输出为 None
#         这里用 sleep 模拟耗时操作，便于在 on_end 中观察时间差
#
# Runnable.with_listeners(
#     on_start: Callable[[Run, RunnableConfig], None] = None,
#     on_end: Callable[[Run, RunnableConfig], None] = None,
#     on_error: Callable[[Run, RunnableConfig], None] = None,
# ) -> RunnableBinding
#   作用：为 Runnable 绑定生命周期回调，返回带监听能力的新组件
#   参数：
#     on_start - 执行前回调，签名为 (Run, RunnableConfig) -> None
#     on_end - 执行成功后回调，与 on_error 互斥
#     on_error - 执行异常时回调，与 on_end 互斥
#     三个参数均为可选，可只绑定需要的钩子
#   返回：RunnableBinding 实例，仍实现 Runnable 协议，可参与 | 管道组合
#   特性：不修改原 Runnable，遵循不可变设计
runnable = RunnableLambda(lambda x: time.sleep(x)).with_listeners(
    on_start=on_start,
    on_end=on_end,
    on_error=on_error,
)

# 本例中链仅由单个组件构成，便于清晰观察监听器的触发时序
chain = runnable

# 2.调用并执行链
# chain.invoke(input: int, config: dict = None) -> None
#   作用：执行带监听器的 Runnable
#   参数：
#     input - 2，作为 lambda 的参数传给 time.sleep（阻塞 2 秒）
#     config - {"configurable": {"name": "慕小课"}}
#              自定义配置会完整传递给三个回调函数，可用于携带业务上下文
#   返回：None（因为 time.sleep 返回 None）
#   执行时序：
#     1. 触发 on_start(run_obj, config) → 打印开始信息（outputs 为空）
#     2. 执行 time.sleep(2) → 阻塞 2 秒
#     3. 触发 on_end(run_obj, config) → 打印结束信息（含 end_time）
#     4. 返回 None
#   观察要点：对比 run_obj 中 start_time 与 end_time 的差值，应约为 2 秒
chain.invoke(2, config={"configurable": {"name": "慕小课"}})

# ==================== 最佳实践与扩展写法 ====================
# 1. 仅绑定需要的钩子（三个参数都是可选的）
# runnable = RunnableLambda(lambda x: x * 2).with_listeners(on_end=on_end)
#
# 2. 性能监控：统计组件执行耗时
# def log_duration(run_obj: Run, config: RunnableConfig) -> None:
#     duration = (run_obj.end_time - run_obj.start_time).total_seconds()
#     print(f"[{run_obj.name}] 耗时 {duration:.3f}s")
# runnable = RunnableLambda(lambda x: time.sleep(x)).with_listeners(on_end=log_duration)
#
# 3. 异常告警：将失败上报监控平台
# def alert_on_error(run_obj: Run, config: RunnableConfig) -> None:
#     print(f"[ALERT] run_id={run_obj.id} error={run_obj.error} inputs={run_obj.inputs}")
#     # 实际生产中此处调用监控 SDK，如 Sentry / Prometheus / 内部告警平台
# runnable = RunnableLambda(lambda x: 1 / 0).with_listeners(on_error=alert_on_error)
# # 注意：监听器不会吞掉异常，异常仍会抛给调用方
#
# 4. 为 LLM 调用添加监听（统计 Token 与耗时的真实场景）
# import dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# dotenv.load_dotenv()
# def log_llm_usage(run_obj: Run, config: RunnableConfig) -> None:
#     print("LLM 输出:", run_obj.outputs)
#     # run_obj.outputs 中通常含 llm_output.token_usage，可用于成本核算
# llm = ChatOpenAI(model="deepseek-flash").with_listeners(on_end=log_llm_usage)
# chain = ChatPromptTemplate.from_template("{query}") | llm | StrOutputParser()
# chain.invoke({"query": "你好"})
#
# 5. 从 config 中读取业务上下文（多租户场景常用）
# def log_with_context(run_obj: Run, config: RunnableConfig) -> None:
#     user = config.get("configurable", {}).get("name", "unknown")
#     print(f"用户 {user} 执行了 {run_obj.name}")
# chain.invoke(2, config={"configurable": {"name": "慕小课"}})
#
# 6. 为链中的多个组件分别绑定监听器（精细化定位瓶颈环节）
# chain = (
#     prompt.with_listeners(on_end=log_duration)
#     | llm.with_listeners(on_end=log_duration)
#     | StrOutputParser().with_listeners(on_end=log_duration)
# )
#
# 7. 需要更细粒度事件时改用 CallbackHandler
# from langchain_core.callbacks import BaseCallbackHandler
# class TokenCounter(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         print("新 token:", token)
# chain.invoke({"query": "你好"}, config={"callbacks": [TokenCounter()]})
#
# 8. 注意事项
#   - 回调函数内部应做异常保护，避免监控逻辑影响主业务流程
#   - 回调是同步执行的，耗时操作（如网络上报）建议异步化或批量提交
#   - 异步链请使用 with_alisteners 绑定 async 回调函数
