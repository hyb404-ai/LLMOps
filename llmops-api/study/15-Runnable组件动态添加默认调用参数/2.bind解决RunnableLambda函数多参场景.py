#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 11:48
@Author  : thezehui@gmail.com
@File    : 2.bind解决RunnableLambda函数多参场景.py

===================================================================================
知识点讲解：用 bind() 解决 RunnableLambda 多参数调用难题
===================================================================================

1. RunnableLambda 的单参数限制
   - RunnableLambda 将普通 Python 函数包装为 Runnable 组件
   - Runnable 协议规定 invoke(input) 只接收一个位置参数
   - 因此被包装的函数最多只能有一个必填位置参数，多参函数直接调用会报错

2. 为什么会出现多参数需求：一个反面示例
   - 工具函数天然需要多个参数，例如 get_weather(location, unit)
   - 直觉上会尝试 get_weather_runnable.invoke({"location": "广州", "unit": "摄氏度"})
   - 但 invoke 只接收一个参数，整个字典会被当作唯一值塞进第一个位置参数 location：
       location: {'location': '广州', 'unit': '摄氏度'}
       unit: None
   - 结果 unit 变成 None，参数全部错位 —— 这正是多参场景的痛点

3. 设计动机：分离运行时数据与固定配置
   - 工具函数天然需要多个参数（如 get_weather(location, unit, name)）
   - 其中部分是「运行时数据」（location 来自上游链），部分是「固定配置」（unit、name）
   - 需要一种机制把配置型参数与数据型参数分离

4. bind() 的解决思路
   - 用 bind(**kwargs) 提前固定除主输入外的所有参数
   - 剩下唯一的位置参数由 invoke(input) 传入
   - 本质上是函数式编程中的偏函数应用（Partial Application）

5. 执行时的参数合并机制
   - RunnableLambda(get_weather).bind(unit="摄氏度", name="慕小课")
   - invoke("广州") 时，框架实际执行：
       get_weather("广州", unit="摄氏度", name="慕小课")
   - 位置参数来自 invoke，关键字参数来自 bind

6. 三种多参处理方案对比
   - bind()：适合参数值固定不变，写法最简洁（参与 Runnable 生态）
   - functools.partial()：Python 原生偏函数，效果等价但不参与 Runnable 生态
   - 字典入参 + lambda 解包：适合所有参数都动态变化
       RunnableLambda(lambda d: get_weather(d["location"], d["unit"], d["name"]))

7. 与第 1 节 llm.bind() 的本质统一
   - 两者都是 Runnable.bind()，同一个方法
   - llm.bind(model=...) 绑定的是 API 请求参数
   - RunnableLambda(func).bind(...) 绑定的是 Python 函数的关键字参数
   - 体现了 Runnable 协议的一致性设计

8. 数据流转过程
   - "广州"（str）
   - → RunnableBinding(RunnableLambda(get_weather), kwargs={"unit": "摄氏度", "name": "慕小课"})
   - → 实际调用 get_weather("广州", unit="摄氏度", name="慕小课")
   - → 返回 str 格式的天气描述

===================================================================================
"""
import random

from langchain_core.runnables import RunnableLambda


def get_weather(location: str, unit: str, name: str) -> str:
    """根据传入的位置+温度单位获取对应的天气信息

    这是一个典型的多参数工具函数，直接用 RunnableLambda 包装后
    无法通过单个 invoke 参数满足全部入参需求。

    参数：
        location: str - 位置名称，由 invoke() 以位置参数形式传入（动态数据）
        unit: str - 温度单位，由 bind() 预先绑定（固定配置）
        name: str - 调用者名称，由 bind() 预先绑定（固定配置）

    返回：
        str - 拼接后的天气描述文本

    说明：
        函数内部的 print 用于验证三个参数是否都被正确接收
    """
    print("location:", location)
    print("unit:", unit)
    print("name:", name)

    # random.randint(a: int, b: int) -> int
    #   作用：生成 [a, b] 闭区间内的随机整数，此处模拟温度值
    #   参数：a - 下界，b - 上界
    #   返回：随机整数
    return f"{location}天气为{random.randint(24, 40)}{unit}"


# 关键步骤：包装函数并绑定固定参数
#
# RunnableLambda(func: Callable) -> RunnableLambda
#   作用：将普通 Python 函数（或 lambda）包装为符合 Runnable 协议的组件
#   参数：func - 被包装的可调用对象
#   返回：RunnableLambda 实例，支持 invoke/batch/stream 及 | 管道组合
#   限制：invoke 只能传入一个位置参数，因此多参函数需配合 bind 使用
#
# Runnable.bind(**kwargs) -> RunnableBinding
#   作用：为 Runnable 绑定固定的关键字参数（偏函数应用）
#   参数：**kwargs - 要固定的参数键值对，此处为 unit 与 name
#         键名必须与被包装函数的参数名完全一致，否则会抛 TypeError
#   返回：RunnableBinding 实例，内部保存原 Runnable 与绑定的 kwargs
#   执行效果：invoke(x) 等价于 get_weather(x, unit="摄氏度", name="慕小课")
get_weather_runnable = RunnableLambda(get_weather).bind(unit="摄氏度", name="慕小课")

# get_weather_runnable.invoke(input: str) -> str
#   作用：执行被包装并绑定参数的函数
#   参数：input - "广州"，作为 get_weather 的第一个位置参数 location
#   返回：函数的返回值（天气描述字符串）
#   参数合并过程：
#     位置参数：location = "广州"（来自 invoke）
#     关键字参数：unit = "摄氏度", name = "慕小课"（来自 bind）
#     最终调用：get_weather("广州", unit="摄氏度", name="慕小课")
resp = get_weather_runnable.invoke("广州")

print(resp)

# ==================== 最佳实践与其他调用方式 ====================
# 1. 使用 functools.partial（Python 原生偏函数，效果等价）
# from functools import partial
# get_weather_runnable = RunnableLambda(
#     partial(get_weather, unit="摄氏度", name="慕小课")
# )
# resp = get_weather_runnable.invoke("广州")
#
# 2. 字典入参 + lambda 解包（适合所有参数都动态变化的场景）
# get_weather_runnable = RunnableLambda(
#     lambda data: get_weather(data["location"], data["unit"], data["name"])
# )
# resp = get_weather_runnable.invoke({
#     "location": "广州",
#     "unit": "摄氏度",
#     "name": "慕小课",
# })
#
# 3. 在 LCEL 管道中组合使用（体现 bind 后仍是标准 Runnable）
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# chain = (
#     get_weather_runnable                                    # str → str
#     | (lambda weather: {"weather": weather})                # str → dict
#     | ChatPromptTemplate.from_template("请用一句话点评天气：{weather}")
#     | ChatOpenAI(model="deepseek-flash")
#     | StrOutputParser()
# )
# print(chain.invoke("广州"))
#
# 4. 批量执行（bind 的参数对所有输入生效）
# resp_list = get_weather_runnable.batch(["广州", "北京", "上海"])
# print(resp_list)
#
# 5. 覆盖已绑定的参数（链式 bind，后者生效）
# celsius_runnable = RunnableLambda(get_weather).bind(unit="摄氏度", name="慕小课")
# fahrenheit_runnable = celsius_runnable.bind(unit="华氏度")   # name 保留，unit 被覆盖
