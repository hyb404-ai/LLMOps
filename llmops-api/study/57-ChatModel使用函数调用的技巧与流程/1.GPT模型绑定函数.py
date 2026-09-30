#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/9 17:53
@Author  : thezehui@gmail.com
@File    : 1.GPT模型绑定函数.py

===================================================================================
知识点讲解：ChatModel 使用函数调用（Function Calling）的技巧与流程
===================================================================================

1. 函数调用（Function Calling）本质
   - LLM 原生支持的能力，识别用户意图并生成结构化工具调用请求
   - LLM 不直接执行函数，只返回"调哪个函数、传什么参数"，实际执行由应用程序完成
   - 解决 LLM 两大缺陷：输出非结构化、知识有截止日期

2. 函数调用的设计初衷
   - 问题1：响应是非结构化且不稳定的，下游需要 JSON；prompt 强约束不稳定
   - 问题2：知识有截止日期，无法提供最新消息；重训成本高
   - 通过"结构化输出 + 外部工具"同时解决

3. 函数调用的运行架构（5 步循环）
   - 步骤1：LLM 接收用户查询 + 工具定义（tools）
   - 步骤2：LLM 返回 AIMessage，tool_calls 含调用请求
   - 步骤3：应用解析 tool_calls 并执行对应工具
   - 步骤4：将结果包装为 ToolMessage
   - 步骤5：把含 ToolMessage 的完整历史再次发给 LLM，生成最终回答

4. bind_tools() 方法详解
   - 将工具列表绑定到 ChatModel，工具被 convert_to_openai_tool 转为 OpenAI 函数调用格式（JSON Schema）
   - 参数 tools 接受 BaseTool / Pydantic 模型 / 函数 / 字典；tool_choice 控制调用策略：
     * "auto"（默认）LLM 自主决定；"none" 不调用；"any"/"required" 强制至少调一个
     * 传具体工具名则强制调用该工具；源码中 "any" 会被规整为 "required"
   - 返回绑定了工具的新 Runnable 实例

5. 更底层的 bind() 与 convert_to_openai_tool
   - .bind() 可直接透传任意运行时参数（tools、tool_choice、temperature 等），bind_tools 是其便捷封装
   - convert_to_openai_tool(tool) 把 BaseTool / Type[BaseModel] / Callable / dict 统一转换成 OpenAI 函数描述，是 bind_tools 的内部第一步

6. tool_calls 结构详解（关键数据格式）
   - 每个 tool_call 是字典，含 id（调用唯一标识，用于关联请求与结果）、name（工具名）、args（参数字典）
   - 支持并行调用多个工具（一次返回多个 tool_calls）
   - 示例：[{"id": "call_abc123", "name": "gaode_weather", "args": {"city": "广州"}}]

7. ToolMessage 的作用与结构
   - 承载工具执行结果，必须含 tool_call_id 与 AIMessage 的 tool_call id 对应
   - LLM 通过 tool_call_id 关联请求与结果；content 为工具返回结果

8. 消息历史组装顺序（严格顺序）
   - SystemMessage → HumanMessage → AIMessage（含 tool_calls）→ ToolMessage（每个工具一条，含 tool_call_id）→ 最终 LLM 生成可读回答

9. 并行工具调用
   - 现代 LLM 支持一次返回多个 tool_calls，可并行执行独立工具
   - 系统提示可引导"需要多个工具时一次性调用所有工具"

10. tool_dict 设计模式
    - 用字典映射工具名到工具实例，按 tool_call 的 name 快速查找，避免遍历列表

11. 函数调用参数格式（GPT 模型示例）
    - 工具描述用 JSON Schema：{"type":"function","function":{"name":..., "description":..., "parameters":{"type":"object","properties":{...},"required":[...]}}}

12. 使用场景
    - 实时数据问答（天气、新闻、股票）、执行计算、访问外部 API、多步推理、结构化数据提取

13. 最佳实践
    - 工具描述要清晰，帮助 LLM 正确选择；用 tool_dict 管理工具便于查找；妥善处理工具执行失败
    - 注意：上述「绑定工具 + 循环解析 tool_calls + 追加 ToolMessage」的写法已是在用程序搭第一个 Agent；复杂 Agent 建议用 LangChain 传统 Agent 或 LangGraph 封装

===================================================================================
"""
import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# 加载环境变量（需要 GAODE_API_KEY 和 SERPER_API_KEY）
dotenv.load_dotenv()


# 高德天气工具的参数模式
class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


# Google 搜索工具的参数模式
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


# 高德天气查询工具
# 继承 BaseTool 实现外部 API 调用
class GaodeWeatherTool(BaseTool):
    """根据传入的城市名查询天气"""
    name = "gaode_weather"  # 工具名称，LLM 通过此名称调用
    description = "当你想询问天气或与天气相关的问题时的工具。"  # 工具描述
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema  # 参数模式

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """运行工具获取对应城市的天气预报"""
        try:
            # 1.获取高德API秘钥，如果没有则抛出错误
            gaode_api_key = os.getenv("GAODE_API_KEY")
            if not gaode_api_key:
                return f"高德开放平台API秘钥未配置"

            # 2.提取传递的城市名字并查询行政编码
            # 高德天气 API 需要 adcode 而非城市名
            city = kwargs.get("city", "")
            session = requests.session()
            api_domain = "https://restapi.amap.com/v3"
            city_response = session.request(
                method="GET",
                url=f"{api_domain}/config/district?keywords={city}&subdistrict=0&extensions=all&key={gaode_api_key}",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            city_response.raise_for_status()  # 检查 HTTP 状态码
            city_data = city_response.json()

            # 3.提取行政编码调用天气预报查询接口
            if city_data.get("info") == "OK":
                if len(city_data.get("districts")) > 0:
                    # 取第一个匹配的城市的行政区域编码
                    ad_code = city_data["districts"][0]["adcode"]

                    # extensions=all 表示返回未来几天的天气预报
                    weather_response = session.request(
                        method="GET",
                        url=f"{api_domain}/weather/weatherInfo?city={ad_code}&extensions=all&key={gaode_api_key}&output=json",
                        headers={"Content-Type": "application/json; charset=utf-8"},
                    )
                    weather_response.raise_for_status()
                    weather_data = weather_response.json()
                    if weather_data.get("info") == "OK":
                        # 返回 JSON 字符串，便于 LLM 解析
                        return json.dumps(weather_data)

            session.close()  # 关闭 session 释放连接
            return f"获取{kwargs.get('city')}天气预报信息失败"
            # 4.整合天气预报信息并返回
        except Exception as e:
            # 捕获所有异常，返回友好的错误提示
            return f"获取{kwargs.get('city')}天气预报信息失败"


# 1.定义工具列表
# 实例化高德天气工具
gaode_weather = GaodeWeatherTool()

# 实例化 Google 搜索工具
# GoogleSerperRun 是 LangChain 内置的搜索工具
# api_wrapper 参数传入 API 封装类实例
google_serper = GoogleSerperRun(
    name="google_serper",
    description=(
        "一个低成本的谷歌搜索API。"
        "当你需要回答有关时事的问题时，可以调用该工具。"
        "该工具的输入是搜索查询语句。"
    ),
    args_schema=GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper(),
)

# 使用字典管理工具，便于根据工具名称快速查找
# 这是处理 tool_calls 时的常用设计模式
tool_dict = {
    gaode_weather.name: gaode_weather,
    google_serper.name: google_serper,
}
# 提取工具列表，用于绑定到 LLM
tools = [tool for tool in tool_dict.values()]

# 2.创建Prompt
# 系统提示中明确说明工具调用策略：一次性调用所有需要的工具（并行调用）
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是由DeepSeek开发的聊天机器人，可以帮助用户回答问题，必要时刻请调用工具帮助用户解答，如果问题需要多个工具回答，请一次性调用所有工具，不要分步调用"
    ),
    ("human", "{query}"),
])

# 3.创建大语言模型并绑定工具
llm = ChatOpenAI(model="deepseek-v4-pro")
# bind_tools() 将工具转换为 OpenAI 函数调用格式并绑定到模型
# 绑定后，LLM 在响应时可以返回 tool_calls
llm_with_tool = llm.bind_tools(tools=tools)

# 4.创建链应用
# RunnablePassthrough() 将输入字符串传递给 query 键
chain = {"query": RunnablePassthrough()} | prompt | llm_with_tool

# 5.调用链应用，并获取输出响应
# 这个查询同时需要天气工具和搜索工具
query = "上海现在天气怎样，并且请用谷歌搜索工具查询一下2024年巴黎奥运会中国代表团共获得几枚金牌？"
resp = chain.invoke(query)
# tool_calls 是 AIMessage 的属性，包含 LLM 生成的工具调用请求列表
# 每个 tool_call 包含：id（调用ID）、name（工具名）、args（参数）
tool_calls = resp.tool_calls

# 6.判断是工具调用还是正常输出结果
if len(tool_calls) <= 0:
    # 如果没有工具调用，直接输出 LLM 的回答
    print("生成内容: ", resp.content)
else:
    # 7.将历史的系统消息、人类消息、AI消息组合
    # 重建完整的消息历史，这是函数调用的关键步骤
    # prompt.invoke() 生成 SystemMessage 和 HumanMessage
    messages = prompt.invoke(query).to_messages()
    # 追加 AIMessage（包含 tool_calls 信息）
    messages.append(resp)

    # 8.循环遍历所有工具调用信息
    # 支持并行调用多个工具
    for tool_call in tool_calls:
        # 根据工具名称从 tool_dict 中获取工具实例
        tool = tool_dict.get(tool_call.get("name"))  # 获取需要执行的工具
        print("正在执行工具: ", tool.name)
        # 使用 tool_call 中的参数执行工具
        content = tool.invoke(tool_call.get("args"))  # 工具执行的内容/结果
        print("工具返回结果: ", content)
        # 获取 tool_call_id，用于关联请求和结果
        tool_call_id = tool_call.get("id")
        # 将工具结果包装为 ToolMessage 追加到消息历史
        # tool_call_id 必须与 AIMessage 中的 tool_call id 一致
        messages.append(ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
        ))
    # 将完整的消息历史（含工具结果）发送给 LLM
    # LLM 会基于工具结果生成人类可读的最终回答
    # 注意：这里使用原始 llm（未绑定工具），避免再次触发工具调用
    print("输出内容: ", llm.invoke(messages).content)
