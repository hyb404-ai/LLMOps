#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/11 10:01
@Author  : thezehui@gmail.com
@File    : 1.不支持函数调用的模型解决示例.py

===================================================================================
知识点讲解：不支持函数调用的大语言模型解决技巧
===================================================================================

1. 函数调用支持的模型差异
   - 原生支持：GPT-3.5+、GPT-4、Claude、DeepSeek 等
   - 不支持：早期模型、开源小模型、某些定制模型
   - 不支持时 bind_tools() 不会报错，但 LLM 不会生成 tool_calls

2. 替代方案：文本协议 + JSON 解析
   - 核心思路：通过提示词让 LLM 输出结构化 JSON
   - 使用 render_text_description_and_args() 生成工具描述文本
   - JsonOutputParser() 解析 LLM 的 JSON 输出
   - 手动调用工具函数执行

3. render_text_description_and_args() 详解
   - 将工具列表转换为文本描述
   - 包含工具名称、描述、参数说明
   - 注入到系统提示词中
   - 帮助 LLM 理解可用工具

4. 文本协议的 JSON 格式要求
   - name：工具名称字符串
   - arguments：参数字典，键名对应工具参数
   - 提示词中明确指定 JSON 格式和字段名

5. 与原生函数调用的对比
   - 原生方式：bind_tools() + tool_calls 自动解析
   - 文本协议：手动构造提示词 + JSON 解析
   - 文本协议需要更强的指令遵循能力

6. invoke_tool() 工具调用函数
   - 接收 ToolCallRequest（name + arguments）
   - 从 tool_dict 查找对应工具
   - 执行工具并返回结果
   - 统一的工具执行接口

7. RunnablePassthrough.assign() 的妙用
   - 保留输入的所有字段
   - 添加新字段（output）
   - 创建包含工具调用和结果的完整输出

8. 链式处理流程
   步骤1：提示词生成（包含工具描述）
   步骤2：LLM 生成 JSON 响应
   步骤3：JsonOutputParser 解析为字典
   步骤4：invoke_tool 执行工具
   步骤5：组合工具调用信息和结果

9. 适用场景
   - 使用不支持函数调用的开源模型
   - 需要兼容多种模型（统一接口）
   - 对延迟不敏感的场景
   - 工具数量较少的场景

10. 注意事项
    - 需要强指令遵循能力的模型
    - JSON 解析可能失败，需要错误处理
    - 性能略低于原生函数调用
    - 不支持并行工具调用
    - 提示词要清晰明确

11. 最佳实践
    - 提供清晰的工具描述和参数说明
    - 在提示词中给出 JSON 输出示例
    - 使用 temperature=0 提高稳定性
    - 添加 JSON 解析的异常处理
    - 优先使用支持原生函数调用的模型

===================================================================================
"""
import json
import os
from typing import Type, Any, TypedDict, Dict, Optional

import dotenv
import requests
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.tools import BaseTool, render_text_description_and_args
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 定义工具参数模式
class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


# 高德天气工具实现
class GaodeWeatherTool(BaseTool):
    """根据传入的城市名查询天气"""
    name = "gaode_weather"
    description = "当你想询问天气或与天气相关的问题时的工具。"
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """运行工具获取对应城市的天气预报"""
        try:
            # 1.获取高德API秘钥，如果没有则抛出错误
            gaode_api_key = os.getenv("GAODE_API_KEY")
            if not gaode_api_key:
                return f"高德开放平台API秘钥未配置"

            # 2.提取传递的城市名字并查询行政编码
            city = kwargs.get("city", "")
            session = requests.session()
            api_domain = "https://restapi.amap.com/v3"
            city_response = session.request(
                method="GET",
                url=f"{api_domain}/config/district?keywords={city}&subdistrict=0&extensions=all&key={gaode_api_key}",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            city_response.raise_for_status()
            city_data = city_response.json()

            # 3.提取行政编码调用天气预报查询接口
            if city_data.get("info") == "OK":
                if len(city_data.get("districts")) > 0:
                    ad_code = city_data["districts"][0]["adcode"]

                    weather_response = session.request(
                        method="GET",
                        url=f"{api_domain}/weather/weatherInfo?city={ad_code}&extensions=all&key={gaode_api_key}&output=json",
                        headers={"Content-Type": "application/json; charset=utf-8"},
                    )
                    weather_response.raise_for_status()
                    weather_data = weather_response.json()
                    if weather_data.get("info") == "OK":
                        return json.dumps(weather_data)

            session.close()
            return f"获取{kwargs.get('city')}天气预报信息失败"
            # 4.整合天气预报信息并返回
        except Exception as e:
            return f"获取{kwargs.get('city')}天气预报信息失败"


# 定义工具调用请求的类型
# 这是 LLM 输出的 JSON 结构
class ToolCallRequest(TypedDict):
    name: str  # 工具名称
    arguments: Dict[str, Any]  # 工具参数字典


# 1.定义工具列表
gaode_weather = GaodeWeatherTool()
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
# 使用字典管理工具，便于根据名称查找
tool_dict = {
    gaode_weather.name: gaode_weather,
    google_serper.name: google_serper,
}
tools = [tool for tool in tool_dict.values()]


# 工具调用执行函数
# 这是手动执行工具的统一接口，替代原生函数调用的自动执行
def invoke_tool(
        tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None,
) -> str:
    """
    我们可以使用的执行工具调用的函数。

    :param tool_call_request: 一个包含键名和参数的字典，名称必须与现有的工具名称匹配，参数是该工具的参数。
    :param config: 这是LangChain中包含回调、元数据等信息的配置信息。
    :return: 工具执行的结果。
    """
    # 从请求中提取工具名称
    name = tool_call_request["name"]
    # 根据名称从 tool_dict 查找工具实例
    requested_tool = tool_dict.get(name)
    # 调用工具的 invoke() 方法，传入参数
    # config 参数传递 LangChain 的运行时配置
    return requested_tool.invoke(tool_call_request.get("arguments"), config=config)


# 构建系统提示词
# 关键点：
# 1. {rendered_tools} 占位符用于插入工具描述文本
# 2. 明确指定 JSON 输出格式（name + arguments）
# 3. 说明 arguments 的结构（字典，键对应参数名）
system_prompt = """你是一个由DeepSeek开发的聊天机器人，可以访问以下工具。
以下是每个工具的名称和描述：

{rendered_tools}

根据用户输入，返回要使用的工具的名称和输入。
将您的响应作为具有`name`和`arguments`键的JSON块返回。
`arguments`应该是一个字典，其中键对应于参数名称，值对应于请求的值。"""

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
# partial() 提前绑定 rendered_tools
# render_text_description_and_args() 将工具列表转换为文本描述
]).partial(rendered_tools=render_text_description_and_args(tools))

# 创建 LLM，temperature=0 提高输出稳定性
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# 构建链式处理流程
# 步骤1：prompt 生成包含工具描述的提示词
# 步骤2：llm 根据提示词生成 JSON 响应
# 步骤3：JsonOutputParser() 解析 JSON 字符串为字典
# 步骤4：RunnablePassthrough.assign() 执行工具并添加 output 字段
#   - assign() 保留原始字段（name、arguments）
#   - 添加新字段：output=invoke_tool(原始字段)
chain = prompt | llm | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)

# 执行链
# 最终输出包含三个字段：
# - name: 工具名称
# - arguments: 工具参数
# - output: 工具执行结果
print(chain.invoke({"query": "马拉松的世界记录是多少？"}))
