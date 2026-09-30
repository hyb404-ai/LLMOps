#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/13 11:00
@Author  : thezehui@gmail.com
@File    : 1.多模态LLM调用工具.py

===================================================================================
知识点讲解：多模态 LLM 执行函数调用的技巧
===================================================================================

1. 多模态 LLM 简介
   - 能同时处理文本与图像（部分模型支持音频、视频），代表：GPT-4o、GPT-4V、Claude 3、Gemini
   - 可以「看懂」图片内容并基于此推理，GPT-4o 即属于输入多模态模型

2. 多模态输入的本质：只是把 content 换成列表
   - 普通文本消息的 content 是字符串；多模态消息的 content 改为一个内容块列表
   - 用法和普通的提示模板没有差异，只是把原本的字符串替换成了「列表 + 字典」的格式
   - 在 LangChain 中只需按模型对应的消息结构创建 LangChain 消息 / 提示模板即可

3. 多模态 + 函数调用的组合能力
   - LLM 先「理解」图片内容，再从图片提取关键信息作为工具参数，最后调用工具获取更多信息
   - 本示例：识别图片中的城市 → 调用天气工具查询天气，形成「识图 + 查询」闭环

4. 多模态提示词的构造方式
   - content 是内容块列表，每个块是一个含 type 字段的字典
   - type="text"：文本内容块；type="image_url"：图片内容块
   - 示例：
       ChatPromptTemplate.from_messages([
           ("human", [
               {"type": "text", "text": "描述这张图片"},
               {"type": "image_url", "image_url": {"url": "{image_url}"}}
           ])
       ])

5. image_url 的两种形式
   - 网络 URL：{"url": "https://example.com/image.jpg"}（推荐，省 token）
   - Base64 编码：{"url": "data:image/jpeg;base64,{base64_string}"}
   - 可选参数 detail：控制图片分析精度（low/high/auto）

6. 本示例的完整流程
   - 步骤1：用户传入图片 URL
   - 步骤2：多模态 LLM 分析图片，识别出城市（如广州）
   - 步骤3：LLM 生成 gaode_weather 工具调用，参数 city="广州"
   - 步骤4：执行天气工具，获取天气 JSON 数据
   - 步骤5：将天气数据传给第二个提示词
   - 步骤6：LLM 将 JSON 整理为用户友好的文本

7. tool_choice 参数的作用
   - bind_tools(tools=[...], tool_choice="gaode_weather")
   - 强制 LLM 必须调用指定工具，确保图片识别后一定会查询天气
   - 避免 LLM 只描述图片而不调用工具

8. 两阶段链与嵌套字典
   - 第一阶段：图片识别 + 工具调用（获取原始数据）
   - 第二阶段：数据整理 + 格式化输出（提升可读性），分离关注点
   - 链用嵌套字典 {"weather": (子链)} 组织，子链为
     {"image_url": RunnablePassthrough()} | prompt | llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | GaodeWeatherTool()
   - image_url 经 RunnablePassthrough 直接透传进 prompt；嵌套字典执行子链后将结果赋给 weather 键，输出 {"weather": 子链结果} 作为下一提示词输入

9. 典型输出示例与观察点
   - 传入广州图片后，第二阶段整理出用户友好的天气预报：
       ### 广州市天气预报
       **发布时间：2024年7月11日 12:00**
       #### 2024年7月11日 (星期四)
       - 白天天气：大雨  夜间天气：大雨
       - 白天温度：33°C  夜间温度：25°C
       - 风力等级：1-3级
       请注意天气变化，做好防雨准备。
   - 观察点：最终输出是 LLM 把工具返回的 JSON 二次整理后的纯文本，而非原始 JSON

10. 应用场景
   - 图片识别 + 信息查询（本示例）、图片 OCR + 数据库查询
   - 商品图片识别 + 价格查询、医疗影像分析 + 病历检索、场景识别 + 推荐系统

11. 注意事项与最佳实践
   - 图片 URL 必须可公开访问；图片大小 / 分辨率影响成本与延迟
   - 不是所有模型都支持「多模态 + 函数调用」；Base64 编码会显著增加 token 消耗
   - 用 tool_choice 确保工具被调用；URL 优于 Base64（省 token）
   - 提示词明确说明期望的工具调用；分阶段处理便于调试；建议添加识别失败兜底

===================================================================================
"""
import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 高德天气工具的参数模式
class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


# 高德天气查询工具
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


# 1.构建prompt
# 多模态提示词模板
# 关键点：content 是内容块列表，而不是简单字符串
#   - {"type": "text", ...}：文本内容块，说明任务
#   - {"type": "image_url", ...}：图片内容块，传入图片
#   - {image_url} 是模板变量，运行时会被实际 URL 替换
prompt = ChatPromptTemplate.from_messages([
    ("human", [
        {"type": "text", "text": "请获取下上传图片所在城市的天气预报。"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ])
])

# 第二阶段的提示词：将天气 JSON 数据整理为用户友好的输出
# 使用 <weather> 标签包裹数据，帮助 LLM 识别数据边界
weather_prompt = ChatPromptTemplate.from_template("""请整理下传递的城市的天气预报信息，并以用户友好的方式输出。

<weather>
{weather}
</weather>""")

# 2.构建LLM并绑定工具
# 注意：模型必须支持多模态（图片理解）能力
llm = ChatOpenAI(model="deepseek-v4-pro")
# bind_tools() 参数说明：
#   - tools=[GaodeWeatherTool()]：绑定天气工具
#   - tool_choice="gaode_weather"：强制调用该工具
#     确保 LLM 识别图片后一定会查询天气，而不只是描述图片
llm_with_tools = llm.bind_tools(tools=[GaodeWeatherTool()], tool_choice="gaode_weather")

# 3.创建链应用并执行
# 链的结构分析（两阶段处理）：
#
# 第一阶段（内层子链）：图片识别 + 工具调用
#   {"image_url": RunnablePassthrough()}  # 将输入 URL 赋值给 image_url
#   | prompt                              # 生成多模态提示词
#   | llm_with_tools                      # LLM 识别图片并生成工具调用
#   | (lambda msg: msg.tool_calls[0]["args"])  # 提取工具参数（含城市名）
#   | GaodeWeatherTool()                  # 执行天气查询，返回 JSON 字符串
#
# 第二阶段（外层）：数据整理
#   {"weather": 第一阶段结果}             # 将天气数据赋值给 weather 键
#   | weather_prompt                      # 生成整理提示词
#   | llm                                 # LLM 整理数据（注意：使用未绑定工具的 llm）
#   | StrOutputParser()                   # 提取文本内容
chain = (
        {
            "weather": (
                    {"image_url": RunnablePassthrough()}
                    | prompt
                    | llm_with_tools |
                    (lambda msg: msg.tool_calls[0]["args"])
                    | GaodeWeatherTool()
            )
        }
        | weather_prompt | llm | StrOutputParser()
)

# 执行链
# 输入：图片 URL（图片内容是广州的某个地标）
# 输出：用户友好的天气预报文本
print(chain.invoke("https://imooc-langchain.shortvar.com/guangzhou.jpg"))
