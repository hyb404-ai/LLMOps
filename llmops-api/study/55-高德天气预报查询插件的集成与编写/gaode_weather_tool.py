#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 15:00
@Author  : thezehui@gmail.com
@File    : gaode_weather_tool.py

===================================================================================
知识点讲解：高德天气预报查询工具的集成与编写
===================================================================================

1. 高德开放平台天气 API
   - 提供国内城市的天气预报查询服务
   - 需要注册账号并申请 API Key
   - API 文档：https://lbs.amap.com/api/webservice/guide/api/weatherinfo
   - 提供实名认证后免费调用额度

2. 天气查询的两步流程（架构设计）
   - 第一步：通过城市名查询行政区域编码（adcode）
     * 接口：/v3/config/district
   - 第二步：通过 adcode 查询天气预报信息
     * 接口：/v3/weather/weatherInfo
   - 分离查询的原因：支持多种城市名称输入方式（简称、全称、拼音等）

3. API 接口说明
   - 行政区域查询：/v3/config/district
     参数：keywords（城市名）、subdistrict（下级行政区数量）、key（API密钥）
   - 天气预报查询：/v3/weather/weatherInfo
     参数：city（adcode）、extensions（all=未来预报/base=实况）、key（API密钥）

4. 工具设计要点
   - 使用 BaseTool 继承方式，适合复杂的外部 API 调用
   - 参数验证：使用 Pydantic 模型定义城市名称参数
   - 错误处理：捕获网络异常、API 错误等
   - 返回格式：返回 JSON 字符串，便于 LLM 理解

5. 环境变量配置
   - GAODE_API_KEY：高德开放平台的 API 密钥
   - 使用 dotenv.load_dotenv() 加载 .env 文件
   - 避免在代码中硬编码密钥

6. HTTP 请求处理
   - 使用 requests.session() 管理连接
   - raise_for_status() 检查 HTTP 状态码
   - 设置 Content-Type 为 application/json

7. 返回数据处理
   - 高德 API 返回 JSON 格式数据
   - 使用 json.dumps() 转换为字符串
   - 保留完整的天气信息供 LLM 提取

8. 工具集成到 Agent
   - 工具可以单独调用测试
   - 绑定到 LLM 后，LLM 会根据用户问题自动调用
   - 工具描述要清晰，帮助 LLM 判断何时使用

9. API 申请流程（实操步骤）
   - 访问高德开放平台：https://lbs.amap.com/
   - 注册账号并完成实名认证
   - 创建应用，选择 Web 服务类型
   - 获取 API Key 并配置到环境变量

10. 最佳实践
    - API 密钥使用环境变量管理
    - 添加详细的错误处理和日志
    - 返回结构化的 JSON 数据
    - 工具描述要明确使用场景
    - 考虑添加缓存机制（相同城市短时间内无需重复查询）

===================================================================================
"""
import json
import os
from typing import Any, Type

import dotenv
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# 从 .env 文件加载环境变量
# 确保 GAODE_API_KEY 已配置
dotenv.load_dotenv()


# 定义高德天气工具的参数模式
# 使用 Pydantic 模型确保参数验证和类型安全
class GaodeWeatherArgsSchema(BaseModel):
    # city 参数：目标城市名称
    # Field 的 description 会被 LLM 用于理解参数含义
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


# 高德天气查询工具类
# 继承 BaseTool，实现复杂的外部 API 调用逻辑
class GaodeWeatherTool(BaseTool):
    """根据传入的城市名查询天气"""

    # 工具名称，LLM 通过这个名称识别工具
    name = "gaode_weather"

    # 工具描述，帮助 LLM 理解何时使用该工具
    description = "当你想查询天气或者与天气相关的问题时可以使用的工具"

    # 参数模式，定义工具接收的参数结构
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """根据传入的城市名称运行调用api获取城市对应的天气预报信息"""
        try:
            # 1. 从环境变量获取高德 API 密钥
            # 如果未配置密钥，返回错误提示
            gaode_api_key = os.getenv("GAODE_API_KEY")
            if not gaode_api_key:
                return f"高德开放平台API未配置"

            # 2. 从参数中获取城市名称
            # kwargs 包含经过 Pydantic 验证的参数
            city = kwargs.get("city", "")
            api_domain = "https://restapi.amap.com/v3"
            session = requests.session()

            # 3. 第一步：查询行政区域编码（adcode）
            # 高德天气 API 需要使用 adcode 而非城市名称
            # 参数说明：
            #   - keywords：城市名称
            #   - subdistrict：下级行政区数量（0表示不返回下级）
            city_response = session.request(
                method="GET",
                url=f"{api_domain}/config/district?key={gaode_api_key}&keywords={city}&subdistrict=0",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            # 检查 HTTP 状态码，如果不是 2xx 则抛出异常
            city_response.raise_for_status()
            city_data = city_response.json()

            # 检查 API 响应是否成功
            if city_data.get("info") == "OK":
                # 提取第一个匹配城市的行政区域编码
                ad_code = city_data["districts"][0]["adcode"]

                # 4. 第二步：根据 adcode 查询天气预报
                # 参数说明：
                #   - city：行政区域编码
                #   - extensions：all=返回未来3天预报，base=返回当天实况
                weather_response = session.request(
                    method="GET",
                    url=f"{api_domain}/weather/weatherInfo?key={gaode_api_key}&city={ad_code}&extensions=all",
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                # 检查 HTTP 状态码
                weather_response.raise_for_status()
                weather_data = weather_response.json()

                # 检查天气 API 响应是否成功
                if weather_data.get("info") == "OK":
                    # 5. 返回 JSON 格式的天气数据
                    # json.dumps() 将字典转换为 JSON 字符串
                    return json.dumps(weather_data)

            # 如果任何步骤失败，返回错误提示
            return f"获取{city}天气预报信息失败"
        except Exception as e:
            # 捕获所有异常（网络错误、JSON 解析错误等）
            # 返回友好的错误信息
            return f"获取{kwargs.get('city', '')}天气预报信息失败"


# 实例化高德天气工具
gaode_weather = GaodeWeatherTool()

# 测试工具调用
# invoke() 方法接收字典格式的参数
print(gaode_weather.invoke({"city": "深圳"}))
