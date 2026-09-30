#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 16:53
@Author  : thezehui@gmail.com
@File    : google_serp_tool.py

===================================================================================
知识点讲解：Google Serper API 实时搜索工具集成
===================================================================================

1. Google Serper API 简介
   - 低成本的 Google 搜索 API 服务，提供实时搜索结果
   - 需要注册账号并获取 API Key，API 文档：https://serper.dev/
   - 比 Google 官方 API 更快、更便宜、更易用

2. Serper 与 Google 官方 API 的对比
   - Serper 在后台用浏览器模拟真实页面运行，完全模仿人类操作，确保获得用户真正看到的内容
   - 课程实测：Serper 比谷歌官方检索服务内容更全面、响应速度更快
   - 谷歌官方 API 访问速度慢、体验感不佳，因此本课的实时搜索选用第三方 Serper

3. Serper 与 SerpAPI 的区别
   - 两个产品名字和功能非常接近，容易混淆
   - Serper 官网：https://serper.dev/；SerpAPI 官网：https://serpapi.com/
   - 两者 LangChain 都有封装，使用时需注意区分

4. LangChain 内置的 GoogleSerperRun
   - langchain_community.tools 提供的预构建工具，封装了 Serper API 的调用逻辑
   - 无需自己实现 HTTP 请求和数据解析

5. GoogleSerperRun 与 GoogleSerperResults 的区别
   - GoogleSerperRun：返回纯文本搜索结果摘要
   - GoogleSerperResults：返回包含网页链接等元数据的结构化结果

6. GoogleSerperAPIWrapper
   - 底层 API 调用封装类，处理 HTTP 请求、认证、错误处理
   - 可单独使用，也可配合 GoogleSerperRun

7. 工具配置要点（二次封装）
   - name：工具名称，LLM 通过名称识别工具
   - description：详细描述工具用途和场景；内置工具描述是英文，需改成中文
   - args_schema：定义查询参数的 Pydantic 模型；内置工具未加参数说明，需自定义
   - api_wrapper：传入 GoogleSerperAPIWrapper 实例

8. 环境变量配置
   - SERPER_API_KEY：Serper API 的密钥
   - 用 .env 文件管理，避免硬编码；GoogleSerperAPIWrapper 会自动读取

9. API 申请流程
   - 访问 https://serper.dev/，用邮箱/Github/Google 账号注册登录
   - 控制台创建 API Key，配置到 .env：SERPER_API_KEY=你的密钥

10. 工具描述的重要性
   - description 是 LLM 理解工具的关键，应明确说明：做什么、何时用、传什么
   - 好的描述能显著提高 LLM 的工具选择准确性

11. 参数模式设计
   - query：搜索查询字符串；Field 的 description 要清晰明确
   - 单参数工具适合简单搜索场景

12. 使用场景
   - 实时信息查询（新闻、股票、天气等）、事实核查
   - 补充 LLM 知识截止日期后的信息、多步推理中的信息检索

13. 与 DuckDuckGo 的对比
   - GoogleSerperRun：需要 API Key，搜索质量更高
   - DuckDuckGoSearchRun：免费，但可能受网络限制；生产环境推荐 GoogleSerperRun

14. 最佳实践
   - 在工具描述中明确使用场景（如"当你需要回答有关时事的问题时"）
   - 合理设置 API 调用限制防止滥用；将搜索结果交给 LLM 二次处理；考虑添加结果数量限制参数

15. 本文件定位与完整示例（占位说明）
   - 本文件为示例占位文件，实际的工具调用可参考：
     57-ChatModel使用函数调用的技巧与流程/1.GPT模型绑定函数.py
     62-基于ReACT架构的Agent智能体设计与实现/1.ReAct智能体示例.py
   - 课程给出的二次封装完整示例（将描述与参数说明改为中文）：
       from langchain_community.tools import GoogleSerperRun
       from langchain_community.utilities import GoogleSerperAPIWrapper
       from pydantic import BaseModel, Field

       class GoogleSerperArgsSchema(BaseModel):
           query: str = Field(description="执行谷歌搜索的查询语句")

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
       result = google_serper.invoke({"query": "LangChain 最新版本"})
       print(result)
   - 典型输出示例：查询"马拉松的世界记录是多少?" 返回
     "2004年1月1日……肯尼亚人基普图姆在芝加哥马拉松以2小时00分35秒打破基普乔格保持的世界纪录。"

===================================================================================
"""
