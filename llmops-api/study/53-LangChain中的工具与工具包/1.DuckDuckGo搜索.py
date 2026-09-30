#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/8 11:08
@Author  : thezehui@gmail.com
@File    : 1.DuckDuckGo搜索.py

===================================================================================
知识点讲解：LangChain 工具与工具包
===================================================================================

1. 工具（Tool）的概念
   - 工具是 Agent 可以调用的函数或服务
   - 每个工具有名称、描述、参数模式和执行逻辑
   - 工具是 Agent 与外部世界交互的桥梁
   - 本质：函数调用（Function Calling）的具体实现，LangChain 封装了大量预设工具，开箱即用

2. LangChain 内置工具生态
   - langchain_community.tools 提供了大量预构建工具，涵盖搜索、图像生成、百科、视频信息提取、代码执行、数据分析等
   - DuckDuckGoSearchRun：隐私友好的搜索引擎工具（依赖 pip install -U duckduckgo-search）
   - GoogleSerperRun：Google 搜索 API 工具
   - WikipediaQueryRun：维基百科查询工具
   - 更多工具：计算器、文件读写、API 调用、图像生成等
   - 官方文档：https://imooc-langchain.shortvar.com/docs/integrations/tools/

3. 工具的核心属性（四要素）
   - name：工具的唯一标识符，LLM 通过名称选择工具
   - description：工具的功能描述，帮助 LLM 理解何时使用该工具（是 LLM 选择工具的关键依据，应明确功能、适用场景、输入格式）
   - args：工具的参数模式（JSON Schema），定义输入格式
   - return_direct：是否直接返回工具结果（跳过 LLM 后处理）

4. BaseTool 基类与 Runnable 接口
   - 所有工具都是 BaseTool 的子类
   - 工具也是 Runnable 可运行组件，支持 invoke()、batch()、stream() 等标准方法
   - 可以使用 | 操作符组合到 LCEL 链中
   - 实例化工具后直接调用 invoke 即可，无需手动编排

5. DuckDuckGo 搜索工具特点
   - 无需 API 密钥，开箱即用
   - 支持隐私保护的网络搜索，返回搜索结果摘要
   - 适合快速原型开发和测试，但可能受地区网络限制

6. 工具调用流程
   - invoke() 方法执行工具，传入查询字符串或字典
   - 工具内部发起 HTTP 请求到 DuckDuckGo，返回搜索结果文本

7. 工具的使用场景
   - 单独调用：直接使用工具获取信息
   - Agent 集成：将工具绑定到 Agent，由 LLM 自主决策何时调用
   - 链式组合：在 LCEL 链中组合多个工具
   - 函数调用：与 ChatModel.bind_tools() 配合使用

8. 工具包（Toolkit）的概念
   - 工具包是一组设计用于一起执行特定任务的工具集，具有便捷加载方法 get_tools()
   - 所有工具包都公开 get_tools() 返回工具列表，无需一个一个加载
   - 通常是同一服务提供商的一系列工具，例如 AzureAiServicesToolkit 提供 5 个工具：图像分析、文档智能、语音转文本、文本转语音、医疗文本分析
   - 工具包包含多组工具，一般很少单独拆分使用，通常用在 Agent 或函数调用中，实际开发更多手动实例化工具组装
   - 官方文档：https://imooc-langchain.shortvar.com/docs/integrations/toolkits/

9. convert_to_openai_tool() 辅助函数
   - 将 LangChain 工具转换为 OpenAI 函数调用格式（符合 OpenAI tools 参数规范的 JSON Schema）
   - 便于与原生 OpenAI API 集成，示例：convert_to_openai_tool(search)

10. 工具设计的最佳实践
   - name 要简洁明确，避免歧义
   - description 要清晰详细，包含使用场景说明
   - 参数模式要完整，每个参数都要有 description
   - 返回值要结构化，便于 LLM 理解
   - 生产环境优先使用 API 密钥方式的工具（更稳定）

11. 典型输出示例与观察点
   - 实例化 DuckDuckGoSearchRun() 后打印四大属性，得到：
       name: duckduckgo_search
       description: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
       args: {'query': {'title': 'Query', 'description': 'search query to look up', 'type': 'string'}}
       return_direct: False
   - 观察点：args 是标准 JSON Schema，仅含一个字符串字段 query；description 直接决定了 LLM 是否会选用该工具
   - convert_to_openai_tool(search) 的输出包成 {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': {...}}}，可直接喂给 OpenAI 的 tools 参数

===================================================================================
"""
from langchain_community.tools import DuckDuckGoSearchRun

# 创建 DuckDuckGo 搜索工具实例
# description 参数可以自定义工具描述，帮助 LLM 理解工具用途
# 如果不传入 description，工具会使用默认描述
search = DuckDuckGoSearchRun(description="xxx")

# 调用工具执行搜索
# invoke() 方法接收搜索查询字符串，返回搜索结果摘要
print(search.invoke("LangChain的最新版本是什么?"))

# 工具的核心属性
print("名字：", search.name)  # 工具的唯一标识符
print("描述：", search.description)  # 工具的功能描述
print("参数：", search.args)  # 工具的参数模式（JSON Schema）
print("是否直接返回：", search.return_direct)  # 是否跳过 LLM 后处理

# convert_to_openai_tool() 可将 LangChain 工具转换为 OpenAI 函数调用格式
# print(convert_to_openai_tool(search))
