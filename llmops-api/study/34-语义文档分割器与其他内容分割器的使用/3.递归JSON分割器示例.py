#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/2 21:56
@Author  : thezehui@gmail.com
@File    : 3.递归JSON分割器示例.py

===================================================================================
知识点讲解：递归 JSON 分割器（RecursiveJsonSplitter）
===================================================================================

1. 组件定位：递归 JSON 分割器（RecursiveJsonSplitter）
   - 专门处理 JSON 数据（API 响应、配置文件、数据交换），解决大 JSON 超过模型上下文窗口的问题
   - 既要切小，又要保持 JSON 语法有效性与结构完整性，普通字符分割会破坏 JSON

2. 工作原理：深度优先遍历
   - 按深度优先方式遍历整个 JSON，一层层往下读取数据，将对应数据提取生成新的 JSON
   - 直到数据大小接近块大小才停止；极端情况下仍会超过预设大小（如 Key 或 Value 过长、单条数据就超预设）
   - 优先在对象 / 数组顶层切分，必要时递归到更深层次，尽量保持嵌套对象完整

3. max_chunk_size 与 min_chunk_size 参数
   - max_chunk_size：每个 JSON 块的最大字符数（必填，本例 300）
   - min_chunk_size：最小块大小（可选，默认无限制）
   - 分割器尽量把每块控制在二者之间，但单条数据超长时无法保证

4. split_json() vs create_documents()
   - split_json()：接收 JSON 对象（dict/list），返回 JSON 块列表（List[Dict]）
   - create_documents()：接收 JSON 块列表，返回 Document 对象列表
   - 通常组合使用：split_json() → create_documents()；create_documents 把每个块封装为 Document，page_content 为 JSON 字符串

5. 典型输出示例与观察点
   - 对 OpenAPI 规范（大 JSON）拆分后输出类似：
       page_content='{"openapi": "3.1.0", "info": {"title": "LangSmith", "version": "0.1.0"}, ...}'
       page_content='{"paths": {"/api/v1/sessions/{session_id}": {"get": {...}}}}'
   - 观察点：每个块的 page_content 仍是合法 JSON 片段，且尽可能保留嵌套结构

6. 局限与组合使用建议
   - 局限：若某个值不是嵌套 JSON 而是超长字符串，RecursiveJsonSplitter 不会拆分该字符串
   - 应对：配合 RecursiveCharacterTextSplitter 对分割结果做二次拆分，降低单条超预设大小的风险
   - 适合独立性较强、跨块引用关系少的 JSON 结构

7. 使用场景
   - 大型 API 响应（OpenAPI / Swagger 文档）、配置文件、结构化数据、基于 JSON 的知识库

===================================================================================
"""
import json

import requests
from langchain_text_splitters import RecursiveJsonSplitter

# 1.获取并加载json
# 从 LangSmith API 获取 OpenAPI 规范文档（大型 JSON）
url = "https://api.smith.langchain.com/openapi.json"

# requests.get(): 发起 HTTP GET 请求
# .json(): 将响应解析为 Python 字典
json_data = requests.get(url).json()

# 输出原始 JSON 的字符数（序列化后的长度）
print(len(json.dumps(json_data)))

# 2.递归JSON分割器
# RecursiveJsonSplitter(): 递归 JSON 分割器
#   参数:
#     - max_chunk_size: 每个 JSON 块的最大字符数
#       300 表示每块尽量不超过 300 字符
#     - min_chunk_size: 最小块大小（可选），默认无限制
#   返回: RecursiveJsonSplitter 实例
#   作用:
#     - 递归遍历 JSON 结构
#     - 在对象键、数组元素边界分割
#     - 确保分割后每块都是有效的 JSON
text_splitter = RecursiveJsonSplitter(max_chunk_size=300)

# 3.分割json数据并创建文档
# split_json(): 分割 JSON 对象
#   参数:
#     - json_data: Python 字典或列表（已解析的 JSON）
#   返回: List[Dict]，JSON 块列表
#   工作流程:
#     1. 递归遍历 JSON 结构
#     2. 计算每个子结构的序列化大小
#     3. 在 max_chunk_size 限制下尽量保持结构完整
#     4. 必要时继续递归分割嵌套对象/数组
json_chunks = text_splitter.split_json(json_data)

# create_documents(): 将 JSON 块转换为 Document 对象
#   参数:
#     - texts: JSON 块列表（可以是字典列表或字符串列表）
#   返回: List[Document]
#   作用:
#     - 将每个 JSON 块封装为 Document 对象
#     - page_content 包含 JSON 的字符串表示
#     - metadata 可能包含 JSON 路径等信息
chunks = text_splitter.create_documents(json_chunks)

# 4.输出内容
# 统计所有块的总字符数，验证分割效果
count = 0
for chunk in chunks:
    count += len(chunk.page_content)

# 输出总字符数（应该接近原始 JSON 的长度）
print(count)
