#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/1 23:17
@Author  : thezehui@gmail.com
@File    : 3.URL网页加载器.py

===================================================================================
知识点讲解：URL 网页加载器的使用
===================================================================================

1. 网络类加载器全景
   - 除本地文件，LangChain 还封装了大量加载「网络文件」的加载器，例如：
     * WebBaseLoader            : 普通网页
     * 腾讯云 COS 对象存储加载器  : 云端对象存储
     * Bilibili 字幕加载器       : 视频字幕
     * Notion 数据库加载器       : 在线协作文档/数据库
   - 使用技巧与本地文件加载器大差不差：构建加载器 → 调用 load()

2. WebBaseLoader 的核心功能
   - 从网页 URL 加载内容并转换为 Document 对象
   - 基于 BeautifulSoup 解析 HTML，提取可见文本
   - 自动处理 HTTP 请求，支持基本的网页抓取

3. HTML 解析与文本提取
   - 自动去除 HTML 标签，只保留纯文本
   - 会保留基本的文本结构（段落、换行等）
   - 过滤掉 script、style 等不可见标签内容

4. metadata 中的信息
   - source  : 网页的 URL 地址
   - title   : 网页标题（从 <title> 标签提取）
   - language: 网页语言，未检测到时为字符串 'No language found.'

5. 高级用法
   - bs_kwargs 参数可自定义 BeautifulSoup 解析器
   - requests_kwargs 可设置 HTTP 请求头、超时等
   - 支持加载多个 URL（传入 URL 列表）

6. WebBaseLoader 的底层行为
   - 会从 HTML 网页中加载「所有文本」（去除 HTML 标签）并合并为一个 Document
   - page_content 是整页的纯文本
   - 利用它可快速实现一个「基于特定网页问答」的聊天机器人
   - 翻译文档：https://imooc-langchain.shortvar.com/docs/integrations/document_loaders/web_base/

7. 典型输出示例（加载 https://imooc.com）
   [Document(page_content='\n\n\n\n\n慕课网-程序员的梦工厂\n\n\n\n...（大量 \n 与空白）...',
             metadata={'source': 'https://imooc.com',
                       'title': '慕课网-程序员的梦工厂',
                       'language': 'No language found.'})]
   - 单个 URL 返回一个 Document；metadata 字段含义见第 4 条

8. 重要实践坑：大量空白字符
   - 输出 page_content 中夹杂极多空格、换行、Tab（HTML 排版结构被剥离标签后留下的残渣）
   - 危害：直接分割存入向量库会大幅稀释信息密度，并极大降低检索与生成的效率和正确性
   - 因此 WebBaseLoader 输出通常需要二次清洗，这正是下一节「自定义文档加载器」要解决的典型问题之一

9. 注意事项
   - 需要网络连接才能访问 URL
   - 某些网站有反爬虫机制，需要设置请求头
   - 动态加载内容（JavaScript 渲染）可能无法获取
   - 建议遵守网站的 robots.txt 规则

===================================================================================
"""
from langchain_community.document_loaders import WebBaseLoader

# WebBaseLoader(): 网页内容加载器
#   参数:
#     - web_path: 网页 URL 地址（字符串或字符串列表）
#     - bs_kwargs: BeautifulSoup 解析器参数（可选）
#     - requests_kwargs: HTTP 请求参数（可选），如 headers、timeout
#   返回: WebBaseLoader 实例
#   作用: 发起 HTTP 请求获取网页内容，解析 HTML 提取文本
loader = WebBaseLoader("https://imooc.com")

# load(): 执行网页加载和解析
#   返回: List[Document]
#   作用: 发送 HTTP GET 请求，解析 HTML，提取可见文本内容
documents = loader.load()

# 输出文档对象列表
print(documents)

# 输出文档数量（单个 URL 返回一个 Document）
print(len(documents))

# 输出元数据信息
# 包含 source（URL）、title（网页标题）等
print(documents[0].metadata)
