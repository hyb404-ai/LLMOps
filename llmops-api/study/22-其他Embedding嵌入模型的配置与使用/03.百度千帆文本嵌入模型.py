#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/26 15:53
@Author  : thezehui@gmail.com
@File    : 03.百度千帆文本嵌入模型.py

===================================================================================
知识点讲解：QianfanEmbeddingsEndpoint 百度千帆嵌入模型
===================================================================================

1. 为什么需要国产嵌入模型
   - 网络可达性：国内访问 OpenAI 需代理，千帆为国内直连，延迟低且稳定
   - 中文效果：Embedding-V1 基于中文语料训练，中文语义理解优于 all-MiniLM 等英文模型
   - 合规要求：数据留存在境内，满足国内数据合规与备案要求
   - 成本：千帆嵌入模型提供免费额度，中小项目基本够用
   - 如果想对接国内的文本嵌入模型提供商，可以考虑百度千帆，是目前国内生态最好，
     支持的模型最多，速度最快的 AI 应用开发平台

2. QianfanEmbeddingsEndpoint 组件
   - 位于 langchain_community.embeddings.baidu_qianfan_endpoint
   - 底层依赖 qianfan 官方 SDK：pip install qianfan
   - 同样实现 Embeddings 接口，embed_query / embed_documents 用法与 OpenAI 完全一致，
     这正是 LangChain 抽象层的价值：更换模型供应商无需改动上层业务代码
   - 由于目前百度千帆并没有单独封装到独立的包，可以直接从 langchain_community
     中导入

3. 鉴权配置（.env 文件）
   - QIANFAN_AK=你的 API Key
   - QIANFAN_SK=你的 Secret Key
   - 在百度智能云千帆控制台「应用接入」中创建应用后获取
   - 也可通过构造参数 qianfan_ak / qianfan_sk 显式传入
   - 注意：新版千帆也支持 QIANFAN_ACCESS_KEY / QIANFAN_SECRET_KEY 的 IAM 鉴权方式

4. 千帆可选嵌入模型
   - Embedding-V1  : 默认模型，384 维，中文通用场景，单条文本上限 384 token
   - bge-large-zh  : 1024 维，中文效果更优
   - bge-large-en  : 1024 维，英文场景
   - tao-8k        : 1024 维，支持 8K 长文本输入
   - 指定方式：QianfanEmbeddingsEndpoint(model="bge-large-zh")
   - 支持的模型涵盖：Embedding-V1、bge-large-zh、bge-large-en、tao-8k

5. 平台限制（重要注意事项）
   - 单次请求的文本条数上限为 16 条，超出会报错
     => LangChain 内部已做分批处理（chunk_size=16），但自行调用原生 SDK 时需注意
   - Embedding-V1 单条文本上限 384 token，超长文本必须先做文档分割
   - 存在 QPS 限流，免费额度下并发较低，批量嵌入需自行控制速率

6. 最佳实践建议
   - 中文知识库优先选择 bge-large-zh（1024 维）而非默认的 Embedding-V1
   - 务必配合 CacheBackedEmbeddings 缓存，减少调用次数以规避 QPS 限流
   - 长文本先经 RecursiveCharacterTextSplitter 切分到 384 token 以内
   - 与 OpenAI 向量不可混用：维度不同（384 vs 1536），语义空间也完全不同，
     更换模型后必须重建整个向量数据库
   - 生产环境建议用 with_retry() 包装，应对偶发限流与网络错误

===================================================================================
"""
import dotenv
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint

# 从 .env 文件加载环境变量，此处关键变量为 QIANFAN_AK 与 QIANFAN_SK
# 千帆 SDK 会用这两个凭证换取 access_token 后再调用嵌入接口
dotenv.load_dotenv()

# 1.创建百度千帆嵌入模型实例
# QianfanEmbeddingsEndpoint 参数说明（本例全部使用默认值）：
#   model      : 嵌入模型名，默认 "Embedding-V1"（384 维中文通用模型）
#                可改为 "bge-large-zh"（1024 维）以获得更好的中文效果
#   endpoint   : 自定义推理服务地址，用于对接千帆上自行部署的模型
#   qianfan_ak : API Key，不传则读取环境变量 QIANFAN_AK
#   qianfan_sk : Secret Key，不传则读取环境变量 QIANFAN_SK
#   chunk_size : 单次请求提交的文本条数，默认 16（受千帆平台上限约束）
# 执行副作用：首次调用时 SDK 会用 AK/SK 请求一次 OAuth 接口获取 access_token
embeddings = QianfanEmbeddingsEndpoint()

# 2.嵌入单条查询文本
# embed_query(text) 作用：把单条文本转换为向量
#   参数：text -> str，待嵌入文本（需在 384 token 以内）
#   返回：list[float]，长度等于模型维度（Embedding-V1 为 384）
#   底层流程：
#     a) 用 AK/SK 获取（或复用缓存的）access_token
#     b) POST 千帆 embeddings 接口，body 为 {"input": ["文本"]}
#     c) 解析响应中的 data[0].embedding 字段返回向量
#   注意：批量场景请改用 embed_documents(texts)，
#        LangChain 会按 chunk_size=16 自动分批，规避平台单次条数上限
query_vector = embeddings.embed_query("我叫慕小课，我喜欢打篮球游泳")

# 打印完整向量内容（384 个浮点数）
print(query_vector)
# 打印向量维度，Embedding-V1 应输出 384
# 对比参考：OpenAI text-embedding-3-small=1536，bge-large-zh=1024
# 这再次说明「不同嵌入模型的向量不可互换」，向量库必须与模型一一绑定
print(len(query_vector))
