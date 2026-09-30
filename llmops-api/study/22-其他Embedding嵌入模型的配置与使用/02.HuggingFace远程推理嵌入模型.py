#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/26 15:36
@Author  : thezehui@gmail.com
@File    : 02.HuggingFace远程推理嵌入模型.py

===================================================================================
知识点讲解：HuggingFaceEndpointEmbeddings 远程推理嵌入模型
===================================================================================

1. 本地加载 vs 远程推理
   - HuggingFaceEmbeddings（上一个示例）：把模型权重下载到本机，用本地算力推理
       优点：完全离线、无调用次数限制、数据不外发
       缺点：占用磁盘与内存、依赖本机算力、首次需下载大文件
   - HuggingFaceEndpointEmbeddings（本示例）：只发 HTTP 请求，推理在远端完成
       优点：零本地算力消耗、无需下载权重、启动快、客户端依赖轻
       缺点：依赖网络、受限流约束、数据需外发、免费额度有限
   - 部分模型的文件比较大，如果只是短期内调试，可以考虑使用 HuggingFace
     提供的远程嵌入模型

2. HuggingFace Inference API 两种形态
   - Serverless Inference API（免费/共享）：
     地址形如 https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
     由 HuggingFace 托管公共模型，有速率限制；冷启动时首次请求可能返回 503，
     表示模型正在加载（loading），需要重试
   - Dedicated Inference Endpoint（付费/独占）：
     用户自行部署的专属推理端点，有独立 URL 与稳定性能，
     通过 model="https://xxx.endpoints.huggingface.cloud" 直接传入完整地址

3. 鉴权配置
   - 需要 HuggingFace Access Token，在 .env 中配置环境变量：
     HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxx
   - Token 在 https://huggingface.co/settings/tokens 创建，读权限（read）即可
   - 也可通过构造参数 huggingfacehub_api_token="hf_xxx" 显式传入
   - 需要在 Hugging Face 官网（https://huggingface.co/）的 setting 中
     添加对应的访问秘钥，并配置到 .env 文件中

4. 重要参数说明
   - model     : 模型标识（组织名/模型名）或完整的 Inference Endpoint URL
   - task      : 推理任务类型，嵌入场景默认为 "feature-extraction"
   - huggingfacehub_api_token : 访问令牌，不传则读取环境变量
   - model_kwargs : 附加到请求体的参数

5. 与本地模型的一致性
   - 同一个模型（如 all-MiniLM-L12-v2）无论本地推理还是远程推理，
     输出维度相同（384），语义空间也相同，理论上向量可以互换使用
   - 因此可以在开发期用远程 API 快速验证，上线后切换为本地部署，
     无需重建向量数据库（前提是模型版本完全一致）

6. 最佳实践建议
   - 远程推理存在网络抖动风险，生产环境务必配合 with_retry() 或重试装饰器
   - 免费 Serverless API 有严格限流，不适合批量嵌入大规模知识库，
     大批量任务建议改用本地模型或自建专属端点
   - 结合 CacheBackedEmbeddings 缓存结果，可显著减少重复请求
   - 敏感数据不要走公共 Serverless API，应使用本地模型或私有化端点
   - 在本地服务器上就无需配置对应的文本嵌入模型了

===================================================================================
"""
import dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# 从 .env 文件加载环境变量，此处关键变量为 HUGGINGFACEHUB_API_TOKEN
# 该 Token 用于 Inference API 的 Bearer 鉴权，缺失会导致 401 错误
dotenv.load_dotenv()

# 1.创建远程推理嵌入模型实例
# HuggingFaceEndpointEmbeddings 参数说明：
#   model="sentence-transformers/all-MiniLM-L12-v2"
#       既可以是 Hub 上的模型标识（走 Serverless Inference API），
#       也可以是自建专属端点的完整 URL（走 Dedicated Endpoint）
#   task        : 默认 "feature-extraction"，即特征抽取（句向量）任务
#   huggingfacehub_api_token : 未显式传入，自动读取 HUGGINGFACEHUB_API_TOKEN
# 关键差异：本行只创建 HTTP 客户端，不会下载任何模型权重，
#          因此实例化几乎瞬间完成，内存占用极小
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")

# 2.嵌入单条查询文本
# embed_query(text) 作用：把单条文本转换为向量
#   参数：text -> str，待嵌入文本
#   返回：list[float]，长度 384（与本地加载同一模型的输出维度一致）
#   底层流程：
#     a) 向 HuggingFace Inference API 发起 POST 请求，body 为 {"inputs": text}
#     b) 远端服务器执行模型推理并做池化
#     c) 返回向量数组，客户端解析为 list[float]
#   常见异常：
#     · 401 Unauthorized -> Token 缺失或无效
#     · 503 Model is loading -> 模型冷启动中，稍后重试即可
#     · 429 Too Many Requests -> 触发免费额度限流
query_vector = embeddings.embed_query("你好，我是慕小课，我喜欢打篮球游泳")

# 打印完整向量内容
print(query_vector)
# 打印向量维度，应输出 384，与本地加载 all-MiniLM-L12-v2 的结果完全一致，
# 这验证了「同一模型不同部署方式，向量空间相同」的结论
print(len(query_vector))
