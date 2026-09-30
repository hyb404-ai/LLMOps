#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/26 14:51
@Author  : thezehui@gmail.com
@File    : 01.Hugging Face本地嵌入模型.py

===================================================================================
知识点讲解：HuggingFaceEmbeddings 本地嵌入模型
===================================================================================

1. 本地嵌入 vs 云端嵌入
   - 云端（OpenAI/千帆）：按 token 计费、依赖网络、数据需外发、无需本地算力
   - 本地（HuggingFace）：一次下载永久免费、完全离线、数据不出内网、
     但首次需下载模型权重，且推理依赖本机 CPU/GPU 算力
   - 涉及敏感数据、需要极致成本控制或高频批量嵌入的场景，优先考虑本地模型
   - 在对数据保密要求极高的场合下，数据不允许传递到外网，这个时候就可以考虑
     使用本地的文本嵌入模型

2. HuggingFaceEmbeddings 组件
   - 位于 langchain_huggingface 包，底层依赖 sentence-transformers 库
   - 首次实例化时会自动从 HuggingFace Hub 下载模型权重到 cache_folder，
     之后从本地缓存加载，不再联网
   - 实例化即完成模型加载（占用内存/显存），因此应当全局复用同一个实例，
     不要在循环或请求处理函数内反复创建
   - sentence-transformers 是一个用于生成和使用预训练的文本嵌入的库，
     基于 transformer 架构，也是目前使用量最大的本地文本嵌入模型
   - 首次加载的时候，会将模型从本地应用加载到内存中，首次加载相对会慢，
     后续调用速度即可恢复正常

3. 依赖安装
   - pip install langchain-huggingface sentence-transformers
   - sentence-transformers 会连带安装 torch、transformers，体积较大（约 2GB+）

4. 常用 sentence-transformers 模型对比
   - all-MiniLM-L6-v2   : 6 层，384 维，最快最轻（约 90MB），英文效果好
   - all-MiniLM-L12-v2  : 12 层，384 维，速度与效果折中（本例使用，约 130MB）
   - all-mpnet-base-v2  : 768 维，英文效果最佳，体积与耗时更大
   - paraphrase-multilingual-MiniLM-L12-v2 : 384 维，支持 50+ 语言
   - BAAI/bge-small-zh-v1.5 / bge-large-zh-v1.5 : 中文场景表现优秀，推荐中文项目使用
   - 注意：all-MiniLM 系列主要在英文语料上训练，处理中文时效果有限，
     本例用中文输入只是为了演示 API 调用方式

5. 重要参数说明
   - model_name    : HuggingFace Hub 上的模型仓库标识（组织名/模型名）
   - cache_folder  : 模型权重的本地缓存目录，不传则默认存放在 ~/.cache/huggingface
   - model_kwargs  : 传给 SentenceTransformer 的参数，如 {"device": "cuda"} 使用 GPU，
                     {"device": "mps"} 在 Apple Silicon 上使用 Metal 加速
   - encode_kwargs : 传给 encode() 的参数，如 {"normalize_embeddings": True}
                     会对输出向量做 L2 归一化，使余弦相似度等价于点积
   - multi_process : 是否启用多进程编码，大批量离线嵌入时可显著提速

6. 最佳实践建议
   - 显式指定 cache_folder 并纳入部署镜像/数据卷，避免生产环境启动时才下载模型
   - 中文业务请改用 bge-*-zh 系列，不要直接用 all-MiniLM
   - 生产环境建议开启 normalize_embeddings=True，让相似度计算更稳定
   - 国内网络下载困难时，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 走镜像
   - 切换嵌入模型后，向量数据库中的历史数据必须全量重建（维度与向量空间都变了）
   - 可以加载任意本地的文本嵌入模型，传递 model_name 与 cache_folder 参数即可

===================================================================================
"""
from langchain_huggingface import HuggingFaceEmbeddings

# 1.创建本地 HuggingFace 嵌入模型实例
# HuggingFaceEmbeddings 参数说明：
#   model_name="sentence-transformers/all-MiniLM-L12-v2"
#       HuggingFace Hub 上的模型标识，格式为「组织名/模型名」
#       该模型输出 384 维向量，属于轻量级通用句向量模型
#   cache_folder="./embeddings/"
#       模型权重的本地缓存目录。首次运行会把 config、tokenizer、
#       pytorch_model.bin 等文件下载到该目录（约 130MB）；
#       后续运行直接从此目录加载，完全离线，无需网络
# 执行副作用：本行会真正把模型加载进内存，耗时从几百毫秒到数十秒不等，
#            因此应当在模块级创建一次并全局复用
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    cache_folder="./embeddings/"
)

# 2.嵌入单条查询文本
# embed_query(text) 作用：把单条文本转换为向量
#   参数：text -> str
#   返回：list[float]，长度等于模型维度（all-MiniLM-L12-v2 为 384）
#   底层流程：tokenizer 分词 -> Transformer 前向推理 -> mean pooling 池化 -> 输出句向量
#   与云端模型的差异：没有 HTTP 请求，全部在本机计算，无网络延迟也无费用
query_vector = embeddings.embed_query("你好，我是慕小课，我喜欢打篮球游泳")

# 打印完整向量内容（384 个浮点数）
print(query_vector)
# 打印向量维度，用于确认模型规格。此处应输出 384，
# 明显小于 OpenAI text-embedding-3-small 的 1536 维，
# 这也意味着存储成本更低，但语义表达能力相对有限
print(len(query_vector))
