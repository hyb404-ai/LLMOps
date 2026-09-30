#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 13:19
@Author  : thezehui@gmail.com
@File    : 1.检索器组件与可运行时配置.py

===================================================================================
知识点讲解：检索器的运行时动态配置（configurable_fields）

1. BaseRetriever 检索器基类（定义与定位）
   - LangChain 对"检索器"的定义极其简洁：传递一段 query，返回与这段文本相关联的文档列表的组件，就叫检索器
   - 所有检索器的基类是 BaseRetriever，它继承了 RunnableSerializable，因此天生就是一个 Runnable 可运行组件，支持 Runnable 的全部配置能力
   - 使用技巧同样简单：按特定规则创建好检索器后（通过 as_retriever() 或直接调用构造函数），调用 invoke() 即可
   - 版本提醒：get_relevant_documents() 从 0.3.0 版本开始被弃用（那是 Runnable 与 LCEL 出现之前的老写法），新代码统一用 invoke()

2. VectorStoreRetriever 的三个专有属性
   - VectorStoreRetriever 是 BaseRetriever 的子类，专门服务向量数据库，内部实现了 _get_relevant_documents()，并额外定义了 3 个属性：
     * vectorstore   - 检索器归属的向量数据库
     * search_type   - 搜索类型
     * search_kwargs - 搜索参数
   - 这 3 个属性全部来源于 as_retriever() 的入参或实例化时传入的参数
   - 正因为它是 Runnable 组件，才可以用 .configurable_fields() 修改类内部的参数 —— 这就是本示例的核心用法

3. _get_relevant_documents 的分发逻辑与三条推论
   - 源码等价逻辑：
       if   search_type == "similarity":
                docs = vectorstore.similarity_search(query, **search_kwargs)
       elif search_type == "similarity_score_threshold":
                docs_and_similarities = \
                    vectorstore.similarity_search_with_relevance_scores(query, **search_kwargs)
                docs = [doc for doc, _ in docs_and_similarities]   # 得分在此被丢弃
       elif search_type == "mmr":
                docs = vectorstore.max_marginal_relevance_search(query, **search_kwargs)
       else:
                raise ValueError(f"search_type of {search_type} not allowed.")
   - 三个关键推论：
     * search_kwargs 是直接用 ** 解包传给底层函数的，所以传了底层函数不认识的键就会报 TypeError —— 这正是"切到 mmr 必须去掉 score_threshold"的根本原因
     * similarity_score_threshold 分支拿到了得分却只保留 Document，这解释了"为什么检索器返回值不带相关性得分"
     * search_type 传错值会直接抛 ValueError，而不是静默降级

4. 为什么检索器需要"运行时可配置"
   - 同一个知识库，不同业务场景需要不同的检索策略：精确问答要 similarity + 高阈值，列举式问答要 mmr + 多样性
   - 若为每种策略都 as_retriever() 一个新实例，会产生大量重复对象，且难以根据用户输入动态切换
   - configurable_fields 让"一个检索器实例"在调用时动态改变检索行为

5. configurable_fields() 的工作原理
   - 它把 Runnable 的某些构造字段"暴露"为运行时可覆盖的配置项
   - 返回的是 RunnableConfigurableFields 包装对象，仍然是 Runnable
   - 调用时通过 config={"configurable": {...}} 传入覆盖值，未覆盖的字段沿用创建时的默认值
   - 这些配置只对本次调用生效，不会污染原实例（无状态、线程安全）

6. ConfigurableField 的关键参数
   - id：配置项的唯一标识，必填，运行时通过这个 id 定位字段
   - name：人类可读名称，用于 LangServe Playground 等 UI 展示
   - description：字段说明
   - annotation：类型标注，配合 ConfigurableFieldSpec 做校验
   - 注意 id 必须全局唯一，多个组件混用时重名会互相覆盖

7. with_config() 与 invoke(config=...) 的区别
   - with_config(configurable={...})：返回一个"已绑定配置"的新 Runnable，适合把配置好的检索器传给下游复用
   - invoke(input, config={"configurable": {...}})：只对这一次调用生效
   - 两者等价，前者更适合链式写法，后者更适合一次性调用

8. 可配置字段的常见搭配
   - search_type + search_kwargs 同时开放，才能完整切换检索策略
   - 只开放 search_kwargs 时，切不了 similarity 与 mmr
   - 只开放 search_type 时，从 similarity_score_threshold 切到 mmr 会带上无效的 score_threshold 参数，可能报错 —— 所以两者要成对开放

9. 与 configurable_alternatives 的区别
   - configurable_fields：改"同一个组件的参数"（本例的用法）
   - configurable_alternatives：换"整个组件实现"（如 FAISS 换成 Weaviate、GPT 换成 Claude），粒度更粗

10. 最佳实践建议
   - 需要动态切换检索策略时，search_type 与 search_kwargs 务必成对开放为可配置字段
   - ConfigurableField 的 id 加上业务前缀（如 db_search_type）避免与其他组件冲突
   - 切换到 mmr 时不要保留 score_threshold，切到 similarity_score_threshold 时必须带上它
   - with_config 返回新对象、不修改原实例，可以安全地在多协程/多线程中共享基础检索器
   - 生产环境可以把"检索策略"做成用户可选项或由意图识别模型自动决策，再用 configurable 注入，实现自适应 RAG
   - 本示例直接复用已有 Collection 的数据（未调用 add_documents），避免重复入库

===================================================================================
"""
import dotenv
import weaviate
from langchain_core.runnables import ConfigurableField
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

# 加载 .env 环境变量，提供 OpenAI 凭证
dotenv.load_dotenv()

# 1.构建向量数据库
# weaviate.connect_to_wcs(cluster_url, auth_credentials) -> WeaviateClient
#   作用：连接 Weaviate 云端集群
#   参数：cluster_url      - 集群 REST 端点
#         auth_credentials - AuthApiKey 包装的 API Key
#   返回：Weaviate v4 客户端实例
#
# WeaviateVectorStore(client, index_name, text_key, embedding) -> WeaviateVectorStore
#   作用：把已有 Weaviate Collection 包装为 LangChain 向量库
#   参数：index_name - Collection 名称，指向已写入数据的 "DatasetDemo"
#         text_key   - 存放正文的字段名
#         embedding  - 嵌入模型，必须与入库时一致
#   返回：向量库实例
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://eftofnujtxqcsa0sn272jw.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("21pzYy0orl2dxH9xCoZG1O2b0euDeKJNEbB0"),
    ),
    index_name="DatasetDemo",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# 2.转换检索器
# 第一步：db.as_retriever(search_type, search_kwargs) -> VectorStoreRetriever
#   作用：把向量库包装为标准检索器，这里给出的是"默认策略"
#   参数：search_type="similarity_score_threshold" - 默认走带阈值的相似度检索
#         search_kwargs={"k": 10, "score_threshold": 0.5} - 默认最多 10 条、阈值 0.5
#
# 第二步：.configurable_fields(**kwargs) -> RunnableConfigurableFields
#   作用：把指定字段声明为"运行时可覆盖"，返回一个包装后的 Runnable
#   参数：以 "字段名=ConfigurableField(id=...)" 的形式声明映射关系
#         search_type=ConfigurableField(id="db_search_type")
#           → 运行时用 "db_search_type" 这个 id 覆盖 search_type 字段
#         search_kwargs=ConfigurableField(id="db_search_kwargs")
#           → 运行时用 "db_search_kwargs" 这个 id 覆盖 search_kwargs 字段
#   返回：RunnableConfigurableFields 实例，未传 configurable 时行为与原检索器完全一致
#   注意：两个字段必须同时开放，否则无法完整切换检索策略（见知识点 5）
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5},
).configurable_fields(
    search_type=ConfigurableField(id="db_search_type"),
    search_kwargs=ConfigurableField(id="db_search_kwargs"),
)

# 3.修改运行时配置执行MMR搜索，并返回4条数据
# retriever.with_config(configurable: dict) -> Runnable
#   作用：返回一个"已绑定运行时配置"的新 Runnable，不修改原 retriever 对象
#   参数：configurable - key 为 ConfigurableField 的 id，value 为要覆盖的新值
#         "db_search_type": "mmr"
#           → 把检索策略从 similarity_score_threshold 切换为最大边际相关性搜索
#         "db_search_kwargs": {"k": 4}
#           → 整体替换（不是合并）原 search_kwargs，
#              因此 score_threshold 被自然丢弃，这正是 mmr 所需要的
#              （mmr 不接受 score_threshold，若残留会导致参数错误）
#   返回：配置覆盖后的 Runnable
#
# .invoke(input: str) -> list[Document]
#   作用：执行检索
#   参数：input - 查询字符串
#   返回：Document 列表
#   本次调用的数据流转：
#     "关于应用配置的接口有哪些？"
#       → embed_query 生成查询向量
#       → 因 search_type=mmr，向量库先按相似度取 fetch_k（默认 20）条候选
#       → 本地执行 MMR 迭代，按 λ=0.5 权衡相关性与多样性
#       → 返回 k=4 条差异化文档
mmr_documents = retriever.with_config(
    configurable={
        "db_search_type": "mmr",
        "db_search_kwargs": {
            "k": 4,
        }
    }
).invoke("关于应用配置的接口有哪些？")

# 打印完整的检索结果（含 page_content 与 metadata）
# 提示：变量名与打印文案写的是"相似性搜索"，但实际执行的是 MMR 搜索
print("相似性搜索: ", mmr_documents)

# 打印命中条数，预期为 4（由 db_search_kwargs 的 k=4 决定）
# 对比：若不覆盖配置，默认走阈值检索，条数 <= 10 且可能为 0
print("内容长度:", len(mmr_documents))

# 抽样打印前两条文档的开头 20 个字符
# 观察要点：两条内容应当明显不同，体现 MMR 的多样性效果
print(mmr_documents[0].page_content[:20])
print(mmr_documents[1].page_content[:20])
