#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/5 16:28
@Author  : thezehui@gmail.com
@File    : 10.自查询检索器实现元数据过滤.py

===================================================================================
知识点讲解：自查询检索器实现动态元数据过滤

1. 设计动机：用户提问隐式携带筛选条件
   - 普通相似性搜索对所有文本一视同仁，无法识别"2023年""评分高于9.5"这类隐含的过滤条件
   - 例如"关于2023年全年AI的新闻"，year=2023 这个条件在向量空间里和 2022 年数据非常接近，普通检索极易把其他年份也召回
   - 自查询的解决思路：让 LLM 从原始问题里把筛选条件抽出来，构建成元数据过滤器，既压缩检索范围又提升准确性

2. 组件定位：什么是自查询检索器（SelfQueryRetriever）
   - 传统检索器只能做语义相似度检索；自查询检索器能理解用户问题中的过滤条件，自动构建元数据过滤器
   - 它在 RAG 6 阶段中属于第 3 阶段"查询构建"（自检索/自查询），与"查询转换"阶段（子查询、问题分解、假设性文档）相对：后者执行检索时仍用固定的筛选条件，前者把隐含条件也变成真正的过滤器
   - 查询构建在不同数据库上的对应形态：关系型数据库 -> LLM 生成 SQL；图数据库 -> LLM 生成图查询；向量数据库 -> LLM 生成元数据过滤器（本示例）

3. 使用前提（并非所有数据都适合）
   - 需同时满足三条件：①存储的 Document 本身带有可用于筛选的 metadata；②对应数据库类型支持按字段筛选；③LangChain 针对该向量库做了 SelfQueryRetriever 适配（未适配则自行实现转换器与解析器难度很大）
   - 注意：向量库迭代快，可能出现"LangChain 已更新该库集成，但 SelfQueryRetriever 内部逻辑尚未同步"的兼容性坑

4. 核心组件：AttributeInfo 与 metadata_field_info
   - AttributeInfo：描述一个元数据字段的类型与含义，供 LLM 判断该不该用此字段做过滤
   - 关键约束：AttributeInfo 的 name 必须与 Document.metadata 中的键名严格一致；description 要写清业务含义（LLM 靠它决策）
   - document_contents：对文档正文内容的简短描述（如"电影的名字"），帮助 LLM 区分"用于语义比对的文本"与"过滤条件"

5. 底层实现原理：预设 Prompt 生成规则化文本 -> 解析 -> 调用接口
   - SelfQueryRetriever 没有黑魔法，走的是通用链路：用 FewShotPromptTemplate + 函数回调/结构化输出，让 LLM 按 DEFAULT_SCHEMA 生成一段 JSON 查询语句
   - 该 schema 的硬性约定：query 字段只放"用于和文档内容比对的文本"；filter 字段放逻辑过滤条件表达式且 query 中不得重复 filter 的条件；比较语句形如 comp(attr, val)（如 gt(attr, val)）；逻辑语句形如 op(statement1, statement2, ...)）；只能使用被允许的比较符与逻辑符、只能引用真实存在的字段；日期必须用 YYYY-MM-DD；无过滤条件时 filter 返回字符串 "NO_FILTER"
   - 对"查找下评分高于9.5分的电影"，LLM 生成的查询语句原文是 {"query": "", "filter": "gt(\"rating\", 9.5)"}（query 为空串，因为该提问没有语义检索成分）
   - 再用"特定向量数据库的转换器"把这段中间表示翻译成该库能识别的过滤器，并在检索时作为参数传入；不同向量库的转换器差异极大，所以必须由框架逐库适配

6. 典型输出示例与观察点
   - 自查询检索器对"评分高于9.5分的电影"返回：
       [Document(metadata={'director': '陈凯歌', 'rating': 9.6, 'year': 1993.0}, page_content='霸王别姬'),
        Document(metadata={'director': '弗兰克·德拉邦特', 'rating': 9.7, 'year': 1994.0}, page_content='肖申克的救赎')]
       2
   - 观察点：只返回 rating 严格大于 9.5 的两部；评分正好等于 9.5 的《阿甘正传》因用的是 gt(严格大于)而非 gte 被正确排除
   - 对照：普通检索器无法理解"高于9.5分"，会把不达标的电影也一并返回

7. 适用性评估
   - 对面向特定领域的专用 Agent 效果较好，对通用 Agent 效果较差
   - 原因：专用领域文档规范，能稳定剥离出可过滤的元数据字段（如财报、新闻、自媒体文章、教培等行业）；通用场景文档结构参差不齐，抽不出统一的过滤字段

8. 应用场景
   - 电商搜索：价格范围、品牌、分类过滤
   - 文档检索：时间范围、作者、文档类型过滤
   - 内容推荐：评分、年份、标签过滤
   - 日志查询：时间、级别、来源过滤

9. 最佳实践
   - 为元数据字段提供清晰的描述，正确设置字段类型（integer / float / string）
   - 使用 enable_limit 限制返回结果数量
   - 对比普通检索器验证过滤效果
   - AttributeInfo.name 与 metadata 键名严格一致，description 写清业务含义

10. 依赖与版本对照
   - 解析过滤表达式依赖 lark 解析库，需额外安装：pip install --upgrade --quiet lark
   - 版本对照：0.x 用 from langchain.chains.query_constructor.base import AttributeInfo / from langchain.retrievers import SelfQueryRetriever；1.x 迁移到 langchain_classic（AttributeInfo 从 base 模块移到更语义化的 schema 模块）
   - 注意 import 路径随 LangChain 1.x 的包拆分而变化

11. 延伸：从自查询到 LLM 对接企业自有系统
   - 自查询体现的思想已触及"如何把 LLM 接入现有业务系统"：让 LLM 按既定参数规则生成一段描述性参数，本地解析后传给现成的、本身不带智能的接口（如 PPT 生成 API），即可快速实现"自然语言 -> PPT"
   - 落地时需处理四个难点：①上下文长度（多次生成参数要连贯）；②prompt 描述（复杂接口参数如何塞进 prompt）；③生成内容转调用参数（函数回调生成的参数数量有限，需把字符串无损转换成真实调用参数）；④参数校验（LLM 输出有随机性，必须校验）
   - 更智能的对接方式（Agent、AI 工作流）见后续章节

===================================================================================
"""
import dotenv
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 加载环境变量配置
dotenv.load_dotenv()

# 1.构建文档列表并上传到数据库
# 每个文档包含电影名称和结构化的元数据（年份、评分、导演）
documents = [
    Document(
        page_content="肖申克的救赎",
        metadata={"year": 1994, "rating": 9.7, "director": "弗兰克·德拉邦特"},
    ),
    Document(
        page_content="霸王别姬",
        metadata={"year": 1993, "rating": 9.6, "director": "陈凯歌"},
    ),
    Document(
        page_content="阿甘正传",
        metadata={"year": 1994, "rating": 9.5, "director": "罗伯特·泽米吉斯"},
    ),
    Document(
        page_content="泰坦尼克号",
        metadat={"year": 1997, "rating": 9.5, "director": "詹姆斯·卡梅隆"},
    ),
    Document(
        page_content="千与千寻",
        metadat={"year": 2001, "rating": 9.4, "director": "宫崎骏"},
    ),
    Document(
        page_content="星际穿越",
        metadat={"year": 2014, "rating": 9.4, "director": "克里斯托弗·诺兰"},
    ),
    Document(
        page_content="忠犬八公的故事",
        metadat={"year": 2009, "rating": 9.4, "director": "莱塞·霍尔斯道姆"},
    ),
    Document(
        page_content="三傻大闹宝莱坞",
        metadat={"year": 2009, "rating": 9.2, "director": "拉库马·希拉尼"},
    ),
    Document(
        page_content="疯狂动物城",
        metadat={"year": 2016, "rating": 9.2, "director": "拜伦·霍华德"},
    ),
    Document(
        page_content="无间道",
        metadat={"year": 2002, "rating": 9.3, "director": "刘伟强"},
    ),
]

# PineconeVectorStore：创建 Pinecone 向量数据库实例
# - index_name：Pinecone 中的索引名称
# - embedding：用于生成文档嵌入向量的模型
# - namespace：用于隔离不同数据集的命名空间
# - text_key：文档内容在向量数据库中的字段名
db = PineconeVectorStore(
    index_name="llmops",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    namespace="dataset",
    text_key="text"
)

# as_retriever：将向量数据库转换为检索器接口
retriever = db.as_retriever()

# add_documents：将文档添加到向量数据库（首次运行时取消注释）
# db.add_documents(documents)

# 2.创建自查询元数据
# AttributeInfo：定义元数据字段的信息，帮助模型理解如何过滤
# - name：字段名称，必须与 metadata 中的键名一致
# - description：字段描述，用于帮助模型理解字段含义
# - type：数据类型，支持 integer、float、string 等
metadata_filed_info = [
    AttributeInfo(name="year", description="电影的年份", type="integer"),
    AttributeInfo(name="rating", description="电影的评分", type="float"),
    AttributeInfo(name="director", description="电影的导演", type="string"),
]

# 3.创建自查询检索
# SelfQueryRetriever.from_llm：从大语言模型创建自查询检索器
# - llm：用于理解用户查询和构建过滤条件的大语言模型
# - vectorstore：底层的向量数据库
# - document_contents：文档内容的语义描述，帮助模型理解内容类型
# - metadata_field_info：元数据字段信息列表
# - enable_limit：是否允许限制返回结果数量
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    vectorstore=db,
    document_contents="电影的名字",
    metadata_field_info=metadata_filed_info,
    enable_limit=True,
)

# 4.检索示例
# 自查询检索器会自动识别"评分高于9.5分"这个过滤条件
# 构建类似 WHERE rating > 9.5 的过滤器
docs = self_query_retriever.invoke("查找下评分高于9.5分的电影")
print(docs)
print(len(docs))

print("===================")

# 对比：普通检索器无法理解过滤条件
# 只能进行语义检索，可能返回不符合评分要求的结果
base_docs = retriever.invoke("查找下评分高于9.5分的电影")
print(base_docs)
print(len(base_docs))
