#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 15:00
@Author  : thezehui@gmail.com
@File    : 1.自定义检索器使用技巧.py

===================================================================================
知识点讲解：自定义检索器（BaseRetriever 继承）
===================================================================================

1. 为什么需要自定义检索器（内置检索器全景与局限）
   - LangChain 内置了大量第三方检索器：维基百科搜索、Weaviate 混合搜索、Zep 检索器、ES 搜索等
     （清单 https://imooc-langchain.shortvar.com/docs/integrations/retrievers/）
   - 但这些内置检索器开发较早，多在 Runnable 与 LCEL 出现之前发布，部分并非 Runnable 可运行组件，
     用法也有差异，使用前需查文档
   - 当内置检索器满足不了需求时就要自定义，典型场景：
     * 对接公司内部搜索服务 / ES / 自研召回引擎（如百度、慕课网搜索）
     * 结合业务规则做权限过滤、时间衰减、人工加权
     * 组合多路召回并自定义融合逻辑
   - 只要继承 BaseRetriever 并实现 _get_relevant_documents，即获得完整 Runnable 能力
   - 课程提示：检索器模块正常只需重点掌握向量数据库检索器，其他检索器会用即可；
     尤其网络/爬虫类检索正被「工具回调」逐步替代

2. BaseRetriever 的设计（模板方法模式）
   - BaseRetriever 继承自 RunnableSerializable，已实现 invoke / ainvoke / batch / stream
   - 子类只需实现受保护的 _get_relevant_documents()，公开 invoke() 由父类调用并附加回调、追踪、异常包装
   - 因此绝不要重写 invoke()，只重写 _get_relevant_documents()

3. _get_relevant_documents 方法签名解析
     def _get_relevant_documents(
         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
     ) -> List[Document]
   - query：检索查询字符串（Retriever 的输入固定是 str）
   - `*` 之后的参数强制按关键字传递，避免位置参数错位
   - run_manager：回调管理器，用于上报检索过程事件；若内部还要调用其他 Runnable，
     应通过 run_manager.get_child() 传递回调，这样 LangSmith 上能看到完整的嵌套调用树
   - 返回值必须是 List[Document]，这是检索器的契约

4. BaseRetriever 基于 Pydantic 的字段声明
   - BaseRetriever 本质是 Pydantic 模型，类属性即构造参数
   - 本例声明 documents: list[Document] 与 k: int，
     实例化 CustomRetriever(documents=..., k=3) 会自动完成类型校验
   - 好处：参数类型安全、自动序列化、可被 configurable_fields 等机制识别

5. 同步与异步实现
   - 只实现 _get_relevant_documents 时，ainvoke 会用线程池自动兜底（可用但非真异步）
   - 若内部是 IO 密集操作（HTTP、数据库），建议同时实现
     async def _aget_relevant_documents(...)，获得真正的并发收益

6. 本例的关键词检索逻辑与其边界
   - 逻辑：遍历预设文档，用 query.lower() in page_content.lower() 做子串包含匹配，
     命中即收集，收集数量超过 k 就提前返回
   - 这是最朴素的关键词匹配，不涉及任何向量计算，零成本、可解释性强
   - 局限：无法处理同义词（"猫" 匹配不到 "喵星人"）、无相关性排序、
     无法理解语义（"宠物" 匹配不到 "猫咪"）
   - 实现细节：判断条件是 len(...) > self.k 而非 >= self.k，k=3 时最多返回 4 条；
     严格按 k 截断应写 >= self.k，或直接用 matching_documents[:self.k]
   - 典型输出：query="猫" 命中 page=1、page=3 返回 2 条；page=10 讲狗字面不含"猫"被漏召回——
     这正说明纯字面匹配与语义检索的差异，是引入向量语义检索与混合检索的动机

7. 最佳实践
   - 只重写 _get_relevant_documents，永远不要重写 invoke（会丢失回调与追踪能力）
   - 内部调用其他 Runnable 时，务必传 config={"callbacks": run_manager.get_child()}，保持调用链可观测
   - 用 Pydantic 字段声明所有依赖（检索器、模型、参数），而不是在 __init__ 里硬编码
   - 边界条件（k 的截断、query 为空、无命中）要显式处理，避免返回 None 或抛异常
   - 关键词检索适合做"精确术语命中"，与向量检索互补；
     生产环境建议用 EnsembleRetriever 把两者融合（详见第 45 节混合检索）
   - IO 密集型自定义检索器请补齐 _aget_relevant_documents，避免异步链被阻塞

===================================================================================
"""
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class CustomRetriever(BaseRetriever):
    """自定义检索器"""

    # 【Pydantic 字段声明】
    # documents: 检索的数据源，实例化时必须传入的预设文档列表
    # k:         最多返回的文档条数上限
    # 这两个字段由 BaseRetriever 的 Pydantic 基类自动转为构造参数并做类型校验
    documents: list[Document]
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """根据传入的query，获取相关联的文档列表

        这是 BaseRetriever 要求子类实现的唯一抽象方法，
        外部调用 retriever.invoke(query) 时，父类会在附加好回调与追踪后调用本方法。

        参数：
            query: str - 检索查询字符串
            run_manager: CallbackManagerForRetrieverRun - 回调管理器（关键字参数）
                         用于上报检索事件；若内部要调用其他 Runnable，
                         应通过 run_manager.get_child() 传递，保持调用链完整

        返回：
            List[Document] - 匹配到的文档列表

        执行流程：
            1) 初始化空的结果容器
            2) 顺序遍历所有预设文档
            3) 先判断是否已达数量上限，达到则提前返回（短路优化）
            4) 统一转小写做子串包含匹配，命中则收集
            5) 遍历结束后返回全部命中结果
        """
        # 结果容器，用于存放命中的文档
        matching_documents = []

        # 顺序遍历数据源中的每一篇文档
        for document in self.documents:
            # 数量上限检查（短路返回）：
            # 一旦已收集的文档数超过 k，立即返回，避免无意义的后续遍历。
            # 注意这里是 > self.k 而非 >= self.k，
            # 因此在文档充足的情况下实际最多返回 k+1 条。
            if len(matching_documents) > self.k:
                return matching_documents

            # 关键字匹配：
            # 双方统一 .lower() 实现大小写不敏感的子串包含判断。
            # query.lower() in document.page_content.lower()
            #   → 只要查询词作为子串出现在正文中即视为命中
            # 局限：纯字面匹配，不具备语义理解与同义词扩展能力
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)

        # 遍历完成，返回所有命中的文档
        # 若一条都没命中，返回空列表（这是合法结果，上层应做兜底处理）
        return matching_documents


# 1.定义预设文档
# Document(page_content: str, metadata: dict) -> Document
#   作用：构造 LangChain 标准文档对象
#   参数：page_content - 正文内容，本例中作为关键词匹配的目标文本
#         metadata     - 元数据，本例用 page 标记页码，便于溯源
#   数据特征：包含 "猫" 字的有 page=1（笨笨是猫咪）和 page=3（猫咪打盹），
#             page=10 是"狗"，字面上不含"猫"，因此关键词检索无法命中——
#             这正是纯字面匹配与语义检索的差异所在
documents = [
    Document(page_content="笨笨是一只很喜欢睡觉的猫咪", metadata={"page": 1}),
    Document(page_content="我喜欢在夜晚听音乐，这让我感到放松。", metadata={"page": 2}),
    Document(page_content="猫咪在窗台上打盹，看起来非常可爱。", metadata={"page": 3}),
    Document(page_content="学习新技能是每个人都应该追求的目标。", metadata={"page": 4}),
    Document(page_content="我最喜欢的食物是意大利面，尤其是番茄酱的那种。", metadata={"page": 5}),
    Document(page_content="昨晚我做了一个奇怪的梦，梦见自己在太空飞行。", metadata={"page": 6}),
    Document(page_content="我的手机突然关机了，让我有些焦虑。", metadata={"page": 7}),
    Document(page_content="阅读是我每天都会做的事情，我觉得很充实。", metadata={"page": 8}),
    Document(page_content="他们一起计划了一次周末的野餐，希望天气能好。", metadata={"page": 9}),
    Document(page_content="我的狗喜欢追逐球，看起来非常开心。", metadata={"page": 10}),
]

# 2.创建检索器
# CustomRetriever(documents: list[Document], k: int) -> CustomRetriever
#   作用：实例化自定义检索器
#   参数：documents - 数据源文档列表，赋值给 Pydantic 字段 documents
#         k         - 返回条数上限，赋值给 Pydantic 字段 k
#   返回：CustomRetriever 实例，因继承 BaseRetriever，天然具备
#         invoke / ainvoke / batch / stream 以及 LCEL 管道组合能力
#   说明：这里没有写 __init__，构造与类型校验完全由 Pydantic 自动完成
retriever = CustomRetriever(documents=documents, k=3)

# 3.调用检索器获取搜索结果并打印
# retriever.invoke(input: str, config: RunnableConfig = None) -> list[Document]
#   作用：执行检索，Runnable 协议的统一入口
#   参数：input - 查询字符串 "猫"
#   返回：匹配到的 Document 列表
#   内部执行流程（数据流转）：
#     invoke("猫")
#       → BaseRetriever.invoke() 创建 run_manager 并触发 on_retriever_start 回调
#       → 调用 self._get_relevant_documents("猫", run_manager=run_manager)
#       → 方法内遍历 10 条文档做子串匹配，命中 page=1 与 page=3
#       → 触发 on_retriever_end 回调
#       → 返回 Document 列表
retriever_documents = retriever.invoke("猫")

# 打印命中的完整文档信息（含 page_content 与 metadata）
print(retriever_documents)

# 打印命中条数
# 预期为 2：只有 page=1、page=3 的正文中字面包含"猫"字；
# page=10 讲的是狗，语义上是"宠物"但字面不含"猫"，关键词检索必然漏掉，
# 这个漏召回的例子正是引入向量语义检索与混合检索的动机
print(len(retriever_documents))
