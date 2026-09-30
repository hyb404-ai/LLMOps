#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/01 19:53
@Author  : thezehui@gmail.com
@File    : 1.回调功能使用技巧.py

===================================================================================
知识点讲解：Callback 回调机制与链路可观测性
===================================================================================

1. 什么是 Callback（回调）
   - LangChain 提供的「事件钩子」机制，在链执行的各个生命周期节点触发回调方法
   - 作用：让链的执行过程从黑盒变成透明，可用于日志、监控、计费、调试
   - 不侵入业务代码：回调通过 config 传入，无需修改链本身的结构

2. BaseCallbackHandler 的核心事件（按组件分类）
   - LLM / ChatModel 类：
       on_llm_start          普通 LLM 开始执行
       on_chat_model_start   聊天模型开始执行（ChatOpenAI 走的是这个）
       on_llm_new_token      流式输出时每产生一个 token 触发
       on_llm_end            模型执行结束，携带完整的 LLMResult
       on_llm_error          模型执行出错
   - Chain 类：on_chain_start / on_chain_end / on_chain_error
   - Tool 类：on_tool_start / on_tool_end / on_tool_error
   - Retriever 类：on_retriever_start / on_retriever_end / on_retriever_error
   - 其他：on_text、on_agent_action、on_agent_finish

3. on_chat_model_start 与 on_llm_start 的区别（常见踩坑点）
   - ChatModel（如 ChatOpenAI）触发的是 on_chat_model_start，不是 on_llm_start
   - 若只实现 on_llm_start，使用 ChatOpenAI 时回调不会被触发
   - 但结束事件是共用的：ChatModel 结束时触发的仍是 on_llm_end

4. 回调方法的通用参数说明
   - serialized: Dict - 组件的序列化信息（类名、模块路径、参数等）
   - run_id: UUID - 本次运行的唯一标识，可用于串联同一次调用的各个事件
   - parent_run_id: Optional[UUID] - 父级运行 ID，据此可还原出完整的调用树
   - tags / metadata - 通过 config 传入的自定义标签与元数据，便于分类统计
   - **kwargs - 预留扩展参数，子类实现必须保留，否则新版本新增参数时会报错

5. 回调的三种注册方式（作用域不同）
   - 请求级：chain.invoke(input, config={"callbacks": [handler]})，仅本次调用生效
   - 组件级：ChatOpenAI(model=..., callbacks=[handler])，该组件的所有调用都生效
   - 全局级：langchain_core.globals.set_debug(True) 或环境变量 LANGCHAIN_TRACING_V2

6. 内置回调处理器
   - StdOutCallbackHandler：把链的进入/退出信息打印到标准输出，最简单的调试工具
   - ConsoleCallbackHandler：输出更详细的彩色树形结构日志
   - FileCallbackHandler：把回调日志写入文件
   - LangSmith Tracer：上报到 LangSmith 平台，提供完整可视化追踪

7. 本示例实现的 LLMOps 场景
   - on_chat_model_start 记录起始时间戳 self.start_at，并打印输入消息
   - on_llm_end 计算 end_at - start_at 得到本次调用耗时
   - 这是 LLMOps 平台统计「响应延迟、token 消耗、成本」的基础能力

8. 注意事项
   - Handler 实例中保存状态（如 start_at）在并发场景下会互相覆盖，
     生产环境应以 run_id 为 key 存入字典，或每次请求新建 Handler 实例
   - 回调方法中的异常默认会被吞掉（raise_error=False），调试时需注意
   - 流式调用必须消费完生成器，on_llm_end 才会被触发（本例末尾的 for 循环即为此目的）

===================================================================================
"""
import time
from typing import Dict, Any, List, Optional
from uuid import UUID

import dotenv
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()


class LLMOpsCallbackHandler(BaseCallbackHandler):
    """
    自定义LLMOps回调处理器

    设计目的：
        统计单次大模型调用的耗时，并记录输入消息与完整输出，
        这是 LLMOps 平台做性能监控、成本核算的最小可用实现。

    继承关系：
        BaseCallbackHandler 定义了全部生命周期钩子的默认空实现，
        子类按需覆写关心的事件方法即可，无需实现全部方法。
    """
    # 实例级状态：记录模型开始执行的时间戳（单位：秒）
    # 注意：并发场景下多个请求共用同一个 Handler 实例会互相覆盖，
    #       生产环境建议改为 dict[run_id, float] 或每次请求新建实例
    start_at: float = 0

    def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        """
        on_chat_model_start(...) -> Any
          作用：聊天模型（ChatModel）开始执行时触发的钩子

          参数：
            serialized: Dict[str, Any] - 模型的序列化信息，包含 id（类路径）、
                                         kwargs（模型参数如 model 名称、temperature）
            messages: List[List[BaseMessage]] - 发送给模型的消息批次。
                                         外层 list 对应 batch 中的每条输入，
                                         内层 list 是该条输入的消息序列
            run_id: UUID - 本次模型运行的唯一标识
            parent_run_id: Optional[UUID] - 父级运行 ID（这里是外层 chain 的 run_id）
            tags: Optional[List[str]] - config 中传入的标签列表
            metadata: Optional[Dict[str, Any]] - config 中传入的元数据
            **kwargs: Any - 预留扩展参数，必须保留以兼容未来版本

          返回：Any - 返回值不被框架使用，通常返回 None

          副作用：记录开始时间戳到 self.start_at，供 on_llm_end 计算耗时

          重要提示：ChatOpenAI 触发的是本方法，而不是 on_llm_start，
                   只实现 on_llm_start 会导致回调静默失效
        """
        print("聊天模型开始执行了")
        # 打印序列化信息：可看到模型类名与初始化参数
        print("serialized:", serialized)
        # 打印实际发送给模型的消息，用于核对 Prompt 是否渲染正确
        print("messages:", messages)
        # 记录起始时间戳，用于后续计算耗时
        self.start_at = time.time()

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """
        on_llm_end(response: LLMResult, ...) -> Any
          作用：LLM 与 ChatModel 执行结束时共用的钩子（结束事件不区分两种模型）

          参数：
            response: LLMResult - 模型的完整输出结果对象，关键字段：
                        - generations: List[List[Generation]] 生成结果的二维列表
                        - llm_output: dict 模型返回的附加信息，
                                      通常包含 token_usage（prompt_tokens /
                                      completion_tokens / total_tokens）与 model_name
                        - run: 运行信息列表
            run_id: UUID - 本次运行的唯一标识，与 on_chat_model_start 的 run_id 一致
            parent_run_id: Optional[UUID] - 父级运行 ID
            **kwargs: Any - 预留扩展参数

          返回：Any - 返回值不被框架使用

          副作用：打印完整响应与本次调用耗时

          触发时机：流式调用时，必须把生成器消费完毕本事件才会触发
        """
        # 记录结束时间戳
        end_at: float = time.time()
        # 打印完整的 LLMResult，可从 response.llm_output 中提取 token 用量做计费
        print("完整输出:", response)
        # 耗时 = 结束时间 - 开始时间，即本次模型调用的端到端延迟
        print("程序消耗:", end_at - self.start_at)


# 1.编排prompt
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建仅含单个变量的最简提示词模板
#   参数：template - "{query}" 表示直接把用户输入作为提示词
#   返回：ChatPromptTemplate 实例
prompt = ChatPromptTemplate.from_template("{query}")

# 2.创建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建聊天模型实例
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例
#   补充：也可以在这里传 callbacks=[...] 做组件级回调绑定，
#         那样该模型的每次调用都会触发回调，而不必每次 invoke 都传 config
llm = ChatOpenAI(model="deepseek-flash")

# 3.构建链
# {"query": RunnablePassthrough()} 的作用：
#   把「裸字符串输入」包装成 prompt 需要的 dict 结构，
#   使得调用方可以写 chain.stream("你好，你是？") 而不是 chain.stream({"query": "..."})
#
#   RunnablePassthrough() -> RunnablePassthrough
#     作用：原样返回输入，invoke(x) == x
#     在此处：把传入的字符串直接填到 "query" 键上
#
#   该 dict 会被 LCEL 隐式转换为 RunnableParallel
#
# 数据流转：
#   "你好，你是？" → {"query": "你好，你是？"} → prompt → messages
#                 → llm → AIMessageChunk 流 → StrOutputParser → str 流
chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 4.调用链并执行
# chain.stream(input: Any, config: RunnableConfig) -> Iterator[str]
#   参数：
#     input: str - 由于链首做了 RunnablePassthrough 包装，这里可直接传字符串
#     config: dict - 运行时配置，关键字段：
#       - callbacks: List[BaseCallbackHandler] - 回调处理器列表，按顺序依次触发
#           * StdOutCallbackHandler()：内置处理器，打印链的进入/退出信息
#           * LLMOpsCallbackHandler()：自定义处理器，统计耗时与输入输出
#       - 其他可选字段：tags（标签）、metadata（元数据）、
#                       run_name（运行名称）、max_concurrency（并发上限）
#   返回：Iterator[str] - 生成器，逐块产出解析后的字符串增量
#
#   回调触发顺序：
#     on_chain_start（RunnableSequence）
#       → on_chain_start（RunnableParallel）→ on_chain_end
#       → on_chain_start（prompt）→ on_chain_end
#       → on_chat_model_start → on_llm_new_token × N → on_llm_end
#       → on_chain_start（parser）→ on_chain_end
#     → on_chain_end（RunnableSequence）
resp = chain.stream(
    "你好，你是？",
    config={"callbacks": [StdOutCallbackHandler(), LLMOpsCallbackHandler()]}
)

# 5.消费生成器
# 关键点：stream() 返回的是惰性生成器，不遍历就不会真正发起请求，
#        也就不会触发 on_llm_end 回调。这里的空循环仅用于「把流跑完」，
#        以便观察完整的回调事件序列。
# 实际业务中应在循环体内输出内容，例如 print(chunk, end="", flush=True)
for chunk in resp:
    pass

# ===================================================================================
# 最佳实践与其他调用方式
# ===================================================================================
#
# 1. 组件级绑定回调（该组件所有调用都生效，无需每次传 config）：
#    llm = ChatOpenAI(model="deepseek-flash", callbacks=[LLMOpsCallbackHandler()])
#
# 2. 使用 with_config 为链预置回调，得到一个已配置好的新链：
#    chain_with_cb = chain.with_config({"callbacks": [LLMOpsCallbackHandler()]})
#    chain_with_cb.invoke("你好")
#
# 3. 配合 tags / metadata 做分类统计（回调方法中可从入参读取）：
#    chain.invoke("你好", config={
#        "callbacks": [LLMOpsCallbackHandler()],
#        "tags": ["chat", "prod"],
#        "metadata": {"user_id": "u_1001", "app_id": "app_88"},
#        "run_name": "chat_completion",
#    })
#
# 4. 统计 token 消耗做成本核算（在 on_llm_end 中读取）：
#    def on_llm_end(self, response, **kwargs):
#        usage = (response.llm_output or {}).get("token_usage", {})
#        print("prompt_tokens:", usage.get("prompt_tokens"))
#        print("completion_tokens:", usage.get("completion_tokens"))
#
# 5. 逐 token 统计首字延迟（TTFT，衡量流式体验的核心指标）：
#    def on_llm_new_token(self, token: str, **kwargs):
#        if self.first_token_at == 0:
#            self.first_token_at = time.time()
#
# 6. 异步回调：继承 AsyncCallbackHandler，实现 async def on_xxx 方法，
#    配合 ainvoke / astream 使用，避免阻塞事件循环
#
# 7. 常见错误与解决方案：
#    - 问题：回调完全没有触发
#      原因一：使用 ChatModel 却只实现了 on_llm_start
#      解决：改为实现 on_chat_model_start
#      原因二：stream() 返回的生成器没有被消费
#      解决：遍历生成器，或改用 invoke()
#
#    - 问题：并发请求时耗时统计错乱
#      原因：Handler 实例状态 start_at 被多个请求共享覆盖
#      解决：改用 self.starts: dict[UUID, float]，以 run_id 为键隔离
#
#    - 问题：自定义 Handler 在新版 LangChain 中报 TypeError
#      原因：覆写方法时没有保留 **kwargs，新增的框架参数无法接收
#      解决：方法签名末尾始终保留 **kwargs: Any
