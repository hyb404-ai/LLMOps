#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 17:30
@Author  : thezehui@gmail.com
@File    : 1.RunnableParallel使用技巧.py

===================================================================================
知识点讲解：RunnableParallel 并行运行组件
===================================================================================

1. 组件定位：为什么需要 RunnableParallel
   - RunnableParallel 是 LangChain 中封装的支持运行多个 Runnable 的核心类，又称 RunnableMap
   - 官方定位：用于「操作 Runnable 的输出，以匹配序列中下一个 Runnable 的输入」，
     起到并行运行 Runnable 并格式化输出结构的作用
   - 这句定义包含两个并列职责，不要只记住「并行」：
     * 职责一（并行执行）：同时执行多条 Chain，再以字典形式返回各个 Chain 的结果，
       对比每一条链单独执行，效率会高很多 —— 本文件示例演示这一职责
     * 职责二（格式化输出结构）：把上游数据整形成下游 Runnable 需要的输入结构，
       典型写法是 context=检索器、query=透传 —— 见 2.RunnableParallel模拟检索.py
   - 类型签名可理解为 Runnable[Input, dict[str, Any]]

2. 构造方式与参数宽容性
   - 支持传递「字典、函数、映射、键值对数据」等多种方式，底层会执行检测并统一转换为 Runnable
   - 四种等价写法：
     * 关键字参数：RunnableParallel(joke=joke_chain, poem=poem_chain)
     * 字典位置参数：RunnableParallel({"joke": joke_chain, "poem": poem_chain})
     * 混合写法：RunnableParallel({"joke": joke_chain}, poem=poem_chain)
     * 隐式写法：LCEL 管道中直接写裸 dict，由 __or__ / __ror__ 自动转换成 RunnableParallel
   - 因此 value 不必是 Runnable 实例，普通函数、lambda、itemgetter 乃至嵌套 dict 都可以

3. 归一化原理：__init__ 源码（langchain-core/runnables/base.py）
   - 源码要点：
       def __init__(self, steps__=None, **kwargs):
           # 1.检测是否传递字典，如果传递，则提取字段内的所有键值对
           merged = {**steps__} if steps__ is not None else {}
           # 2.传递了键值对，则将键值对更新到 merged 进行合并
           merged.update(kwargs)
           super().__init__(
               # 3.循环遍历 merged 的所有键值对，并将每一个元素转换成 Runnable
               steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
           )
   - 由此得到三条确定性结论：
     * 位置参数 steps__ 与关键字参数 kwargs 会 merge 到同一个 dict，
       混合写法合法；键名冲突时因为 update 在后，kwargs 会覆盖 steps__
     * 每个 value 都经过 coerce_to_runnable 转换，这是「参数宽容性」的底层来源
     * 这也解释了关键字写法与字典写法完全等价：二者最终汇入同一个 merged 字典

4. coerce_to_runnable 的类型分发规则（langchain_core/runnables/base.py）
   - 分发顺序决定了「LCEL 里能塞什么」：
     * 已是 Runnable            -> 原样返回
     * 生成器函数 / 异步生成器   -> 包装成 RunnableGenerator
     * 其他可调用对象           -> 包装成 RunnableLambda
       （lambda、普通函数、itemgetter 都命中这一支）
     * dict                     -> 递归包装成 RunnableParallel
     * 其他类型                 -> 直接抛 TypeError
       （报错信息：Expected a Runnable, callable or dict.）
   - 注意分支顺序：callable 的判断在 dict 之前，
     因此对既可调用又是 dict 子类的对象，会优先当作函数处理
   - 与管道运算符的联动：管道里写裸 dict 之所以能工作，
     正是 __or__ / __ror__ 内部调用了 coerce_to_runnable 把 dict 转成了 RunnableParallel
     （__or__ / __ror__ 的说明见 4-LCEL表达式与Runnable可运行协议）

5. 并行执行原理与性能收益
   - invoke 时内部使用 ThreadPoolExecutor 线程池并发调用各分支
   - ainvoke 时使用 asyncio.gather 并发协程
   - LLM 调用属于 IO 密集型（等待网络响应），并行可显著降低总耗时
   - 总耗时 ≈ max(各分支耗时)，而不是 sum(各分支耗时)

6. 输入分发规则（非常重要）
   - RunnableParallel 把「同一个原始输入」原封不动传给每一个分支
   - 本例中 {"subject": "程序员"} 会同时传给 joke_chain 和 poem_chain
   - 因此各分支的 Prompt 变量名必须能从同一份输入里取到（这里都用 {subject}）

7. 与 RunnableSequence 的对比
   - RunnableSequence（管道 |）：串行，上一步输出是下一步输入，A → B → C
   - RunnableParallel（dict）：并行，同一输入分发到多个分支，{A, B, C} 同时跑
   - 二者可以任意嵌套组合，构成复杂的 DAG 数据流

8. 典型输出示例与观察点
   - 并行两条链后得到一个 dict，key 即分支名：
       {'joke': '为什么程序员总是用尺子测电脑屏幕？因为他们听说了"像素"是屏幕上的一种"尺寸"。',
        'poem': '在代码的海洋里徜徉，\n程序员心怀梦想与创意。\n键盘敲击是旋律，\n
                bug 是诗歌的瑕疵。\n\n算法如诗的韵律，\n逻辑是句子的构思。\n
                编程者如诗人般，\n创造出数字的奇迹。'}
   - 观察点 1：两个 value 都已是纯字符串，因为每条分支链末尾都挂了 StrOutputParser；
     若去掉 parser，value 会是 AIMessage 对象
   - 观察点 2：输出 dict 的 key 完全由构造时的分支名决定，与执行完成的先后无关，
     即结果顺序是确定的，不受并行调度影响

9. 典型应用场景
   - 一次输入生成多种形式的结果（笑话 + 诗歌 + 摘要）
   - RAG 场景中同时准备 context（检索）与 query（透传）
   - 多路召回：同时查询向量库、关键词库、知识图谱后再融合
   - 多模型对比评测：同一 prompt 同时发给多个 LLM

10. 注意事项
    - 并行分支会同时发起多个 LLM 请求，注意 API 的 QPS/RPM 限流
    - 任意一个分支抛异常，整个 RunnableParallel 都会失败（可配合 with_fallbacks 兜底）
    - 分支越多，token 消耗越大，成本按分支数线性增长
    - 分支内部若共享可变状态，需要注意线程安全问题

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()

# 1.编排prompt
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：根据字符串模板创建聊天提示词模板
#   参数：template - 模板字符串，{subject} 是变量占位符
#   返回：ChatPromptTemplate 实例，实现 Runnable 协议
#   注意：两个模板使用了同一个变量名 subject，这样才能共享同一份输入
joke_prompt = ChatPromptTemplate.from_template("请讲一个关于{subject}的冷笑话，尽可能短一些")
poem_prompt = ChatPromptTemplate.from_template("请写一篇关于{subject}的诗，尽可能短一些")

# 2.创建大语言模型
# ChatOpenAI(model: str) -> ChatOpenAI
#   作用：创建 OpenAI 兼容协议的聊天模型客户端
#   参数：model - 模型名称
#   返回：ChatOpenAI 实例，invoke 接收消息列表，返回 AIMessage
#   说明：同一个 llm 实例可被多个并行分支安全复用（内部无请求级可变状态）
llm = ChatOpenAI(model="deepseek-flash")

# 3.创建输出解析器
# StrOutputParser() -> StrOutputParser
#   作用：把 AIMessage 解析成纯字符串，等价于取 message.content
#   返回：StrOutputParser 实例，invoke 接收 AIMessage，返回 str
parser = StrOutputParser()

# 4.编排链
# 使用管道操作符 | 串联组件，得到两条独立的 RunnableSequence
#   joke_chain: Runnable[dict, str]，数据流转 dict → messages → AIMessage → str
#   poem_chain: Runnable[dict, str]，数据流转同上
joke_chain = joke_prompt | llm | parser
poem_chain = poem_prompt | llm | parser

# 5.并行链
# RunnableParallel(**steps: Runnable) -> RunnableParallel
#   作用：把多个 Runnable 组合成并行执行的映射（Map）结构
#   参数：以「分支名=Runnable」形式传入，分支名将成为输出 dict 的 key
#         joke=joke_chain  → 输出 dict 的 "joke" 键
#         poem=poem_chain  → 输出 dict 的 "poem" 键
#   返回：RunnableParallel 实例，类型为 Runnable[dict, dict[str, str]]
#   执行方式：内部通过线程池并发调用每个分支的 invoke
#
#   数据流转示意：
#                   ┌──→ joke_chain（joke_prompt | llm | parser）──→ "冷笑话文本"
#   {"subject":...} ┤                                                            ├─→ {"joke":..., "poem":...}
#                   └──→ poem_chain（poem_prompt | llm | parser）──→ "诗歌文本"
map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

# 等价写法一：直接传入字典（当分支名不是合法 Python 标识符时必须用这种写法）
# map_chain = RunnableParallel({
#     "joke": joke_chain,
#     "poem": poem_chain,
# })

# 等价写法二：隐式转换。在 LCEL 管道中直接写 dict，
# LangChain 会自动调用 coerce_to_runnable 把 dict 包装成 RunnableParallel
# map_chain = {"joke": joke_chain, "poem": poem_chain} | RunnablePassthrough()

# map_chain.invoke(input: dict) -> dict[str, str]
#   参数：input - 输入字典，必须包含所有分支模板需要的变量，这里是 subject
#   执行流程：
#     1. 把 {"subject": "程序员"} 同时分发给 joke_chain 与 poem_chain
#     2. 两个分支在独立线程中并发调用 LLM
#     3. 等待全部分支完成后，按分支名汇总结果
#   返回：dict - 形如 {"joke": "冷笑话内容...", "poem": "诗歌内容..."}
#   性能：总耗时约等于较慢的那个分支的耗时，而非两者之和
res = map_chain.invoke({"subject": "程序员"})

# 打印并行结果字典，可通过 res["joke"] / res["poem"] 分别取值
print(res)

# ===================================================================================
# 最佳实践与其他调用方式
# ===================================================================================
#
# 1. 并行结果继续参与后续链路（RunnableParallel 可以接在管道中间）：
#    summary_prompt = ChatPromptTemplate.from_template(
#        "请把下面的笑话和诗合并成一段话：\n笑话：{joke}\n诗：{poem}"
#    )
#    full_chain = map_chain | summary_prompt | llm | parser
#    print(full_chain.invoke({"subject": "程序员"}))
#
# 2. 批量调用（每条输入都会走一遍并行分支）：
#    results = map_chain.batch([{"subject": "程序员"}, {"subject": "产品经理"}])
#
# 3. 流式调用（各分支的 chunk 会以 {"分支名": 增量} 的形式交错产出）：
#    for chunk in map_chain.stream({"subject": "程序员"}):
#        print(chunk)
#
# 4. 异步调用（IO 密集场景下并发效率更高，推荐在 Web 服务中使用）：
#    res = await map_chain.ainvoke({"subject": "程序员"})
#
# 5. 常见错误与解决方案：
#    - 错误：KeyError / Missing variable
#      原因：某个分支的模板变量在输入 dict 中不存在
#      解决：统一各分支的变量名，或先用 RunnablePassthrough.assign 补齐字段
#
#    - 错误：并行分支报 RateLimitError
#      原因：同时发起多个请求触发 API 限流
#      解决：为链配置 max_concurrency，如 config={"max_concurrency": 2}，
#            或使用 .with_retry() 增加重试
#
#    - 错误：误以为 dict 会被串行处理
#      说明：LCEL 中的 dict 一律被解释为 RunnableParallel，是并行而非串行
