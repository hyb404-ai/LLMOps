#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 15:05
@Author  : thezehui@gmail.com
@File    : 1.configurable_fields使用技巧.py

===================================================================================
知识点讲解：configurable_fields 运行时配置链内部组件参数
===================================================================================

1. configurable_fields 要解决的问题
   - 链一旦通过 | 组合完成，内部组件的参数就被固定了
   - 实际业务中常需要按请求动态调整（如不同用户使用不同 temperature）
   - 重复构建多套链代价高，configurable_fields 提供了运行时改参的能力
   - 与 bind() 的对比：bind() 是在构建时指定参数，configurable_fields() 是运行时才指定

2. configurable_fields 的工作机制与底层原理
   - 在组件上声明"哪些字段允许运行时配置"，并为每个字段分配唯一 id
   - 返回 RunnableConfigurableFields 对象，替代原组件参与链的组合
   - 调用时通过 config={"configurable": {id: value}} 传入新值
   - 框架在执行前根据 config 动态创建一个参数被替换的组件副本
   - 底层实现：调用 invoke() 时会先调用 _prepare() 预处理函数，该函数依据
     原有参数 + 配置的参数重新创建对应的组件进行覆盖

3. ConfigurableField 的三个参数
   - id: str - 配置键名（必填），是运行时 configurable 字典的键
   - name: str - 人类可读的字段名称（可选），用于 UI 展示或文档
   - description: str - 字段说明（可选），解释该参数的作用与取值影响

4. configurable_fields 与 bind() 的适用范围对比
   - configurable_fields() 和 bind() 非常接近，但可配置范围更广
   - 只要 Runnable 组件下有的所有属性，都可以通过 configurable_fields() 进行配置
   - bind() 只能配置**调用参数**（一般调用参数都和组件参数有关系）
   - 可通过 Runnable.__fields__.keys() 查看 configurable_fields() 支持配置哪些字段
     （父类属性也可以配置，但在这里不显示）

5. 两种传递配置的方式对比
   - with_config(configurable={...})：返回配置好的新链，适合复用同一配置多次调用
   - invoke(input, config={"configurable": {...}})：单次调用生效，适合每次都不同
   - 两者效果等价，前者更适合"配置固定、多次调用"的场景

6. temperature 参数的实际含义
   - 取值范围通常为 0 到 2，控制模型输出的随机性
   - temperature=0：输出最确定，相同输入几乎总是相同输出（适合抽取、分类）
   - temperature 较高：输出更多样、更有创造性（适合创作、头脑风暴）
   - 本示例通过生成随机整数来直观体现温度对输出稳定性的影响

7. configurable_fields 与 bind 的选择原则
   - bind()：构建链时固定参数，运行时不可变，写法简单
   - configurable_fields()：声明可配置点，运行时按需传值，灵活但需先声明
   - 只有预先声明为 configurable 的字段才能在运行时被修改

8. 数据流转与配置注入过程
   - chain.with_config(configurable={"llm_temperature": 0})
   - → 生成携带配置的新链（原链不受影响）
   - → invoke({"x": 1000}) 时，框架读取 configurable
   - → 用 temperature=0 替换 llm 的原始 temperature，生成组件副本
   - → {"x": 1000} → prompt → llm(temperature=0) → parser → str

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量（API Key 等）
dotenv.load_dotenv()

# 1.创建提示模板
# PromptTemplate.from_template(template: str) -> PromptTemplate
#   作用：创建纯文本格式的提示词模板（非聊天消息格式）
#   参数：template - 含变量占位符的模板字符串
#   返回：PromptTemplate 实例，invoke 后得到 StringPromptValue
#   说明：与 ChatPromptTemplate 的区别是输出为单一字符串而非消息列表
prompt = PromptTemplate.from_template("请生成一个小于{x}的随机整数")

# 2.创建LLM大语言模型，并配置temperature参数为可在运行时配置，配置键位llm_temperature
#
# Runnable.configurable_fields(**kwargs: ConfigurableField) -> RunnableConfigurableFields
#   作用：声明该组件的哪些字段允许在运行时通过 config 动态修改
#   参数：**kwargs - 键为组件的字段名（如 temperature），值为 ConfigurableField 描述对象
#         字段名必须是该组件真实存在的属性，否则会抛出异常
#   返回：RunnableConfigurableFields 实例，仍实现 Runnable 协议，可参与 | 组合
#   注意：不修改原 ChatOpenAI 对象，返回的是包装后的新对象
#
# ConfigurableField(id: str, name: str = None, description: str = None) -> ConfigurableField
#   作用：描述一个可运行时配置的字段
#   参数：
#     id - 配置键名（必填），运行时通过 config={"configurable": {id: value}} 传值
#     name - 字段的显示名称（可选），便于在管理界面展示
#     description - 字段说明（可选），解释参数含义与取值影响
#   返回：ConfigurableField 实例，作为字段元信息描述
llm = ChatOpenAI(model="deepseek-v4-pro").configurable_fields(
    # 将 temperature 字段声明为可运行时配置，配置键为 llm_temperature
    temperature=ConfigurableField(
        id="llm_temperature",
        name="大语言模型的温度",
        description="温度越低，大语言模型生成的内容越确定，温度越高，生成内容越随机"
    )
)

# 3.构建链应用
# LCEL 管道：dict → prompt → StringPromptValue → llm → AIMessage → parser → str
# 说明：链中的 llm 是 RunnableConfigurableFields，具备运行时改参能力
chain = prompt | llm | StrOutputParser()

# 4.正常调用内容
# chain.invoke(input: dict) -> str
#   作用：使用默认配置执行链（temperature 为 ChatOpenAI 的默认值，通常为 0.7）
#   参数：input - {"x": 1000}，填充模板中的 {x} 变量
#   返回：模型生成的字符串
#   特点：由于温度较高，多次执行结果可能不同
content = chain.invoke({"x": 1000})
print(content)

print("===========================")

# 5.将temperature修改为0调用内容
#
# Runnable.with_config(config: dict = None, **kwargs) -> RunnableBinding
#   作用：为链绑定运行时配置，返回携带该配置的新链
#   参数：configurable - 字典，键为 ConfigurableField 的 id，值为要设置的新值
#   返回：新的 Runnable，原 chain 保持不变（不可变设计）
#   优势：配置一次即可多次 invoke，避免每次调用都重复传 config
#   执行时行为：框架用 temperature=0 创建 llm 副本，再执行整条链
with_config_chain = chain.with_config(configurable={"llm_temperature": 0})

# with_config_chain.invoke(input: dict) -> str
#   作用：以 temperature=0 执行链
#   参数：input - {"x": 1000}
#   返回：模型生成的字符串
#   特点：temperature=0 时输出高度确定，多次执行结果通常一致
content = with_config_chain.invoke({"x": 1000})

# 等价写法：在单次 invoke 时直接传入 config（作用范围仅限本次调用）
#   适用场景：每次调用的配置都不相同，无需复用
# content = chain.invoke(
#     {"x": 1000},
#     config={"configurable": {"llm_temperature": 0}}
# )
print(content)

# ==================== 最佳实践与扩展写法 ====================
# 1. 声明多个可配置字段（同时开放 temperature 与 max_tokens）
# llm = ChatOpenAI(model="deepseek-v4-pro").configurable_fields(
#     temperature=ConfigurableField(id="llm_temperature", name="温度"),
#     max_tokens=ConfigurableField(id="llm_max_tokens", name="最大输出长度"),
#     model_name=ConfigurableField(id="llm_model", name="模型名称"),
# )
# content = chain.invoke(
#     {"x": 1000},
#     config={"configurable": {
#         "llm_temperature": 0,
#         "llm_max_tokens": 128,
#         "llm_model": "deepseek-flash",
#     }}
# )
#
# 2. 派生多个预设配置的链变体（利用 with_config 的不可变特性）
# deterministic_chain = chain.with_config(configurable={"llm_temperature": 0})
# creative_chain = chain.with_config(configurable={"llm_temperature": 1.5})
#
# 3. 查看链支持的所有可配置项（便于调试与文档生成）
# print(chain.config_specs)
#
# 4. 与 bind 的选择建议
#   - 参数在构建时即可确定且永不变化 → 用 bind()
#   - 参数需按请求/用户动态变化 → 用 configurable_fields()
#   - 需要整体替换组件（而非改参数）→ 用 configurable_alternatives()（见第 17 章）
#
# 5. 流式调用同样支持配置传递
# for chunk in chain.stream({"x": 1000}, config={"configurable": {"llm_temperature": 0}}):
#     print(chunk, end="", flush=True)
