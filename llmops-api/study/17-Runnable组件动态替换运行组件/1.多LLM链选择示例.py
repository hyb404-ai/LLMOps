#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 17:10
@Author  : thezehui@gmail.com
@File    : 1.多LLM链选择示例.py

===================================================================================
知识点讲解：configurable_alternatives 动态替换运行组件
===================================================================================

1. 组件定位：运行时组件替换功能
   - 在 LLMOps 项目的应用编排页面中，调试时常常需要"替换大语言模型继续之前的对话"进行调试，
     这就是运行时组件替换功能
   - 它指在构建好的链应用中，动态替换掉特定的模型、提示词等"整个组件本身"，
     而不是替换组件里的参数信息（改参数用 configurable_fields）
   - LangChain 提供 configurable_alternatives() 方法实现，所有 Runnable 组件均支持

2. 典型应用场景
   - 多模型路由：根据任务难度选择 flash（快而便宜）或 pro（强而贵）
   - 多厂商适配：在 OpenAI、文心一言、通义千问之间动态切换
   - 提示词策略切换：为不同用户群体注册不同的 prompt 实现
   - 成本与效果权衡：高价值请求走强模型，普通请求走轻量模型

3. 方法签名与参数含义
   - which: ConfigurableField - 选择器字段，其 id 是运行时的配置键
   - default_key: str - 默认使用的备选项名称，指向调用该方法的原组件
   - **alternatives - 其余备选项，键为选项名，值为对应的 Runnable 实例

4. default_key 的特殊语义
   - default_key 并不新建组件，而是给"调用者自身"起了一个名字
   - 本例中 default_key="deepseek_flash" 指向 ChatOpenAI(model="deepseek-flash")
   - 不传 config 时使用该默认项；传入其他 key 时替换为对应备选组件

5. 底层实现原理
   - 运行流程非常简单：底层通过一个字典 alternatives 存储所有替换组件
   - 从传递的 configurable 字典中获取当前需要选择的组件 key
   - 根据 key 在 alternatives 中找到对应组件返回并执行后续操作
   - 核心代码在 RunnableConfigurableAlternatives._prepare() 方法中：
       which = config.get("configurable", {}).get(self.which.id, self.default_key)
       if which == self.default_key:
           return (self.default, config)
       elif which in self.alternatives:
           alt = self.alternatives[which]
           return (alt, config) if isinstance(alt, Runnable) else (alt(), config)
       else:
           raise ValueError(f"Unknown alternative: {which}")
   - 关键点：提供未注册的 key 会直接抛出 ValueError

6. 组件替换的类型兼容要求
   - 所有备选项必须具备兼容的输入输出类型，才能在同一条链中互换
   - 本例中 ChatOpenAI 与 QianfanChatEndpoint 都接收 Messages、返回 AIMessage
   - 类型不兼容会导致上下游组件报错，需保证接口一致

7. 与 with_fallbacks 的区别（易混淆）
   - configurable_alternatives：主动选择，由调用方通过 config 显式指定
   - with_fallbacks：被动兜底，只有当主组件抛异常时才自动切换（见第 18 章）
   - 两者可组合使用：先按配置选模型，该模型失败时再回退到备用模型

8. 运行流程与数据流转
   - chain.invoke({"query": ...}, config={"configurable": {"llm": "deepseek_pro"}})
   - → 框架读取 configurable.llm = "deepseek_pro"
   - → 从注册表中取出 ChatOpenAI(model="deepseek-v4-pro") 替换链中的 llm 位置
   - → {"query": ...} → prompt → Messages → 被选中的 llm → AIMessage → parser → str

9. 典型输出示例（来自课程文档）
   - 切换到文心一言时，返回内容体现百度模型身份而非默认模型：
       "您好，我是百度研发的知识增强大语言模型，中文名是文心一言，英文名是ERNIE Bot……"
   - 验证点：回复内容随 config 中 llm 的取值变化，说明组件替换在运行时真实生效

===================================================================================
"""
import dotenv
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

# load_dotenv() -> bool
#   作用：加载 .env 文件中的环境变量
#   注意：文心一言需额外配置 QIANFAN_AK / QIANFAN_SK 环境变量
dotenv.load_dotenv()

# 1.创建提示模板&定义默认大语言模型
# ChatPromptTemplate.from_template(template: str) -> ChatPromptTemplate
#   作用：创建仅透传用户输入的聊天提示词模板
#   参数：template - "{query}" 表示直接使用用户输入作为消息内容
#   返回：ChatPromptTemplate 实例
prompt = ChatPromptTemplate.from_template("{query}")

# Runnable.configurable_alternatives(
#     which: ConfigurableField,
#     default_key: str = "default",
#     prefix_keys: bool = False,
#     **kwargs: Runnable | Callable,
# ) -> RunnableConfigurableAlternatives
#   作用：为组件注册多个可运行时切换的备选实现
#   参数：
#     which - ConfigurableField 对象，其 id 是运行时用于选择的配置键
#     default_key - 默认选项的名称，指向调用该方法的组件本身
#     prefix_keys - 是否给备选项键加上 which.id 前缀，默认 False
#     **kwargs - 其他备选项，键为选项名，值为对应的 Runnable 实例
#   返回：RunnableConfigurableAlternatives 实例，实现 Runnable 协议，可参与 | 组合
#   选择规则：
#     - config 未提供 llm 键 → 使用 default_key 对应的组件（deepseek-flash）
#     - config 提供 llm="deepseek_pro" → 使用 ChatOpenAI(model="deepseek-v4-pro")
#     - config 提供 llm="wenxin" → 使用 QianfanChatEndpoint()
#     - 提供未注册的键 → 抛出 ValueError
llm = ChatOpenAI(model="deepseek-flash").configurable_alternatives(
    # ConfigurableField(id: str) -> ConfigurableField
    #   作用：定义选择器字段，id="llm" 即运行时的配置键名
    #   参数：id - 配置键名
    #   返回：ConfigurableField 实例
    ConfigurableField(id="llm"),

    # default_key：为"当前组件自身"命名
    # 即 deepseek_flash 指向 ChatOpenAI(model="deepseek-flash")
    default_key="deepseek_flash",

    # 备选项 1：更强的 DeepSeek Pro 模型（同为 ChatOpenAI 实现）
    deepseek_pro=ChatOpenAI(model="deepseek-v4-pro"),

    # 备选项 2：百度文心一言（不同厂商的实现，但接口兼容）
    # QianfanChatEndpoint() -> QianfanChatEndpoint
    #   作用：创建百度千帆平台的聊天模型实例
    #   依赖：需配置 QIANFAN_AK 与 QIANFAN_SK 环境变量
    #   返回：QianfanChatEndpoint 实例，接收 Messages 返回 AIMessage
    wenxin=QianfanChatEndpoint(),
)

# 2.构建链应用
# LCEL 管道：dict → prompt → Messages → llm（可切换）→ AIMessage → parser → str
# 关键点：链结构在构建时已固定，但 llm 位置具体用哪个模型由运行时 config 决定
chain = prompt | llm | StrOutputParser()

# 3.调用链并传递配置信息，并切换到文心一言模型或者deepseek-v4-pro模型
#
# chain.invoke(input: dict, config: dict = None) -> str
#   作用：执行链，并通过 config 指定使用哪个备选模型
#   参数：
#     input - {"query": 用户问题}
#     config - {"configurable": {"llm": 备选项名称}}
#              键 llm 对应 ConfigurableField(id="llm")
#              值 deepseek_pro 对应注册的备选项名
#   返回：所选模型生成的字符串回复
#   执行流程：
#     1. 框架解析 config.configurable.llm = "deepseek_pro"
#     2. 从备选注册表取出 ChatOpenAI(model="deepseek-v4-pro")
#     3. 用该实例替换链中 llm 的位置
#     4. 依次执行 prompt → 选中的模型 → StrOutputParser
#   验证点：回复内容应体现为 deepseek-v4-pro 而非默认的 deepseek-flash
content = chain.invoke(
    {"query": "你好，你是什么模型呢?"},
    config={"configurable": {"llm": "deepseek_pro"}}
)
print(content)

# ==================== 最佳实践与其他调用方式 ====================
# 1. 使用默认模型（不传 config，走 default_key）
# content = chain.invoke({"query": "你好，你是什么模型呢?"})
# # 实际使用 ChatOpenAI(model="deepseek-flash")
#
# 2. 切换到文心一言
# content = chain.invoke(
#     {"query": "你好，你是什么模型呢?"},
#     config={"configurable": {"llm": "wenxin"}}
# )
#
# 3. 用 with_config 派生固定模型的链变体（便于复用与注入）
# flash_chain = chain.with_config(configurable={"llm": "deepseek_flash"})
# pro_chain = chain.with_config(configurable={"llm": "deepseek_pro"})
# wenxin_chain = chain.with_config(configurable={"llm": "wenxin"})
#
# 4. 同时切换组件与调整组件内部参数（alternatives + fields 组合）
# llm = ChatOpenAI(model="deepseek-flash").configurable_fields(
#     temperature=ConfigurableField(id="llm_temperature")
# ).configurable_alternatives(
#     ConfigurableField(id="llm"),
#     default_key="deepseek_flash",
#     deepseek_pro=ChatOpenAI(model="deepseek-v4-pro"),
# )
# content = chain.invoke(
#     {"query": "..."},
#     config={"configurable": {"llm": "deepseek_pro", "llm_temperature": 0}}
# )
#
# 5. 同时替换提示词与模型（多个可切换点共存）
# prompt = ChatPromptTemplate.from_template("{query}").configurable_alternatives(
#     ConfigurableField(id="prompt"),
#     default_key="default",
#     translate=ChatPromptTemplate.from_template("请将以下内容翻译为英文：{query}"),
# )
# content = chain.invoke(
#     {"query": "你好"},
#     config={"configurable": {"prompt": "translate", "llm": "deepseek_pro"}}
# )
#
# 6. 配合 with_fallbacks 实现"主动选择 + 被动兜底"双保险
# robust_llm = llm.with_fallbacks([QianfanChatEndpoint()])
# chain = prompt | robust_llm | StrOutputParser()
# # 按 config 选模型；若该模型调用失败，自动回退到文心一言
#
# 7. 查看链上所有可配置项（调试用）
# print(chain.config_specs)
