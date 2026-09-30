#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/4 15:22
@Author  : thezehui@gmail.com
@File    : 2.configurable_fields替换提示词.py

===================================================================================
知识点讲解：configurable_fields 运行时替换提示词模板
===================================================================================

1. 提示词模板也是 Runnable，同样支持运行时配置
   - configurable_fields 不是 LLM 专属能力，任何 Runnable 都可使用
   - PromptTemplate 的 template 字段可被声明为运行时可配置
   - 这使得"同一条链、不同提示词策略"成为可能，无需重建链

2. 运行时替换提示词的典型场景
   - A/B 测试：对比不同提示词写法的效果，链结构保持不变
   - 多任务复用：同一变量（如 subject）配合不同指令（冷笑话/藏头诗/摘要）
   - 多语言支持：根据用户语言偏好切换中英文提示词
   - 提示词热更新：从数据库读取最新模板，无需重启服务

3. 可配置字段名必须是组件的真实属性名（本文件的已知 Bug）
   - PromptTemplate 的模板字符串字段名为 template（不是 template_demo）
   - 若传入不存在的字段名，configurable_fields 会立即抛出 ValueError
   - 可通过 print(prompt.model_fields.keys()) 查看全部可配置的字段名
   - 【本文件 Bug】：原代码使用 template_demo=ConfigurableField(...) 是错误的
   - 正确写法应为 template=ConfigurableField(...)
   - 该 Bug 会在第 77-79 行立即抛出：ValueError: Configuration key template_demo not found
   - 保留该错误写法是为了演示字段名错误的后果，正确写法见文末注释

4. 替换 template 时的变量兼容性要求
   - 新模板中的变量应与原模板保持一致（本例都使用 {subject}）
   - 若新模板引入了原模板没有的变量，invoke 时会因缺少入参而报错
   - 若新模板缺少原模板的变量，多余入参通常被忽略（取决于版本行为）

5. configurable_fields 的三种典型应用对象
   - PromptTemplate.template：替换提示词文本（本文件）
   - ChatOpenAI.temperature / model_name：调整模型参数（见本章第 1 节）
   - Retriever.search_kwargs：动态调整检索的 Top-K 数量

6. 与 configurable_alternatives 的区别
   - configurable_fields：修改组件内部的某个字段值（细粒度改参）
   - configurable_alternatives：整体替换为另一个预注册的组件（粗粒度换件）
   - 替换提示词文本用前者，切换到完全不同类型的 prompt 用后者

7. 数据流转过程（假设使用正确的字段名 template）
   - prompt.invoke({"subject": "程序员"}, config={"configurable": {"prompt_template": 新模板}})
   - → 框架读取 configurable，用新模板字符串重建 PromptTemplate 副本
   - → 副本填充 subject 变量 → StringPromptValue
   - → to_string() → "请写一篇关于程序员主题的藏头诗"

===================================================================================
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

# 1.创建提示模板并配置支持动态配置的字段
#
# PromptTemplate.from_template(template: str) -> PromptTemplate
#   作用：创建纯文本提示词模板
#   参数：template - 含 {subject} 占位符的模板字符串
#   返回：PromptTemplate 实例
#
# Runnable.configurable_fields(**kwargs: ConfigurableField) -> RunnableConfigurableFields
#   作用：声明该组件允许在运行时被 config 覆盖的字段
#   参数：**kwargs - 键必须是组件的真实字段名，值为 ConfigurableField 描述对象
#   返回：RunnableConfigurableFields 实例，仍实现 Runnable 协议
#
# ConfigurableField(id: str, ...) -> ConfigurableField
#   作用：描述一个可运行时配置的字段
#   参数：id - 配置键名，运行时通过 config={"configurable": {id: value}} 传值
#   返回：ConfigurableField 实例
#
# 注意：此处使用的字段名 template_demo 并非 PromptTemplate 的真实属性。
#   PromptTemplate 存放模板字符串的字段名为 template。
#   使用 template_demo 会在本行立即抛出：
#     ValueError: Configuration key template_demo not found in ...
#   正确写法见文件末尾"最佳实践"第 1 条。
prompt = PromptTemplate.from_template("请写一篇关于{subject}主题的冷笑话").configurable_fields(
    template_demo=ConfigurableField(id="prompt_template"),
)

# 2.传递配置更改prompt_template并调用生成内容
#
# prompt.invoke(input: dict, config: dict = None) -> StringPromptValue
#   作用：填充模板变量并生成提示词值对象，同时应用运行时配置
#   参数：
#     input - {"subject": "程序员"}，填充模板中的 {subject} 变量
#     config - {"configurable": {"prompt_template": 新模板字符串}}
#              键 prompt_template 对应上面 ConfigurableField 的 id
#   返回：StringPromptValue 对象
#   执行流程：
#     1. 框架从 config.configurable 中读取 prompt_template 的新值
#     2. 用新的模板字符串创建 PromptTemplate 副本（原 prompt 不变）
#     3. 副本以 subject="程序员" 进行格式化
#     4. 返回 StringPromptValue
#   关键点：模板从"冷笑话"变为"藏头诗"，但链结构与入参完全没变
#
# StringPromptValue.to_string() -> str
#   作用：将提示词值对象转换为纯字符串
#   返回：格式化后的完整提示词文本
#   相关方法：to_messages() 可转换为消息列表（适配 ChatModel）
content = prompt.invoke(
    {"subject": "程序员"},
    config={"configurable": {"prompt_template": "请写一篇关于{subject}主题的藏头诗"}}
).to_string()
print(content)

# ==================== 最佳实践与正确写法 ====================
# 1. 正确的字段名应为 template（PromptTemplate 的真实属性名）
# prompt = PromptTemplate.from_template("请写一篇关于{subject}主题的冷笑话").configurable_fields(
#     template=ConfigurableField(
#         id="prompt_template",
#         name="提示词模板",
#         description="运行时可替换的提示词模板字符串，需保留 {subject} 变量",
#     ),
# )
# content = prompt.invoke(
#     {"subject": "程序员"},
#     config={"configurable": {"prompt_template": "请写一篇关于{subject}主题的藏头诗"}}
# ).to_string()
# print(content)   # 输出：请写一篇关于程序员主题的藏头诗
#
# 2. 查看组件全部可配置的字段名（排查字段名写错的问题）
# print(PromptTemplate.model_fields.keys())
# # 常见可用字段：template / input_variables / partial_variables / template_format
#
# 3. 在完整链中运行时切换提示词（A/B 测试典型用法）
# import dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# dotenv.load_dotenv()
# chain = prompt | ChatOpenAI(model="deepseek-flash") | StrOutputParser()
# joke = chain.invoke({"subject": "程序员"})                                   # 使用默认模板
# poem = chain.invoke(
#     {"subject": "程序员"},
#     config={"configurable": {"prompt_template": "请写一篇关于{subject}主题的藏头诗"}}
# )
#
# 4. 用 with_config 派生固定配置的链变体（便于复用）
# poem_prompt = prompt.with_config(
#     configurable={"prompt_template": "请写一篇关于{subject}主题的藏头诗"}
# )
# print(poem_prompt.invoke({"subject": "程序员"}).to_string())
#
# 5. 同时开放多个字段（模板 + 模板格式）
# prompt = PromptTemplate.from_template("请写一篇关于{subject}主题的冷笑话").configurable_fields(
#     template=ConfigurableField(id="prompt_template"),
#     template_format=ConfigurableField(id="prompt_format"),
# )
#
# 6. 替换提示词时的注意事项
#   - 新模板的变量集合应与 invoke 传入的 input 键保持匹配
#   - 若新模板引入新变量（如 {style}），必须同时在 input 中提供该变量
#   - 模板内容若来自外部输入，需做安全校验，避免提示词注入风险
