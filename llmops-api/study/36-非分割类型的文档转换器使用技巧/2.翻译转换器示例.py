#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/3 9:59
@Author  : thezehui@gmail.com
@File    : 2.翻译转换器示例.py

===================================================================================
知识点讲解：翻译转换器（DoctranTextTranslator）
===================================================================================

1. 非分割类文档转换器的定位
   - LangChain 中除了分割器，还存在另一种「非分割类型」的文档转换器：
     传递文档列表并返回文档列表，一般是将某种文档按需求转换成另外一种格式
     （翻译文档、文档重排、HTML 转文本、文档元数据提取、文档转问答等）
   - 由于输入输出都是文档列表，它可以在文档加载、文档切割、检索器检索等
     任何以文档列表交互的环节插入使用
   - 官方封装的文档转换器清单见：
     https://imooc-langchain.shortvar.com/docs/integrations/document_transformers/

2. 跨语言向量检索的背景
   - 将文档通过嵌入 / 向量的方式进行比较的好处是能跨语言工作：
     「你好，世界！」、「Hello, World!」和「こんにちは、世界！」分别是中英日三国语言，
     但因为语义相近，在向量空间中的位置也非常接近
   - 当一个 RAG 应用需要跨语言工作时，一般有两种策略：
     * 嵌入时翻译：在将文档切块并嵌入存储到向量数据库时，
       同时把文档翻译成多国语言并执行相同的操作
     * 检索时翻译：在进行检索操作时，将检索出来的文档执行翻译，然后使用翻译后的文档
   - 两种策略都依赖「文档翻译」，即把文档转换成另一种形式的文档，
     这与文档转换器的作用完全一致，因此可用该组件实现

3. Doctran 库与安装
   - Doctran 是一个文本翻译库，底层使用 OpenAI 的函数调用功能在不同语言之间实现翻译
   - 安装：
       pip install -U doctran
   - LangChain 针对翻译提供了不少转换器，DoctranTextTranslator 是其中之一

4. DoctranTextTranslator 的用法
   - 代码：
       text_translator = DoctranTextTranslator(openai_api_model="gpt-3.5-turbo-16k")
       translator_documents = text_translator.transform_documents(documents)
       print(translator_documents[0].page_content)
   - 结果与问答转换器不同：翻译会替换 page_content（原文被覆盖）

5. transform_documents() 的行为差异
   - 问答转换器：原文不变，结果存于 metadata
   - 翻译转换器：原文被翻译文本替换（page_content 改变），metadata 保持不变

6. 典型输出示例与观察点
   - 输入中文 / 其他语言文档，输出英文：
       Confidential Document - For Internal Use Only
       Date: July 1, 2023
       Subject: Updates and Discussions on Various Topics

       Dear Team,

       I hope this email finds you all well. ...
   - 观察点 1：段落结构与空行被完整保留，说明转换器只替换文本、不动版式
   - 观察点 2：原文中的邮箱、电话、日期等实体被准确保留，
     这是 LLM 翻译相对传统机翻的优势之一
   - 观察点 3：page_content 被覆盖是不可逆的，若仍需原文，
     应在转换前把原文存入 metadata 或另存一份

7. 语言检测与目标语言
   - 默认自动检测源语言
   - 可通过 language 参数指定目标语言（如 "en"、"zh"）
   - 支持多语言批量翻译

8. openai_api_model 参数
   - 指定 LLM 模型（底层使用 OpenAI 函数调用实现翻译）
   - 建议使用较强模型以保证翻译质量

9. 翻译转换器 vs 传统翻译 API
   - LLM 翻译：理解上下文，更自然流畅，能处理术语、俚语、文化差异
   - 传统翻译：逐句翻译，可能缺乏连贯性，但成本与延迟更低
   - LLM 翻译成本较高、速度较慢

10. 使用场景与注意事项
    - 场景：多语言知识库、文档本地化、跨语言检索、内容全球化
    - 注意：需调用 LLM API 有成本；长文档耗时；专业术语可能需人工校对；
      适合批量离线处理
    - 若只需检索命中，可不必翻译全文——依靠 Embedding 的跨语言对齐能力往往已足够

11. 其他文档转换器
    - DoctranPropertyExtractor：提取文档属性
    - 自定义转换器：继承 BaseDocumentTransformer 并实现 transform_documents()

===================================================================================

===================================================================================
"""
import dotenv
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_core.documents import Document

# 加载环境变量（API 密钥等）
dotenv.load_dotenv()

# 1.构建文档列表
# 示例：中文机密文件（需要翻译为其他语言）
page_content = """机密文件 - 仅供内部使用
日期：2023年7月1日
主题：各种话题的更新和讨论
亲爱的团队，
希望这封邮件能找到你们一切安好。在这份文件中，我想向你们提供一些重要的更新，并讨论需要我们关注的各种话题。请将此处包含的信息视为高度机密。
安全和隐私措施
作为我们不断致力于确保客户数据安全和隐私的一部分，我们已在所有系统中实施了强有力的措施。我们要赞扬IT部门的John Doe（电子邮件：john.doe@example.com）在增强我们网络安全方面的勤奋工作。未来，我们提醒每个人严格遵守我们的数据保护政策和准则。此外，如果您发现任何潜在的安全风险或事件，请立即向我们专门的团队报告，联系邮箱为security@example.com。
人力资源更新和员工福利
最近，我们迎来了几位为各自部门做出重大贡献的新团队成员。我要表扬Jane Smith（社保号：049-45-5928）在客户服务方面的出色表现。Jane一直受到客户的积极反馈。此外，请记住我们的员工福利计划的开放报名期即将到来。如果您有任何问题或需要帮助，请联系我们的人力资源代表Michael Johnson（电话：418-492-3850，电子邮件：michael.johnson@example.com）。
营销倡议和活动
我们的营销团队一直在积极制定新策略，以提高品牌知名度并推动客户参与。我们要感谢Sarah Thompson（电话：415-555-1234）在管理我们的社交媒体平台方面的杰出努力。Sarah在过去一个月内成功将我们的关注者基数增加了20%。此外，请记住7月15日即将举行的产品发布活动。我们鼓励所有团队成员参加并支持我们公司的这一重要里程碑。
研发项目
在追求创新的过程中，我们的研发部门一直在为各种项目不懈努力。我要赞扬David Rodriguez（电子邮件：david.rodriguez@example.com）在项目负责人角色中的杰出工作。David对我们尖端技术的发展做出了重要贡献。此外，我们希望每个人在7月10日定期举行的研发头脑风暴会议上分享他们的想法和建议，以开展潜在的新项目。
请将此文档中的信息视为最机密，并确保不与未经授权的人员分享。如果您对讨论的话题有任何疑问或顾虑，请随时直接联系我。
感谢您的关注，让我们继续共同努力实现我们的目标。
此致，
Jason Fan
联合创始人兼首席执行官
Psychic
jason@psychic.dev"""

# 创建 Document 对象列表
documents = [Document(page_content=page_content)]

# 2.构建翻译转换器并翻译
# DoctranTextTranslator(): 文本翻译转换器
#   参数:
#     - openai_api_model: 使用的 LLM 模型名称
#       "deepseek-v4-pro" 是支持多语言的强大模型
#     - language: 目标语言（可选），如 "en"（英语）、"zh"（中文）
#       不指定时，LLM 会自动检测源语言并选择合适的目标语言
#   返回: DoctranTextTranslator 实例
#   作用:
#     - 使用 LLM 理解源文本的语义和上下文
#     - 翻译为目标语言，保持原文风格和结构
text_translator = DoctranTextTranslator(openai_api_model="deepseek-v4-pro")

# transform_documents(): 执行翻译
#   参数:
#     - documents: Document 对象列表
#   返回: List[Document]，翻译后的文档列表
#   工作流程:
#     1. 提取文档的 page_content
#     2. 发送给 LLM 进行翻译
#     3. 用翻译结果替换 page_content
#     4. metadata 保持不变
#   注意: 原文被翻译文本替换，不是存储在 metadata 中
translator_documents = text_translator.transform_documents(documents)

# 3.输出翻译内容
# page_content 现在包含翻译后的文本（可能是英文或其他语言）
print(translator_documents[0].page_content)
