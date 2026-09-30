#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/3 9:30
@Author  : thezehui@gmail.com
@File    : 1.问答转换器示例.py

===================================================================================
知识点讲解：问答转换器（DoctranQATransformer）
===================================================================================

1. 非分割类文档转换器的定位
   - LangChain 中除了分割器，还存在另一种「非分割类型」的文档转换器：
     这类转换器同样传递文档列表并返回文档列表，一般是将某种文档按需求转换成
     另外一种格式（翻译、文档重排、HTML 转文本、文档元数据提取、文档转问答等）
   - 由于输入输出都是文档列表，它可以在 LLM 应用中任何存在文档列表的地方使用：
     文档加载、文档切割、检索器检索这几个环节的交互数据都是文档列表，
     因此这几个环节都可以插入文档转换器组件
   - 官方封装的文档转换器清单见：
     https://imooc-langchain.shortvar.com/docs/integrations/document_transformers/

2. 文档转换器的分类
   - 分割类转换器：将文档切分为多个小块（TextSplitter 系列）
   - 非分割类转换器：对文档内容做格式转换（翻译、摘要、问答、属性提取、重排等）

3. 问答转换器的设计动机（RAG 优化）
   - 在 RAG 的外挂知识库中，向量库里的文档通常以叙述或对话格式存储，
     但绝大部分用户的查询都是「问题」格式
   - 如果在对文档向量化之前先将其转换为问答格式，可以在一定程度上
     增加检索到相关文档的可能性、降低检索到不相关文档的可能性
   - 这是 RAG 应用开发中常见的一种优化策略：把原始数据转换成 QA 数据后再存储
   - 另外，绝大部分 LLM 微调使用的也是 QA 问答数据，同样可以考虑用该转换器生成

4. Doctran 库与安装
   - LangChain 封装了 Doctran 库并实现了 DoctranQATransformer 类
   - 该库底层使用 OpenAI 的函数回调（function calling）来实现问答数据的提取
   - 安装：
       pip install -U doctran

5. DoctranQATransformer 的用法
   - 代码：
       qa_transformer = DoctranQATransformer(openai_api_model="gpt-3.5-turbo-16k")
       transformer_documents = qa_transformer.transform_documents(documents)
       for qa in transformer_documents[0].metadata.get("questions_and_answers"):
           print(qa)
   - 使用文档转换器的通用套路：按对应组件的构造函数传参，
     然后调用 transform_documents() 即可完成快速转换
     （每个转换器生成的格式不一样，需查看文档了解生成内容详情）

6. transform_documents() 方法
   - 文档转换器的核心方法
   - 接收 Document 列表，返回转换后的 Document 列表
   - 本转换器的结果存储在 metadata 中，原 page_content 保持不变

7. questions_and_answers 元数据
   - 生成的问答对存于 metadata["questions_and_answers"]
   - 格式：List[Dict]，每项为 {"question": ..., "answer": ...}

8. 典型输出示例与观察点
   - 输入一段内部机密邮件（含日期、主题、安全、HR、营销、研发等段落），输出：
       {'question': '文件日期是什么？', 'answer': '2023年7月1日'}
       {'question': '文件主题是什么？', 'answer': '各种话题的更新和讨论'}
       {'question': '谁是IT部门的网络安全负责人？', 'answer': 'John Doe（电子邮件：john.doe@example.com）'}
       {'question': '如果发现安全风险或事件，应该向谁报告？', 'answer': '专门的团队，联系邮箱为security@example.com'}
       {'question': '谁在客户服务方面表现出色？', 'answer': 'Jane Smith（社保号：049-45-5928）'}
       {'question': '员工福利计划的开放报名期是什么时候？', 'answer': '即将到来'}
       ...（共 11 条）
   - 观察点 1：问答对的数量与原文信息密度正相关，长文档会产出更多 QA
   - 观察点 2：answer 会原样保留关键实体（邮箱、电话、日期），
     这类结构化信息正是检索时最容易被问到的部分
   - 观察点 3：原文 page_content 未被改写，QA 只作为 metadata 附加，
     因此同一份文档可以同时保留原文与 QA 两种检索视图

9. openai_api_model 参数
   - 指定使用的 LLM 模型（支持 OpenAI 兼容 API，如 DeepSeek）
   - 不同模型生成的问答质量与风格有差异

10. 使用场景与注意事项
    - 场景：自动生成 FAQ、构建问答系统、生成训练数据、文档内容校验
    - 注意：需调用 LLM API，有成本与延迟；质量依赖模型与文档；
      适合离线批处理，不适合实时链路
    - 长文档建议先切块再逐块转换，避免超出模型上下文窗口

===================================================================================

===================================================================================
"""
import dotenv
from doctran import Doctran
from langchain_community.document_transformers import DoctranQATransformer
from langchain_core.documents import Document

# 导入 Doctran 类（确保库已安装）
_ = Doctran

# 加载环境变量（API 密钥等配置）
dotenv.load_dotenv()

# 1.构建文档列表
# 示例文档：一封包含多个主题的机密邮件
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

# 2.构建问答转换器并转换
# DoctranQATransformer(): 问答转换器
#   参数:
#     - openai_api_model: 使用的 LLM 模型名称
#       "deepseek-v4-pro" 是兼容 OpenAI API 的模型
#   返回: DoctranQATransformer 实例
#   作用:
#     - 使用 LLM 分析文档内容
#     - 生成与文档相关的问答对
qa_transformer = DoctranQATransformer(openai_api_model="deepseek-v4-pro")

# transform_documents(): 执行文档转换
#   参数:
#     - documents: Document 对象列表
#   返回: List[Document]，转换后的文档列表
#   工作流程:
#     1. 将文档内容发送给 LLM
#     2. LLM 理解内容并生成相关问题
#     3. LLM 根据内容生成对应答案
#     4. 将问答对存储到 metadata["questions_and_answers"] 中
#   注意: 原文档的 page_content 不变，问答对存储在 metadata 中
transformer_documents = qa_transformer.transform_documents(documents)

# 3.输出内容
# 从 metadata 中提取生成的问答对
# metadata.get("questions_and_answers") 返回问答对列表
for qa in transformer_documents[0].metadata.get("questions_and_answers"):
    # 每个 qa 是一个字典，包含 "question" 和 "answer" 键
    print("问答数据:", qa)
