#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/10 1:33
@Author  : thezehui@gmail.com
@File    : 2.JsonOutputParser使用技巧.py

===================================================================================
知识点讲解：JsonOutputParser 结构化输出解析器
===================================================================================

1. 结构化输出的需求
   - LLM 默认输出自由文本，难以直接用于程序逻辑
   - 很多场景需要提取结构化数据（姓名、日期、分类标签等）
   - JsonOutputParser 通过 JSON 格式规范 LLM 输出

2. 「输出解析器 = 预设提示 + 解析功能」在本例的完整体现
   - StrOutputParser 只有「解析功能」（且原样返回），没有「预设提示」
   - JsonOutputParser 是两部分齐全的典型：预设提示 = get_format_instructions() 产出的描述文本，需开发者主动嵌入 Prompt（本例通过 {format_instructions} 占位符 + partial 完成）；解析功能 = parse / parse_result，把模型返回的 JSON 文本转成 dict
   - 关键认知：JsonOutputParser 并不能「强制」模型输出 JSON，它只是把格式要求写进提示词再对返回内容做尽力解析，所以「必须把 format_instructions 嵌入 prompt」是使用前提，漏了就会解析失败

3. 使用前必须定义 BaseModel
   - 使用前必须定义 BaseModel 类，告知解析器需要解析的数据结构，并将解析器提供的描述文本嵌入提示中
   - 版本演进对照：文档 0.x 用 langchain_core.pydantic_v1 导入；LangChain 1.x 改为直接使用 pydantic v2（from pydantic import BaseModel, Field）；照搬旧的 pydantic_v1 导入会得弃用警告或导入失败

4. Pydantic 模型与 Field 的作用
   - 定义数据结构：字段名、类型、描述
   - Field(description=...) 用于生成格式指令，指导 LLM 输出（最终写进 schema 再进入提示词）
   - 提供类型校验与文档生成能力

5. get_format_instructions 方法（源码级）
   - 核心逻辑是把 Pydantic 模型的 JSON Schema 塞进一段固定的指令模板：
       def get_format_instructions(self) -> str:
           if self.pydantic_object is None:
               return "Return a JSON object."
           schema = {k: v for k, v in self._get_schema(...).items()}
           reduced_schema = schema
           if "title" in reduced_schema: del reduced_schema["title"]
           if "type" in reduced_schema: del reduced_schema["type"]
           schema_str = json.dumps(reduced_schema)
           return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
   - 不传 pydantic_object 时指令退化为 "Return a JSON object."，结构不可控
   - Field(description=...) 能影响输出，正是因为它随 schema 进入提示词
   - 框架会刻意删掉顶层 title / type 字段来精简提示词

6. 解析流程与 parse_result 容错（源码级）
   - 解析实际发生在 parse_result，使用 parse_json_markdown 而非裸 json.loads，因此模型把 JSON 包在 ```json 代码块里也能正确解析（对常见输出习惯的容错设计）
   - partial=False（默认 invoke 场景）：解析失败抛 OutputParserException，且异常携带 llm_output 原文便于排查
   - partial=True（流式场景）：解析失败返回 None，是 JsonOutputParser 能支持 stream 渐进产出不完整 JSON 的原因
   - text.strip() 说明前后空白会被自动清理，无需自己预处理

7. 典型输出示例
   - 解析结果是一个 Python dict，键名与 Pydantic 模型字段一一对应：
       {'joke': '为什么程序员总是冷静的？',
        'punchline': '因为他们总是有一堆bug在身边。'}
   - 中间态观察：prompt_value.to_string() 可见 format_instructions 展开为含 JSON Schema 的英文指令；ai_message.content 是一段 JSON 文本（可能带代码块围栏）
   - 重要差异：JsonOutputParser 只返回 dict（不做模型实例化校验），与 PydanticOutputParser 不同，需用 joke.get("punchline") 取值

8. 应用场景
   - 信息抽取：从文本提取结构化信息（人名、地点、时间等）
   - 分类任务：输出分类标签与置信度
   - 表单填充：根据用户输入生成结构化表单数据
   - API 集成：将 LLM 输出转换为标准 API 参数

9. 与不使用解析器的方案对比
   - 若模型支持 Function / Tool Calling，可直接定义函数并规定其参数结构，强制模型调用该函数，精准按约束输出，无需 JsonOutputParser
   - 取舍：JsonOutputParser 不依赖模型的工具调用能力、对任意指令遵循型模型都适用；缺点是格式约束是「软约束」，存在解析失败风险

10. 注意事项
    - 需要模型具备一定的指令遵循能力（如 GPT-3.5+、DeepSeek 等）
    - 复杂结构可能导致输出错误，需增加示例或简化结构
    - 解析失败可配合重试 / 修复类解析器、收紧提示词，或改用 with_structured_output()

===================================================================================
"""
import dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()


# 1. 创建 Pydantic 模型，定义期望的 JSON 数据结构
# 该模型既用于生成格式指令，也用于输出验证
class Joke(BaseModel):
    # joke 字段：存储冷笑话内容
    # Field() 的 description 参数会出现在格式指令中，指导 LLM 输出
    joke: str = Field(description="回答用户的冷笑话")
    
    # punchline 字段：存储冷笑话的笑点
    punchline: str = Field(description="这个冷笑话的笑点")


# 创建 JsonOutputParser，传入 Pydantic 模型
# parser 会根据模型生成格式指令，并解析 LLM 输出
parser = JsonOutputParser(pydantic_object=Joke)

# 可以打印格式指令，查看生成的 JSON Schema 说明
# print(parser.get_format_instructions())

# 2. 构建提示模板，包含格式指令
# {format_instructions} 占位符用于插入自动生成的格式说明
# partial() 提前绑定格式指令，避免每次调用都手动传入
prompt = ChatPromptTemplate.from_template("请根据用户的提问进行回答。\n{format_instructions}\n{query}").partial(
    format_instructions=parser.get_format_instructions())
print(prompt)

# 3. 构建大语言模型
llm = ChatOpenAI(model="deepseek-flash")

# 4. 调用模型生成结果
# LLM 会根据格式指令输出符合 Joke 模型结构的 JSON 字符串
message = llm.invoke(prompt.invoke({"query": "请讲一个关于程序员的冷笑话"}))
# print(message)

# 5. 解析 JSON 输出
# parser.invoke() 解析 AIMessage.content 中的 JSON 字符串
# 返回 Python 字典，键为 Pydantic 模型的字段名
joke = parser.invoke(message)

# 访问解析后的数据（Python 字典）
# print(type(joke))  # <class 'dict'>
# print(joke.get("punchline"))  # 获取笑点字段
# print(joke)  # 打印完整字典

# 推荐的 LCEL 管道写法：
# chain = prompt | llm | parser
# joke = chain.invoke({"query": "请讲一个关于程序员的冷笑话"})
