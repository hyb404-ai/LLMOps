#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/21 9:24
@Author  : thezehui@gmail.com
@File    : 1.摘要缓冲混合记忆.py.py

===================================================================================
知识点讲解：用 OpenAI 原生 SDK 手写摘要缓冲混合记忆
===================================================================================

1. 为什么大模型需要「记忆」
   - LLM 接口本质是无状态的：每次请求独立，模型不记得上一轮说过什么
   - 模型「只能依靠用户本身的输入去产生输出」，必须由额外模块保存对话上下文，下次请求时一并输入
   - 因此「记忆」= 历史对话的存储策略 + 注入 Prompt 的方式
   - 落地起来只有两个动作：
     * 在 Prompt 中预留 chat_history 占位符
     * 实时保存 Human/AI 对话信息，每次对话时插入该占位符
   - 整个「记忆」话题的复杂度，都只来自「保存什么、保存多少、怎么压缩」

2. 两种注入形态（Chat Model 与 LLM 的差异）
   - Chat Model（本示例最终对接的形态）：记忆表现为 messages 数组的增长
       第 1 次：[{system}, {human:"你好，我是慕小课，我喜欢打篮球，你是？"}]
       第 2 次：[{system}, {human:第1轮提问}, {ai:第1轮回复}, {human:"我喜欢什么运动呢？"}]
       模型因此能答出「你喜欢篮球，对吗？」
   - 文本补全型 LLM：记忆表现为提示词里一段带标签的纯文本
       第 1 次：<chat-history></chat-history> 为空
       第 2 次：<chat-history>Human: ...\nAi: ...</chat-history>
   - 两种形态本质相同，只是「结构化消息列表」与「拼接文本」的表达差异
   - 本示例把记忆拼成文本塞进一条 user 消息，走的是第二种思路

3. 常见记忆模式全景（7 种）及其权衡
   - 缓冲记忆（Buffer）：全量存储、全量传入
       * 优点：无损记忆、实现最简单、所有模型支持、上下文窗口内完全无损
       * 缺点：token 随轮次线性增长，变慢变贵；超令牌上限后长对话记不住；小模型可用记忆极短
   - 缓冲窗口记忆（Buffer Window）：在缓冲记忆上加窗口值 k，只保留最近 k 轮
       * 优点：限制 token 表现优异、对小模型友好、实现简单
       * 缺点：忘记遥远互动；单轮过长仍可能超限
   - 令牌缓冲记忆（Token Buffer）：用 max_tokens 而非轮次界定保留范围
       * 优点：可按模型上下文长度精确分配记忆预算、对小模型友好
       * 缺点：同样不适合遥远互动
   - 摘要总结记忆（Summary）：把消息交给 LLM 总结，每次只传总结
       * 优点：长短期都能记（模糊记忆）、显著减少长对话 token、长对话优势明显
       * 缺点：细节丢失（有损压缩）；短对话反而增 token；质量依赖中间摘要 LLM
   - 摘要缓冲混合记忆（Summary Buffer）：本示例实现的策略（见第 5 点）
   - 实体记忆（Entity）：抽取「人物、地点、事件」等实体及其事实，按实体聚合存储
   - 向量存储库记忆（VectorStore）：记忆写入向量库，每次检索 Top-K
       * 优点：细节强于摘要、可记忆内容甚至无限长、token 消耗相对平衡
       * 缺点：性能较差、需额外 Embedding+向量库、效果受检索质量影响极大
   - 小结：精准存短期 + 模糊存长期 是业内公认最优解；
     「模糊存长期」主要有摘要总结与向量库两条路线

4. 本示例类的 8 个核心设计点
   - max_tokens：token 阈值，超过则触发摘要生成
   - summary：存储已压缩的摘要文本
   - chat_histories：存储尚未被压缩的近期原文对话
   - get_num_tokens：计算文本 token 数
   - save_context：保存一轮新对话，并在超限时触发摘要
   - get_buffer_string：把结构化历史列表拼成纯文本
   - load_memory_variables：输出可直接填入 Prompt 的记忆变量字典
   - summary_text：调用 LLM 把「旧摘要 + 新对话」融合成新摘要

5. 摘要缓冲混合记忆的模块拆解
   - 该模式可拆成 5 个必备模块，理解它们就等于理解整个类的设计：
     * chat_message_history：历史消息列表 → 本类的 chat_histories
     * moving_summary_buffer：被移除消息的汇总字符串（"moving" 指随对话滚动更新，而非一次性生成）→ 本类的 summary
     * summary_llm：生成摘要的 LLM，接收 summary + query + content 三类输入 → 本类的 self._client + summary_text
     * max_tokens：限制记忆存储的最大 token 数 → 本类的 max_tokens
     * get_num_tokens：统计文本 token 数 → 本类的 get_num_tokens
   - 对照价值：后续学 LangChain 的 ConversationSummaryBufferMemory 时，
     会发现它内部正是这 5 个模块（chat_memory / moving_summary_buffer / llm / max_token_limit / get_num_tokens_from_messages），命名几乎一致

6. 摘要触发的滚动淘汰算法（save_context 核心逻辑）
   - a. 把新一轮 {human, ai} 追加到 chat_histories 末尾
   - b. 拼接全部原文，计算 token 数
   - c. 若 token > max_tokens：取出最早一轮 chat_histories[0]，调用 LLM 将其与旧摘要融合成新摘要，再从 chat_histories 删除这一轮
   - d. 效果：最早的对话「滚动」从原文区迁移到摘要区，形成 FIFO 淘汰
   - 注意：本实现每次只淘汰一轮，若单轮内容极长，可能一次淘汰后仍超限

7. 用真实运行流程理解算法（以 max_tokens=300 为例）
   - 第 1 轮：摘要与消息列表都为空，AI 回复后消息列表存入第 1 组对话
   - 第 2 轮：消息列表累积到 2 组，总长度超 300 触发总结。把第 1 组 + 当前摘要（"-"）送进总结 Prompt，
     得到摘要「慕小课介绍自己喜欢唱跳rap和打篮球，并问AI的喜好……」，同时把第 1 组从消息列表删除
   - 第 3 轮验证：再问「我叫什么名字？」，此时原文已无「慕小课」三字，只有摘要里有
     AI 回复「你的名字是慕小课」——这正是摘要缓冲混合记忆生效的直接证据：原文被淘汰，关键信息通过摘要保留
   - 这也解释摘要提示词必须强调保留「姓名、爱好、性别、重要事件」，一旦丢名该验证即失败

8. 摘要提示词的工程技巧（summary_text 中的 prompt）
   - 明确角色与任务：「总结摘要，并添加到先前提供的摘要中」
   - 限定输出格式：「除了新摘要其他任何数据都不要生成」，避免模型输出解释性文字
   - 强调关键信息保留：姓名、爱好、性别、重要事件等，防止摘要丢失用户画像
   - 用 <example> 标签给 few-shot 示例，并显式声明示例不是真实数据
   - 用分隔线区分示例区与实际数据区，降低模型混淆概率

9. token 计数的简化与真实做法
   - 本示例用 len(query) 即字符数近似代替 token 数，仅为教学演示
   - 真实做法：
     * OpenAI 系：tiktoken.encoding_for_model(model).encode(text) 后取长度
     * LangChain：llm.get_num_tokens(text) 或 llm.get_num_tokens_from_messages(msgs)
   - 中文场景下字符数与 token 数差异较大，生产环境务必使用真实分词器

10. 推理模型（Reasoning Model）的流式处理陷阱
   - deepseek-v4-pro 等推理模型会先把思维链输出到 delta.reasoning_content，此阶段 delta.content 为 None
   - 若用 `if chunk.choices[0].delta.content is None: break` 作退出条件，第一个 chunk 就中断整条流，拿不到任何回答
   - 正确做法：分别判断 reasoning_content 与 content，只累积 content 进记忆

11. 与 LangChain 内置组件的对应关系
   - 本类 ≈ LangChain 0.x 的 ConversationSummaryBufferMemory
   - LangChain 1.x 中该能力由 SummarizationMiddleware 提供（见第 11 章）
   - 字段归属对照：chat_histories → memory.chat_memory / BaseChatMessageHistory；
     summary → moving_summary_buffer；llm → llm；max_tokens → max_token_limit；get_num_tokens → get_num_tokens_from_messages
   - 手写一遍的价值：理解记忆组件内部机制，便于定制与排障

===================================================================================
"""
import os
from typing import Any

import dotenv
from openai import OpenAI

# 从 .env 文件加载环境变量（OPENAI_API_KEY、OPENAI_API_BASE）
dotenv.load_dotenv()


# 1.max_tokens用于判断是否需要生成新的摘要
# 2.summary用于存储摘要的信息
# 3.chat_histories用于存储历史对话
# 4.get_num_tokens用于计算传入文本的token数
# 5.save_context用于存储新的交流对话
# 6.get_buffer_string用于将历史对话转换成字符串
# 7.load_memory_variables用于加载记忆变量信息
# 8.summary_text用于将旧的摘要和传入的对话生成新摘要
class ConversationSummaryBufferMemory:
    """
    摘要缓冲混合记忆类

    设计思想：
        把对话历史切成两部分存储——
          summary        ：早期对话压缩后的摘要（长期记忆，token 恒定可控）
          chat_histories ：近期对话的完整原文（短期记忆，细节无损）
        当原文区 token 超过阈值时，最早的一轮对话被压缩进摘要区并从原文区删除。

    与内置组件对应：
        等价于 LangChain 0.x 的 ConversationSummaryBufferMemory
    """

    def __init__(self, summary: str = '', chat_histories: list = None, max_tokens: int = 300):
        """
        __init__(summary: str, chat_histories: list, max_tokens: int) -> None
          作用：初始化摘要缓冲混合记忆

          参数：
            summary: str - 初始摘要文本，默认空串。可传入上次会话持久化的摘要实现跨会话记忆
            chat_histories: list - 初始历史对话列表，元素形如 {"human": ..., "ai": ...}
                                   默认 None，内部转为空列表
            max_tokens: int - 原文区的 token 上限，超过即触发摘要压缩，默认 300

          返回：None

          注意：chat_histories 默认值写成 None 而非 []，是为了规避 Python
                「可变对象作为默认参数」的经典陷阱（默认列表会在多个实例间共享）
        """
        self.summary = summary
        # 可变默认参数的安全写法：在函数体内创建新列表
        self.chat_histories = [] if chat_histories is None else chat_histories
        self.max_tokens = max_tokens
        # OpenAI(base_url: str) -> OpenAI
        #   作用：创建 OpenAI 兼容协议的客户端，供 summary_text 调用模型生成摘要
        #   参数：base_url - 从环境变量读取的 API 地址，支持指向国内代理或自建网关
        #   返回：OpenAI 客户端实例
        #   说明：api_key 由 SDK 自动从环境变量 OPENAI_API_KEY 读取
        self._client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"))

    @classmethod
    def get_num_tokens(cls, query: str) -> int:
        """
        get_num_tokens(query: str) -> int
          作用：计算传入文本的 token 数，用于判断是否需要触发摘要

          参数：
            query: str - 待计算的文本

          返回：
            int - token 数量

          实现说明：
            此处用字符数 len(query) 做近似，仅用于教学演示。
            生产环境应使用真实分词器：
              import tiktoken
              enc = tiktoken.encoding_for_model("gpt-4")
              return len(enc.encode(query))

          为什么用 classmethod：
            该计算不依赖实例状态，声明为类方法便于外部直接
            ConversationSummaryBufferMemory.get_num_tokens("文本") 调用
        """
        return len(query)

    def save_context(self, human_query: str, ai_content: str) -> None:
        """
        save_context(human_query: str, ai_content: str) -> None
          作用：保存新一轮对话，并在超出 token 阈值时把最早的一轮压缩进摘要

          参数：
            human_query: str - 本轮用户的输入内容
            ai_content: str - 本轮 AI 的完整回复内容

          返回：None

          副作用：
            1. 向 chat_histories 追加一条记录
            2. 可能调用 LLM 生成新摘要（有网络开销与 token 成本）
            3. 可能删除 chat_histories 的首个元素

          执行流程：
            追加新对话 → 拼接原文 → 计算 token → 超限则压缩最早一轮 → 删除该轮
        """
        # 步骤 1：把新一轮对话以 dict 形式追加到原文区末尾
        self.chat_histories.append({"human": human_query, "ai": ai_content})

        # 步骤 2：把原文区全部对话拼成一个字符串，用于计算总长度
        buffer_string = self.get_buffer_string()

        # 步骤 3：计算原文区当前的 token 数
        tokens = self.get_num_tokens(buffer_string)

        # 步骤 4：超出阈值则触发「滚动摘要」
        if tokens > self.max_tokens:
            # 取出最早的一轮对话（FIFO：先进先被压缩）
            first_chat = self.chat_histories[0]
            print("新摘要生成中~")
            # 调用 LLM 把「旧摘要 + 这一轮对话」融合成新摘要
            # 拼接格式 "Human:...\nAI:..." 与摘要提示词中的示例格式保持一致
            self.summary = self.summary_text(
                self.summary,
                f"Human:{first_chat.get('human')}\nAI:{first_chat.get('ai')}"
            )
            print("新摘要生成成功:", self.summary)
            # 该轮已被摘要吸收，从原文区删除，释放 token 空间
            del self.chat_histories[0]
            # 注意：这里只淘汰一轮。若单轮内容极长，淘汰后可能仍然超限，
            #       更严谨的实现应改为 while tokens > self.max_tokens 循环淘汰

    def get_buffer_string(self) -> str:
        """
        get_buffer_string() -> str
          作用：把结构化的历史对话列表序列化成纯文本，便于计算 token 与填入 Prompt

          参数：无

          返回：
            str - 形如 "Human:xxx\\nAI:yyy\\n\\nHuman:zzz\\nAI:www" 的文本，
                  末尾多余空白已用 strip() 清除

          格式说明：
            每轮内部用 \\n 分隔 Human 与 AI，轮次之间用 \\n\\n 分隔，
            这种「角色前缀 + 空行分隔」是让模型准确理解对话边界的常用格式

          对应 LangChain：
            等价于 langchain_core.messages.get_buffer_string(messages)
        """
        buffer: str = ""
        for chat in self.chat_histories:
            buffer += f"Human:{chat.get('human')}\nAI:{chat.get('ai')}\n\n"
        # strip() 去掉末尾多余的两个换行，保持输出整洁
        return buffer.strip()

    def load_memory_variables(self) -> dict[str, Any]:
        """
        load_memory_variables() -> dict[str, Any]
          作用：加载记忆变量字典，供外部格式化到 Prompt 模板中

          参数：无

          返回：
            dict[str, Any] - {"chat_history": "摘要:...\\n\\n历史信息:...\\n"}
                             键名 chat_history 对应 Prompt 模板中的变量名

          组装规则：
            摘要区（长期记忆）与原文区（短期记忆）拼成一个字符串，
            摘要在前提供全局背景，原文在后提供近期细节

          对应 LangChain：
            等价于 BaseMemory.load_memory_variables(inputs)，
            返回 dict 是为了能用 **memory_variables 直接解包进模板
        """
        buffer_string = self.get_buffer_string()
        return {
            "chat_history": f"摘要:{self.summary}\n\n历史信息:{buffer_string}\n"
        }

    def summary_text(self, origin_summary: str, new_line: str) -> str:
        """
        summary_text(origin_summary: str, new_line: str) -> str
          作用：调用 LLM，把旧摘要与新增的一轮对话融合成一份新摘要

          参数：
            origin_summary: str - 当前已有的摘要文本（首次调用时为空串）
            new_line: str - 待并入摘要的对话文本，格式 "Human:...\\nAI:..."

          返回：
            str - LLM 生成的新摘要文本

          副作用：发起一次同步的 LLM 请求，有网络延迟与 token 费用

          提示词设计要点：
            1. 角色设定 + 任务描述：明确要做「增量式摘要融合」而非重新总结
            2. 输出约束：「除了新摘要其他任何数据都不要生成」，避免多余解释
            3. 关键信息清单：姓名、爱好、性别、重要事件必须保留，防止用户画像丢失
            4. few-shot 示例：用 <example> 标签包裹，并显式声明「不是实际数据」
            5. 分隔线：用 ===== 明确切分示例区与实际数据区

          与课程原版提示词的差异（对照理解）：
            课程原版只有三句：角色 + 「总结内容，并将其添加到先前提供的摘要中，
            返回一个新的摘要」+ 一个 <example>。本实现额外加了两处强化：
              a. 「除了新摘要其他任何数据都不要生成」——防止模型输出「好的，这是新摘要：」
                 这类前缀，污染后续拼进 Prompt 的摘要文本
              b. 「姓名、爱好、性别、重要事件等全部都要包括」+「尽可能还原用户的对话记录」
                 ——直接针对手动模拟中「我叫什么名字？」那个验证问题做优化，
                 保证原文被淘汰后姓名仍能通过摘要被记住
            这也说明：摘要记忆的效果上限，完全由这段提示词与摘要模型的能力决定，
            这是摘要类记忆的固有风险（文档中列为「对话历史的记忆完全依赖中间摘要 LLM 的能力」）

          与 LangChain 0.x 的对应实现（源码级对照）：
            langchain/memory/summary_buffer.py 中的 prune() 就是本方法 + save_context
            淘汰逻辑的合体：
              def prune(self) -> None:
                  buffer = self.chat_memory.messages
                  curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
                  if curr_buffer_length > self.max_token_limit:
                      pruned_memory = []
                      while curr_buffer_length > self.max_token_limit:
                          pruned_memory.append(buffer.pop(0))
                          curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
                      self.moving_summary_buffer = self.predict_new_summary(
                          pruned_memory, self.moving_summary_buffer
                      )
            三点值得注意的差异：
              1. 官方用 while 循环「一次性淘汰到不超限」，并把淘汰出的多组对话
                 攒成 pruned_memory 后只调用一次摘要 LLM；
                 本实现是 if + 单轮淘汰，每轮最多压缩一组（见文件末尾最佳实践第 2 条）
              2. 官方的 predict_new_summary 对应本方法 summary_text，
                 入参同样是「待压缩的对话 + 当前摘要」，返回新摘要
              3. 官方在极端场合（第 1 条回复很短、第 2 条提问很长）会执行两次
                 token 长度计算，如果不异步执行，对话速度会变得非常慢——
                 这是 ConversationSummaryMemory 系列的已知性能坑
        """
        prompt = f"""你是一个强大的聊天机器人，请根据用户提供的谈话内容，总结摘要，并将其添加到先前提供的摘要中，返回一个新的摘要，除了新摘要其他任何数据都不要生成，如果用户的对话信息里有一些关键的信息，比方说姓名、爱好、性别、重要事件等等，这些全部都要包括在生成的摘要中，摘要尽可能要还原用户的对话记录。

请不要将<example>标签里的数据当成实际的数据，这里的数据只是一个示例数据，告诉你该如何生成新摘要。

<example>
当前摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量。

新的对话：
Human：为什么你认为人工智能是一股向善的力量？
AI：因为人工智能会帮助人类充分发挥潜力。

新摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量，因为它将帮助人类充分发挥潜力。
</example>

=====================以下的数据是实际需要处理的数据=====================

当前摘要：{origin_summary}

新的对话：
{new_line}

请帮用户将上面的信息生成新摘要。"""
        # client.chat.completions.create(model: str, messages: list) -> ChatCompletion
        #   作用：调用 OpenAI 兼容的对话补全接口生成摘要
        #   参数：
        #     model - 模型名称，摘要任务建议用能力较强的模型保证信息不丢失
        #     messages - 消息列表，这里只有一条 user 消息承载完整摘要指令
        #   返回：ChatCompletion 对象，非流式（摘要是内部步骤，无需流式展示）
        completion = self._client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}]
        )
        # 从响应结构中取出文本：choices[0] 是第一个候选，.message.content 是正文
        return completion.choices[0].message.content


# 1.创建openai客户端
# OpenAI(base_url: str) -> OpenAI
#   作用：创建用于主对话流程的客户端（与 memory 内部的客户端相互独立）
#   参数：base_url - API 地址，从环境变量读取
#   返回：OpenAI 客户端实例
client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"))

# ConversationSummaryBufferMemory(summary, chat_histories, max_tokens) -> 实例
#   参数："" 表示无初始摘要，[] 表示无初始历史，300 是触发摘要的 token 阈值
#   返回：记忆实例，后续通过 load_memory_variables / save_context 读写
memory = ConversationSummaryBufferMemory("", [], 300)

# 2.创建一个死循环用于人机对话
while True:
    # 3.获取人类的输入
    query = input('Human: ')

    # 4.判断下输入是否为q，如果是则退出
    if query == 'q':
        break

    # 5.向openai的接口发起请求获取ai生成的内容
    # memory.load_memory_variables() -> dict[str, Any]
    #   作用：取出「摘要 + 近期原文」组装好的记忆文本
    #   返回：{"chat_history": "摘要:...\n\n历史信息:..."}
    memory_variables = memory.load_memory_variables()

    # 手动拼装 Prompt：系统指令 + 记忆上下文 + 本轮用户提问
    # 这正是 LangChain Prompt 模板在底层帮我们做的事情
    answer_prompt = (
        "你是一个强大的聊天机器人，请根据对应的上下文和用户提问解决问题。\n\n"
        f"{memory_variables.get('chat_history')}\n\n"
        f"用户的提问是: {query}"
    )

    # client.chat.completions.create(model, messages, stream) -> Stream[ChatCompletionChunk]
    #   作用：发起流式对话补全请求
    #   参数：
    #     model: str - 模型名称
    #     messages: list - 消息列表，这里把记忆与提问合并进一条 user 消息
    #     stream: bool - True 表示流式返回，逐块产出 chunk
    #   返回：Stream 对象，可迭代，每次产出一个 ChatCompletionChunk
    response = client.chat.completions.create(
        model='deepseek-v4-pro',
        messages=[
            {"role": "user", "content": answer_prompt},
        ],
        stream=True,
    )

    # 6.循环读取流式响应的内容
    # 注意：deepseek-v4-pro是推理模型，会先把思考过程写入reasoning_content，
    # 此时content为None，不能用content is None做退出条件，否则第一个chunk就会中断整个流
    ai_content = ""  # 累积正式回答，仅这部分会写入记忆（思考过程不入库）
    in_reasoning = False  # 状态标志：当前是否处于「思考阶段」，用于控制提示前缀只打印一次
    for chunk in response:
        # 防御性判断：部分网关会下发 choices 为空的心跳 chunk，需跳过
        if not chunk.choices:
            continue
        # delta 是本次增量，含 content / reasoning_content / role 等字段
        delta = chunk.choices[0].delta

        # 6.1 输出思考过程
        # getattr(delta, "reasoning_content", None)
        #   作用：安全读取推理模型特有的字段，非推理模型没有该属性也不会报错
        #   返回：str | None - 思维链增量文本，非思考阶段为 None
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_reasoning:
                # 首次进入思考阶段时打印一次前缀标记
                print("[思考] ", flush=True, end="")
                in_reasoning = True
            print(reasoning, flush=True, end="")

        # 6.2 输出正式回答，并累积到ai_content中用于记录记忆
        if delta.content:
            if in_reasoning:
                # 从思考阶段切换到回答阶段，换行并打印回答前缀
                print("\n[回答] ", flush=True, end="")
                in_reasoning = False
            # 关键：只把正式回答累积进 ai_content，思考过程不写入记忆，
            #       避免思维链污染上下文并浪费大量 token
            ai_content += delta.content
            print(delta.content, flush=True, end="")
    print("")

    # memory.save_context(human_query: str, ai_content: str) -> None
    #   作用：把本轮完整问答写入记忆，必要时触发摘要压缩
    #   参数：query - 用户输入；ai_content - 累积得到的完整 AI 回答
    #   返回：None
    #   注意：必须在流消费完毕后调用，否则 ai_content 不完整
    memory.save_context(query, ai_content)

# ===================================================================================
# 最佳实践与常见问题
# ===================================================================================
#
# 1. 用真实分词器替换字符数近似（强烈建议）：
#    import tiktoken
#    @classmethod
#    def get_num_tokens(cls, query: str) -> int:
#        enc = tiktoken.get_encoding("cl100k_base")
#        return len(enc.encode(query))
#
# 2. 循环淘汰而非单次淘汰，确保压缩后一定不超限：
#    while self.get_num_tokens(self.get_buffer_string()) > self.max_tokens and self.chat_histories:
#        first_chat = self.chat_histories.pop(0)
#        self.summary = self.summary_text(self.summary, f"Human:{first_chat['human']}\nAI:{first_chat['ai']}")
#
# 3. 用多条 messages 代替单条大 user 消息，让角色边界更清晰、缓存命中率更高：
#    messages = [
#        {"role": "system", "content": f"你是聊天机器人。\n\n{memory_variables['chat_history']}"},
#        {"role": "user", "content": query},
#    ]
#
# 4. 持久化记忆以支持跨进程会话：
#    把 summary 与 chat_histories 序列化成 JSON 存入 Redis/数据库，
#    初始化时回填：ConversationSummaryBufferMemory(saved_summary, saved_histories, 300)
#
# 5. 对应的 LangChain 现代写法（见第 11 章示例）：
#    from langchain.agents.middleware import SummarizationMiddleware
#    agent = create_agent(model=llm, tools=[], middleware=[
#        SummarizationMiddleware(model=llm, trigger=("tokens", 300), keep=("messages", 4))
#    ], checkpointer=InMemorySaver())
#
# 6. 常见错误与解决方案：
#    - 错误：流式输出第一个 chunk 就退出，拿不到回答
#      原因：用 `if delta.content is None: break` 作为退出条件，
#            而推理模型的思考阶段 content 恒为 None
#      解决：改为 `if delta.content:` 累积，用 for 循环自然结束
#
#    - 错误：ai_content 为空导致记忆里 AI 回复缺失
#      原因：把思考内容误存进 reasoning 变量而没有累积 content
#      解决：确认只在 `if delta.content:` 分支内做 ai_content += delta.content
#
#    - 错误：多个实例共享了同一份历史
#      原因：把 chat_histories 的默认值直接写成 []（可变默认参数陷阱）
#      解决：默认值用 None，在 __init__ 内部创建新列表（本示例已正确处理）
#
#    - 错误：摘要生成后信息大量丢失
#      原因：摘要提示词没有强调保留关键信息，或使用了能力过弱的模型
#      解决：在提示词中列出必须保留的信息类型，并使用能力更强的模型做摘要
