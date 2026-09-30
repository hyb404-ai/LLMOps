#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/9 19:03
@Author  : thezehui@gmail.com
@File    : 3.Model流式输出.py

===================================================================================
知识点讲解：Model 流式输出（Stream Processing）
===================================================================================

1. 流式输出的应用场景
   - 实时聊天应用：逐字显示 AI 回复，提升用户体验（打字机效果）
   - 长文本生成：避免用户长时间等待，及时看到部分结果
   - 服务端事件流（SSE）：Web 应用中实现流式推送

2. stream 在 Runnable 方法体系中的定位
   - 文档对 stream 的定义是：
     「invoke 的流式输出版本，大语言模型每生成一个字符就返回一个字符」
   - 实践中服务端通常按 token 而非严格单字符切分，因此一个 chunk 的 content
     可能是一个字、一个词甚至空字符串，不要假设「一个 chunk 一个汉字」
   - stream 同属 Runnable 协议标准接口，整条 Chain 也可以 stream；
     在 Runnable 协议中还有两个相关成员：
     * astream     : stream 的异步版本
     * astream_log : 除了流式返回最终响应块，还会流式返回中间步骤

3. stream() 方法的特性
   - Runnable 协议的标准方法之一（invoke, batch, stream）
   - 返回生成器（Generator），逐块产出 AIMessageChunk 对象
   - 支持 Server-Sent Events（SSE）协议

4. AIMessageChunk 对象
   - 流式输出中的每个数据块
   - content 字段包含本次生成的增量文本（delta）
   - 可以累加多个 chunk 的 content 得到完整响应
   - AIMessageChunk 重载了 + 运算符，chunk1 + chunk2 会合并为新的 chunk

5. 关于「组件不支持流式」时的行为
   - Runnable 协议对 stream 的约定是：
     「将组件的响应块流式返回，如果组件不支持流式则会直接输出」
   - 即对不支持流式的组件调用 stream 不会报错，而是退化为产出单个完整结果块
   - 这保证了整条链上混有非流式组件时，chain.stream(...) 依然可用

6. flush=True 和 end="" 的作用
   - flush=True：强制刷新输出缓冲区，立即显示到终端
   - end=""：取消默认的换行符，实现连续输出效果
   - 配合使用可以实现打字机效果

7. 流式输出 vs 非流式输出
   - 流式：首字节时间短，用户感知延迟低，适合交互场景
   - 非流式：等待完整响应，适合需要完整结果的场景（如数据处理）
   - 流式输出不会减少总耗时，但改善用户体验

8. 性能考虑
   - 流式输出会增加网络往返次数（多次 chunk 传输）
   - 适合对延迟敏感的场景，不适合高吞吐量批处理
   - 部分 API 对流式输出有速率限制

9. 典型输出示例与观察点
   - 逐块打印并拼接后，终端呈现为一段连续文本（与 invoke 的最终内容一致）：
       LLM 和 LLMOps 都是开发者社区中的术语。LLM 是指"Language Model"，
       而 LLMOps 则是指与操作相关的语言模型。... LLMOps 的目标是确保语言模型
       在生产环境中的稳定性、性能和安全性。
   - 观察点 1：肉眼看到的最终结果与非流式一致，差别只在「呈现过程」
   - 观察点 2：本例直接用 print 逐块输出，未做累加；若下游需要完整文本，
     需自行用字符串累加或把多个 AIMessageChunk 相加
   - 观察点 3：流式模式下 token 用量信息通常出现在最后一个 chunk 或需额外开启
     stream_usage 选项，不能像 invoke 那样稳定地从每个 chunk 读到 response_metadata

===================================================================================
"""
from datetime import datetime

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
dotenv.load_dotenv()

# 1. 编排 Prompt：构造提示词模板
prompt = ChatPromptTemplate.from_messages([
    # system 消息：定义 AI 的角色
    ("system", "你是DeepSeek开发的聊天机器人，请回答用户的问题，现在的时间是{now}"),
    
    # human 消息：用户输入占位符
    ("human", "{query}"),
    
# partial() 提前绑定时间参数
]).partial(now=datetime.now())

# 2. 创建大语言模型实例
llm = ChatOpenAI(model="deepseek-flash")

# 3. 使用 stream() 方法启动流式输出
# 返回生成器，逐块产出 AIMessageChunk 对象
response = llm.stream(prompt.invoke({"query": "你能简单介绍下LLM和LLMOps吗?"}))

# 4. 遍历流式响应，逐块输出
for chunk in response:
    # chunk.content 是本次生成的增量文本（delta）
    # flush=True 强制刷新缓冲区，立即显示
    # end="" 取消换行符，实现连续输出（打字机效果）
    print(chunk.content, flush=True, end="")
