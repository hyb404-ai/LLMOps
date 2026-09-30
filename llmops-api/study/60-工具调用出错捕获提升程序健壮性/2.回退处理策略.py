#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/12 10:17
@Author  : thezehui@gmail.com
@File    : 2.回退处理策略.py

===================================================================================
知识点讲解：工具调用的回退处理策略（Fallbacks）
===================================================================================

1. with_fallbacks() 简介
   - LangChain Runnable 的通用容错机制：主链失败时自动切换到备用链
   - 调用方无感知，无需手动 try-except
   - 支持多级回退（按顺序尝试多个 fallback）

2. 参数详解
   - fallbacks：备用 Runnable 列表，按顺序尝试
   - exceptions_to_handle：指定要处理的异常类型（默认 Exception）
   - exception_key：将异常信息注入 fallback 输入的键名（可选，自我纠正时用到）

3. 回退执行流程
   - 步骤1：执行主链；步骤2：抛异常则尝试第一个 fallback
   - 步骤3：第一个也失败则尝试第二个；步骤4：全部失败抛出最后的异常

4. 典型回退场景
   - 模型降级 / 升级：弱模型失败回退强模型（本例），或反之
   - 提供商切换：OpenAI 失败切换到 Anthropic
   - 缓存回退：实时查询失败返回缓存；默认值回退：全失败返回默认响应

5. 本示例设计
   - 主链用 deepseek-flash（快、便宜），备用链用更强模型（实际可换 pro）
   - 备用链提示词更明确（提醒「不要忘记 dict_arg」），提高补全概率
   - 组合：主链.with_fallbacks([备用链])

6. 回退并非万能（重要）
   - 课程强调：即便换用参数更大的模型，生成的参数仍可能不合规
   - 此时应回过头检查工具描述、Prompt 是否写得有问题，而非一味加 fallback

7. with_fallbacks vs try-except
   - try-except：捕获异常、返回错误信息（不重试）
   - with_fallbacks：捕获异常、自动切换到备用方案（自动重试）
   - 需要提高成功率的场景更适合 with_fallbacks

8. 成本与性能 / 异常控制
   - 先试便宜快速方案，失败再回退昂贵可靠方案，优化整体成本与延迟
   - exceptions_to_handle 可精确控制：如只处理 ValidationError，不处理网络超时

9. 注意事项与最佳实践
   - fallback 链输入输出类型应与主链一致；避免过多层级（增加延迟）
   - 记录 fallback 触发次数便于监控；fallback 应比主链更可靠
   - 按可靠性递增排列；可结合 with_retry() 处理临时性故障：
       chain.with_retry().with_fallbacks([...])
   - with_retry()：重试同一个 Runnable（适合临时故障）；with_fallbacks()：切换不同 Runnable（适合逻辑故障）

===================================================================================
"""
import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 定义复杂工具，需要三个参数
# 这个工具容易因参数缺失而调用失败
@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg


# 1.创建大语言模型并绑定工具
# 主链使用的模型：deepseek-flash（速度快、成本低）
llm = ChatOpenAI(model="deepseek-flash").bind_tools([complex_tool])
# 备用链使用的模型：示例中使用相同模型
# 实际应用中应该使用能力更强的模型（如 deepseek-v4-pro）
# 或者使用不同的提示策略
better_llm = ChatOpenAI(model="deepseek-flash").bind_tools([complex_tool])

# 2.创建链并执行工具
# 备用链（fallback chain）
# 执行流程：better_llm 生成 tool_calls -> 提取参数 -> 执行工具
better_chain = (better_llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool)

# 主链 + 回退配置
# with_fallbacks([better_chain]) 的作用：
#   - 主链执行失败时（抛出异常），自动尝试 better_chain
#   - 支持传入多个 fallback，按顺序尝试
#   - 所有 fallback 都失败时，抛出最后的异常
chain = (llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool).with_fallbacks([better_chain])

# 3.调用链
# 注意提示词中明确提到"不要忘记了dict_arg参数"
# 这有助于 LLM 生成完整的参数
# 如果主链失败，会自动回退到 better_chain 重新尝试
print(chain.invoke("使用复杂工具，对应参数为5和2.1，不要忘记了dict_arg参数"))
