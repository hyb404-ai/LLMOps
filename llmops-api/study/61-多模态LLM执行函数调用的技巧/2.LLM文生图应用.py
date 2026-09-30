#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/13 12:34
@Author  : thezehui@gmail.com
@File    : 2.LLM文生图应用.py

===================================================================================
知识点讲解：LLM 文生图应用（DALL-E 图像生成）
===================================================================================

1. 文生图（Text-to-Image）与「多模态输出」的思路
   - 根据文本描述生成图像，代表：DALL-E 3、Stable Diffusion、Midjourney
   - 大语言模型本身不直接输出图像，而是「曲线救国」：通过函数调用触发图像生成工具，实现多模态输出
   - 运行原理：把文本生图工具绑定到 LLM，让 LLM 把用户输入转成绘图 Prompt，再调用工具生成图片

2. LLM + 文生图的组合价值
   - 用户输入简单描述（如「老爷爷爬山」）
   - LLM 将其优化为详细的图像提示词（prompt engineering）
   - DALL-E 根据优化后的提示词生成高质量图像，降低使用门槛、提升效果

3. OpenAIDALLEImageGenerationTool
   - LangChain 内置的 DALL-E 工具，位于 langchain_community.tools.openai_dalle_image_generation
   - 默认工具名称 "openai_dalle"，接收文本提示词，返回图像 URL
   - 通过 api_wrapper 传入 DallEAPIWrapper 实例完成真实 API 调用

4. DallEAPIWrapper 参数详解
   - model：模型版本（"dall-e-2" 或 "dall-e-3"）
   - size：图像尺寸（"1024x1024"、"1792x1024"、"1024x1792"）
   - quality：图像质量（"standard" 或 "hd"，仅 DALL-E 3）
   - style：风格（"vivid" 或 "natural"，仅 DALL-E 3）
   - n：生成图像数量（DALL-E 3 仅支持 1）

5. DALL-E 2 vs DALL-E 3
   - DALL-E 3：质量更高、理解提示词更准确，且会自动优化用户提示词
   - DALL-E 3：仅支持生成 1 张图片
   - DALL-E 2：成本更低，支持批量生成

6. tool_choice 强制调用的作用
   - bind_tools([dalle], tool_choice="openai_dalle")
   - 强制 LLM 必须调用 DALL-E 工具，避免只回复文本而不生成图像
   - 确保每次输入都会触发图像生成

7. 链的执行流程
   - 步骤1：用户输入简单描述（"帮我绘制一张老爷爷爬山的图片"）
   - 步骤2：LLM 理解意图，生成优化的图像提示词
   - 步骤3：lambda 提取 tool_calls[0]["args"]（含优化后的 prompt）
   - 步骤4：dalle 工具调用 DALL-E API 生成图像
   - 步骤5：返回生成的图像 URL

8. LLM 作为提示词优化器
   - 用户输入："老爷爷爬山"
   - LLM 可能优化为："An elderly man with white hair climbing a mountain trail, wearing hiking gear, scenic mountain landscape, golden hour lighting, photorealistic"
   - 显著提升生成图像的质量与细节（DALL-E 3 内部也会再做一次优化）

9. 返回值说明与典型输出
   - DALL-E 工具返回图像的临时 URL，有效期通常约 1 小时，需及时下载保存
   - 典型返回形如：
       https://oaidalleapiprodscus.blob.core.windows.net/.../img-NxKXd4xBMJZdIixcxYQe5dmd.png?st=...&se=...
   - 观察点：拿到的是 Azure Blob 临时链接，生产环境应持久化到本地或对象存储

10. 应用场景
   - AI 绘画、营销素材自动生成、文章配图自动生成、产品原型设计、教育插图生成

11. 注意事项与最佳实践
   - DALL-E API 成本较高、生成较慢（通常 10-30 秒）
   - 图像 URL 有时效性需及时保存；存在内容安全审核可能拒绝某些请求
   - 用 tool_choice 确保工具被调用；让 LLM 优化提示词提升效果
   - 添加生成失败的错误处理与内容过滤，并记录 prompt 和结果便于优化

===================================================================================
"""
import dotenv
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI

# 加载环境变量（需要 OPENAI_API_KEY）
dotenv.load_dotenv()

# 创建 DALL-E 图像生成工具
# api_wrapper 参数传入 DallEAPIWrapper 实例
# DallEAPIWrapper 参数说明：
#   - model="dall-e-3"：使用 DALL-E 3 模型（质量更高）
#   - 可选参数：size（尺寸）、quality（质量）、style（风格）
# 工具默认名称为 "openai_dalle"
dalle = OpenAIDALLEImageGenerationTool(api_wrapper=DallEAPIWrapper(model="dall-e-3"))

# 创建大语言模型
# LLM 在这里扮演"提示词优化器"的角色
llm = ChatOpenAI(model="deepseek-v4-pro")

# 绑定 DALL-E 工具到 LLM
# tool_choice="openai_dalle" 强制 LLM 必须调用该工具
# 这样确保用户输入一定会触发图像生成，而不是只返回文本回复
llm_with_tools = llm.bind_tools([dalle], tool_choice="openai_dalle")

# 构建链
# 执行流程：
#   步骤1：llm_with_tools 接收用户输入，理解意图
#          LLM 会将简单描述优化为详细的图像提示词
#   步骤2：lambda 提取第一个工具调用的参数（含优化后的 prompt）
#   步骤3：dalle 调用 DALL-E API 生成图像，返回图像 URL
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | dalle

# 执行链
# 输入：简单的自然语言描述
# 输出：DALL-E 生成的图像 URL（注意：URL 有时效性，通常 1 小时）
print(chain.invoke("帮我绘制一张老爷爷爬山的图片"))
