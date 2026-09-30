#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/1 18:46
@Author  : thezehui@gmail.com
@File    : 2.Office文档加载器.py

===================================================================================
知识点讲解：Office 文档加载器的使用
===================================================================================

1. Office 文档加载器家族
   - UnstructuredExcelLoader       : 加载 Excel（.xlsx, .xls）
   - UnstructuredWordDocumentLoader: 加载 Word（.docx, .doc）
   - UnstructuredPowerPointLoader  : 加载 PowerPoint（.pptx, .ppt）
   - 三者都基于 unstructured 库，能解析复杂的 Office 格式并提取文本

2. mode 参数与「元素」分离
   - "single"（默认）：将整个文件作为一个 Document
   - "elements"：按文档元素分割（单元格、段落、幻灯片），适合细粒度检索
   - Office 类非结构化加载器使用极简：一般只传对应文档路径即可
   - 需要区分文档中的元素时传 mode="elements"，
     但文档明确建议一般不使用——原因同 Markdown：加载阶段分割无法保证块大小一致性

3. 各加载器特点
   - Excel      : 支持多工作表，elements 模式下每个单元格可作为一个元素，适合结构化表格数据
   - Word       : 保留段落结构，可提取标题、正文、列表，自动去除格式只留文本
   - PowerPoint : 按幻灯片或元素提取，含标题、文本框、备注，metadata 含幻灯片编号

4. 依赖库与精确安装命令
   - 需要 unstructured 库及其 Office 相关依赖：
     * Excel : pip install unstructured openpyxl pandas
     * PPT   : pip install unstructured python-magic python-pptx
     * Word  : pip install unstructured python-docx
   - 注意 PPT 额外需要 python-magic（做文件类型嗅探），Excel 额外需要 pandas

5. 典型输出示例（PPT + mode="elements"）
   ppt_loader = UnstructuredPowerPointLoader("./章节介绍.pptx", mode="elements")
   documents = ppt_loader.load()
   [Document(page_content='LangChain RAG应用开发组件深入学习',
             metadata={'source': './章节介绍.pptx', 'category_depth': 1,
                       'file_directory': '.', 'filename': '章节介绍.pptx',
                       'last_modified': '2024-07-20T11:44:28', 'page_number': 1,
                       'languages': ['zho', 'kor'],
                       'filetype': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                       'category': 'Title'}),
    Document(page_content='章节介绍', metadata={..., 'category': 'Title'})]
   17
   LangChain RAG应用开发组件深入学习
   - metadata 补充字段：
     * category_depth : 元素在文档结构中的层级深度（一级标题为 1）
     * page_number    : PPT 中即幻灯片页码
     * languages      : 识别出的语言列表，'zho' 中文、'kor' 韩文
     * filetype       : Office 文件的完整 MIME 类型（OOXML 规范）

6. 应用延伸：ChatPDF
   - 利用 unstructured 的办公文档加载能力，配合 LLM 即可实现 2023 年爆火的 ChatPDF 功能：
     上传特定 PDF，让 LLM 实现对该 PDF 的问答
   - 本质链路：Office/PDF 加载器 → 分割器 → 向量库 → 检索器 → LLM

===================================================================================
"""
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
)

# UnstructuredExcelLoader(): Excel 文档加载器（已注释）
#   参数:
#     - file_path: Excel 文件路径
#     - mode: "single" 或 "elements"，控制分割粒度
#   返回: UnstructuredExcelLoader 实例
#   作用: 解析 Excel 文件，提取表格数据和文本内容
# excel_loader = UnstructuredExcelLoader("./员工考勤表.xlsx", mode="elements")
# excel_documents = excel_loader.load()

# UnstructuredWordDocumentLoader(): Word 文档加载器（已注释）
#   参数:
#     - file_path: Word 文件路径
#     - mode: 加载模式，默认 "single"
#   返回: UnstructuredWordDocumentLoader 实例
#   作用: 解析 Word 文档，提取段落、标题等内容
# word_loader = UnstructuredWordDocumentLoader("./喵喵.docx")
# documents = word_loader.load()

# UnstructuredPowerPointLoader(): PowerPoint 文档加载器
#   参数:
#     - file_path: PowerPoint 文件路径
#     - mode: 加载模式，默认 "single"，可选 "elements" 按幻灯片/元素分割
#   返回: UnstructuredPowerPointLoader 实例
#   作用: 解析 PPT 文件，提取每页幻灯片的文本内容
ppt_loader = UnstructuredPowerPointLoader("./章节介绍.pptx")

# load(): 执行文档加载
#   返回: List[Document]
#   作用: 解析 PPT 文件结构，提取所有文本内容
documents = ppt_loader.load()

# 输出文档对象列表
print(documents)

# 输出文档数量（single 模式下通常为 1）
print(len(documents))

# 输出元数据信息
# 包含 source（文件路径）、page_number（幻灯片编号）等
print(documents[0].metadata)
