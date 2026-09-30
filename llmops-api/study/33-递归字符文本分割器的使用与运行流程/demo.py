#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@File    : demo.py

===================================================================================
知识点讲解：递归字符分割器源码剖析
===================================================================================

1. 本文件定位
   - 这是 langchain_text_splitters 中 CharacterTextSplitter 与 RecursiveCharacterTextSplitter 的源码摘录，仅用于配合本章示例理解内部运行流程，不含可执行演示
   - 配套运行流程三阶段（RAG 高频考点）：预分割 → 大文档块递归分割 → 小文档块合并

2. TextSplitter 抽象基类的契约
   - 子类只需实现 split_text(text: str) -> List[str]
   - 基类已提供 _merge_splits()（合并碎片段成饱满块）、_length_function（默认 len，可换 token 计数）、_chunk_size、_chunk_overlap
   - split_documents() / create_documents() 均基于 split_text 实现

3. CharacterTextSplitter：单一分隔符的朴素分割
   - 只用单一分隔符（默认 "\n\n"）一刀切
   - 缺陷：不保证块大小——无该分隔符时整篇成超大块；分隔符过密时产生大量碎片
   - 适用：结构极规整、分隔符可靠的文本（如固定格式日志）

4. RecursiveCharacterTextSplitter：递归降级核心思想
   - 维护「语义粒度从粗到细」的分隔符优先级列表，默认 ["\n\n", "\n", " ", ""] = 段落 → 行 → 词 → 字符
   - 优先最粗粒度切分保留语义；某块超 chunk_size 则对该块递归换更细分隔符
   - 降级到 "" 按单字符强制切，保证一定满足约束——「递归」来源

5. _split_text 算法流程（理解重点）
   - 步骤1 选定分隔符：从列表头遍历用 re.search 测试，命中第一个即用，并把更细分隔符记入 new_separators 供递归；遇 "" 直接选用兜底
   - 步骤2 执行切分：_split_text_with_regex 按选定分隔符切开得 splits
   - 步骤3 分类处理：长度 < chunk_size 累积进 _good_splits 缓冲；≥ chunk_size 的超长块先合并输出缓冲，再递归 _split_text(s, new_separators)
   - 步骤4 收尾：循环结束合并残留 _good_splits

6. keep_separator 与 _split_text_with_regex 正则技巧
   - keep_separator=True 用 re.split(f"({separator})", text) 捕获组保留分隔符
   - 随后两两拼回「内容+分隔符」，使标点附着前片段尾
   - 意义：标点本身携带语义边界，丢弃影响可读性

7. is_separator_regex 与 re.escape
   - False（默认）：对分隔符调 re.escape，把 . | ? 转义为字面量，避免误解释
   - True：不转义，分隔符当正则直接用，可写 "。|！|？" 一次匹配多标点
   - 常见错误：想用正则却忘设 True，导致模式被转义失效

8. from_language 与代码分割
   - get_separators_for_language(language) 返回预置列表，如 Python ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
   - 设计：优先在类/函数语法边界切，避免切碎函数体
   - from_language 内部自动设 is_separator_regex=True

9. 与其他分割器选型对比
   - CharacterTextSplitter：规整文本，不推荐通用
   - RecursiveCharacterTextSplitter：通用首选，兼顾语义与大小
   - TokenTextSplitter：需严格控制 token 数
   - SemanticChunker：基于嵌入相似度在语义漂移处切，成本更高
   - MarkdownHeaderTextSplitter：按标题层级切并保留层级元数据

===================================================================================
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_text_splitters.base import Language, TextSplitter


class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters.

    单一分隔符分割器：只用一个固定分隔符切开全文。
    注意它不保证块大小——若文本中不含该分隔符，会返回一个超长块。
    """

    def __init__(
            self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter.

        参数：
            separator: str            - 分隔符，默认 "\n\n"（段落边界）
            is_separator_regex: bool  - 是否把 separator 当作正则模式
                                        False 时会被 re.escape 转义为字面量
            **kwargs: Any             - 透传给基类，如 chunk_size、chunk_overlap、
                                        length_function、keep_separator
        """
        # 调用基类构造，由基类接管 chunk_size / chunk_overlap 等通用配置
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks.

        参数：
            text: str - 待分割的原始文本

        返回：
            List[str] - 分割后的文本块列表
        """
        # First we naively split the large input into a bunch of smaller ones.
        # 非正则模式下用 re.escape 转义元字符，保证分隔符按字面量匹配
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )

        # 按分隔符切开，得到细碎片段列表
        splits = _split_text_with_regex(text, separator, self._keep_separator)

        # keep_separator=True 时分隔符已附着在片段内，合并时无需再补
        # 否则合并时需要用原分隔符把片段重新连接起来
        _separator = "" if self._keep_separator else self._separator

        # 委托基类把碎片按 chunk_size 合并成尽量饱满的块（并处理 chunk_overlap）
        return self._merge_splits(splits, _separator)


def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """按正则分隔符切分文本，可选保留分隔符

    参数：
        text: str            - 待切分文本
        separator: str       - 正则形式的分隔符（调用方已完成转义处理）
        keep_separator: bool - 是否把分隔符保留在结果片段中

    返回：
        List[str] - 切分后的片段列表（已过滤空串）
    """
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            # 模式外加括号构成捕获组，re.split 会把分隔符一并放入结果列表
            # 例如 re.split("(,)", "a,b") → ["a", ",", "b"]
            _splits = re.split(f"({separator})", text)

            # 结果列表形如 [内容0, 分隔符0, 内容1, 分隔符1, 内容2, ...]
            # 从索引 1 起以步长 2 遍历，把「内容 + 其后分隔符」两两拼接
            # 使分隔符附着在前一片段末尾，保留标点携带的边界语义
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

            # 偶数长度说明末尾是「内容」而非分隔符，需补上最后一个片段
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]

            # 索引 0 的片段在上面的推导式中被跳过（range 从 1 开始），此处补回开头
            splits = [_splits[0]] + splits
        else:
            # 不保留分隔符：直接按模式切分，分隔符被丢弃
            splits = re.split(separator, text)
    else:
        # separator 为空串：降级为按单字符切分（递归的最终兜底）
        splits = list(text)

    # 过滤空串，避免产生无意义的空片段
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.

    递归字符分割器：按「粗 → 细」的分隔符优先级列表逐级降级切分。
    先用粗粒度分隔符最大程度保留语义，对仍然超长的块再换更细的分隔符递归处理。
    这是 LangChain 中最通用、最推荐的文本分割器。
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter.

        参数：
            separators: Optional[List[str]] - 分隔符优先级列表，按语义粒度从粗到细排列
                                              默认 ["\n\n", "\n", " ", ""]
                                              对应 段落 → 行 → 词 → 字符
            keep_separator: bool            - 是否在结果中保留分隔符，默认 True
                                              保留可维持标点携带的边界语义
            is_separator_regex: bool        - 分隔符是否为正则模式
                                              False 时会被 re.escape 转义为字面量
            **kwargs: Any                   - 透传给基类，如 chunk_size、chunk_overlap
        """
        super().__init__(keep_separator=keep_separator, **kwargs)

        # 未传入时使用默认列表，末尾的 "" 是兜底项，保证递归一定能终止
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks.

        递归分割的核心实现。

        参数：
            text: str              - 当前待分割的文本
            separators: List[str]  - 本层可用的分隔符列表（递归时传入更细的子集）

        返回：
            List[str] - 满足 chunk_size 约束的文本块列表

        算法概要：
            1. 从 separators 中挑选第一个在 text 中出现的分隔符
            2. 用它切分文本
            3. 对切出的每个片段分类处理：
               - 未超长的累积待合并
               - 超长的用更细分隔符递归拆分
            4. 合并累积的短片段并输出
        """
        final_chunks = []

        # Get appropriate separator to use
        # 步骤 1：挑选本层使用的分隔符
        # 默认取列表最后一项（通常是 ""）作为兜底，防止循环未命中
        separator = separators[-1]

        # new_separators 保存「比当前分隔符更细」的子列表，供递归调用使用
        new_separators = []

        for i, _s in enumerate(separators):
            # 非正则模式下转义元字符，保证按字面量匹配
            _separator = _s if self._is_separator_regex else re.escape(_s)

            # 空串是最后的兜底项，必然可用，直接选定且不再有更细粒度
            if _s == "":
                separator = _s
                break

            # re.search 测试该分隔符是否存在于文本中
            # 命中即选定，并把后续更细的分隔符记入 new_separators
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        # 步骤 2：按选定分隔符执行切分
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        # 步骤 3：逐片段分类处理
        # _good_splits 是「未超长片段」的缓冲区，累积到一定量后交给 _merge_splits 合并
        _good_splits = []

        # keep_separator=True 时分隔符已附着在片段内，合并时无需再补
        _separator = "" if self._keep_separator else separator

        for s in splits:
            # 情况 A：片段未超长，放入缓冲区等待合并
            # _length_function 默认为 len，也可替换为 token 计数函数
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)

            # 情况 B：片段仍然超长，需要进一步处理
            else:
                # 先把缓冲区已累积的短片段合并输出
                # 这一步很关键：保证最终块的顺序与原文一致
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []

                # B1：已无更细的分隔符可用，只能原样输出这个超长块
                # 此时该块会超出 chunk_size，属于无法避免的情况
                if not new_separators:
                    final_chunks.append(s)

                # B2：递归调用自身，用更细的分隔符继续拆分这个超长块
                # 这是「递归字符分割器」名称的由来
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        # 步骤 4：收尾，合并缓冲区残留的短片段
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """TextSplitter 抽象方法实现，作为递归的入口

        参数：
            text: str - 待分割的原始文本

        返回：
            List[str] - 分割后的文本块列表

        说明：
            从完整的 _separators 列表开始递归，后续每层递归会收到更细的子列表
        """
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
            cls, language: Language, **kwargs: Any
    ) -> RecursiveCharacterTextSplitter:
        """按编程语言的语法结构创建分割器

        参数：
            cls           - 类本身，支持子类继承
            language: Language - 语言枚举，如 Language.PYTHON、Language.JS
            **kwargs: Any - 透传给构造函数，如 chunk_size

        返回：
            RecursiveCharacterTextSplitter - 已配置该语言专用分隔符的实例

        设计要点：
            内部自动设置 is_separator_regex=True，因为语言分隔符
            （如 "\nclass "、"\ndef "）是按正则语义编写的
        """
        # 获取该语言预置的分隔符列表（按类、函数等语法结构边界排列）
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        """返回指定编程语言的分隔符优先级列表

        参数：
            language: Language - 语言枚举值

        返回：
            List[str] - 该语言的分隔符列表，按语法结构粒度从粗到细排列

        设计思路（各语言列表的共同规律）：
            1. 先按顶层结构切分：类定义（"\nclass "）
            2. 再按函数/方法定义切分（"\ndef "、"\nvoid " 等）
            3. 然后按控制流语句切分（"\nif "、"\nfor "、"\nwhile " 等）
            4. 最后退化为通用的 "\n\n" → "\n" → " " → ""
            这样能最大程度保证一个函数体不被切碎，提升代码检索的语义完整性

        异常：
            ValueError - 传入不支持的语言时抛出

        注意：
            这些分隔符按正则语义编写，调用方需设置 is_separator_regex=True
            from_language() 已自动处理该设置
        """
        if language == Language.CPP:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.KOTLIN:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PHP:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\begin{align}",
                "$$",
                "$",
                # Now split by the normal type of lines
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\nabstract ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                # Split along divisions
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # Split along sections within DATA DIVISION
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # Split along sections within PROCEDURE DIVISION
                "\nINPUT-OUTPUT SECTION.",
                # Split along paragraphs and common statements
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n",
                " ",
                "",
            ]
        elif language == Language.LUA:
            return [
                # Split along variable and table definitions
                "\nlocal ",
                # Split along function definitions
                "\nfunction ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.HASKELL:
            return [
                # Split along function definitions
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                # Split along type declarations
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                "\n:: ",
                # Split along module declarations
                "\nmodule ",
                # Split along import statements
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                # Split along typeclass declarations
                "\nclass ",
                "\ninstance ",
                # Split along case expressions
                "\ncase ",
                # Split along guards in function definitions
                "\n| ",
                # Split along record field declarations
                "\ndata ",
                "\n= {",
                "\n, ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )
