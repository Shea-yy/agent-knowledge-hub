"""
DocParser Agent 单元测试

测试分块逻辑、文件类型识别 — 纯逻辑，不调 LLM。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from agents.doc_parser_agent import DocParserAgent, DocType


class TestClassify:
    """文件类型识别"""

    def test_pdf(self):
        agent = DocParserAgent()
        assert agent._classify("report.pdf") == DocType.PDF

    def test_image_png(self):
        agent = DocParserAgent()
        assert agent._classify("photo.png") == DocType.IMAGE

    def test_image_jpg(self):
        agent = DocParserAgent()
        assert agent._classify("photo.jpg") == DocType.IMAGE

    def test_csv(self):
        agent = DocParserAgent()
        assert agent._classify("data.csv") == DocType.TABLE

    def test_excel(self):
        agent = DocParserAgent()
        assert agent._classify("report.xlsx") == DocType.TABLE

    def test_text(self):
        agent = DocParserAgent()
        assert agent._classify("readme.txt") == DocType.TEXT

    def test_markdown(self):
        agent = DocParserAgent()
        assert agent._classify("README.md") == DocType.MARKDOWN

    def test_unknown(self):
        agent = DocParserAgent()
        assert agent._classify("file.xyz") == DocType.UNKNOWN

    def test_case_insensitive(self):
        agent = DocParserAgent()
        assert agent._classify("REPORT.PDF") == DocType.PDF


class TestMakeDocId:
    """文档 ID 生成"""

    def test_deterministic(self):
        """同一路径 → 同一 ID"""
        id1 = DocParserAgent._make_doc_id("/path/to/doc.pdf")
        id2 = DocParserAgent._make_doc_id("/path/to/doc.pdf")
        assert id1 == id2

    def test_different_paths(self):
        """不同路径 → 不同 ID"""
        id1 = DocParserAgent._make_doc_id("/path/a.pdf")
        id2 = DocParserAgent._make_doc_id("/path/b.pdf")
        assert id1 != id2

    def test_length(self):
        """ID 长度 16（SHA256 前 16 位）"""
        doc_id = DocParserAgent._make_doc_id("/any/path.pdf")
        assert len(doc_id) == 16

    def test_revision_changes_with_file_content(self):
        """内容变化应更新 revision_id，但不改变稳定的 doc_id。"""
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(b"first revision")
            path = file.name

        try:
            doc_id = DocParserAgent._make_doc_id(path)
            revision_1 = DocParserAgent._make_revision_id(path)
            with open(path, "wb") as file:
                file.write(b"second revision")
            revision_2 = DocParserAgent._make_revision_id(path)

            assert DocParserAgent._make_doc_id(path) == doc_id
            assert revision_1 != revision_2
        finally:
            os.unlink(path)


class TestChunkTexts:
    """文档分块逻辑"""

    def test_chunk_count(self, sample_text):
        agent = DocParserAgent()
        # 修改 chunk size 让测试可控
        agent.CHUNK_SIZE = 100
        agent.CHUNK_OVERLAP = 20

        chunks = agent._chunk_texts([sample_text], "doc-001", DocType.TEXT, "/test/file.txt")

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 100

    def test_chunk_metadata(self, sample_text):
        agent = DocParserAgent()
        agent.CHUNK_SIZE = 200
        agent.CHUNK_OVERLAP = 20

        chunks = agent._chunk_texts([sample_text], "doc-001", DocType.PDF, "/docs/report.pdf")

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.doc_id == "doc-001"
            assert chunk.doc_type == DocType.PDF
            assert chunk.chunk_index == i
            assert chunk.metadata["source"] == "/docs/report.pdf"
            assert "char_start" in chunk.metadata
            assert "char_end" in chunk.metadata

    def test_chunk_id_format(self, sample_text):
        agent = DocParserAgent()
        agent.CHUNK_SIZE = 200

        chunks = agent._chunk_texts([sample_text], "abc123", DocType.TEXT, "/t.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"abc123#chunk-{i}"

    def test_empty_text(self):
        agent = DocParserAgent()
        chunks = agent._chunk_texts([""], "doc-001", DocType.TEXT, "/empty.txt")
        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        agent = DocParserAgent()
        agent.CHUNK_SIZE = 1000
        chunks = agent._chunk_texts(["short text"], "doc-001", DocType.TEXT, "/short.txt")
        assert len(chunks) == 1
        assert chunks[0].content == "short text"

    def test_multiple_texts(self, sample_text):
        agent = DocParserAgent()
        agent.CHUNK_SIZE = 300
        texts = ["first document text here", "second document text here"]
        chunks = agent._chunk_texts(texts, "doc-001", DocType.TEXT, "/multi.txt")
        assert len(chunks) >= len(texts)

    def test_preserves_pdf_page_and_revision_metadata(self):
        agent = DocParserAgent()
        chunks = agent._chunk_texts(
            ["第 2 页内容", "第 3 页内容"],
            "doc-001",
            DocType.PDF,
            "/docs/report.pdf",
            revision_id="rev-001",
            page_numbers=[2, 3],
        )

        assert [chunk.metadata["page"] for chunk in chunks] == [2, 3]
        assert {chunk.metadata["revision_id"] for chunk in chunks} == {"rev-001"}


class TestParseText:
    """纯文本解析"""

    def test_reads_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world")
            tmp_path = f.name

        try:
            result = DocParserAgent._parse_text(tmp_path)
            assert result == ["hello world"]
        finally:
            os.unlink(tmp_path)

    def test_reads_gb18030_text(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as file:
            file.write("中文内容".encode("gb18030"))
            tmp_path = file.name

        try:
            assert DocParserAgent._parse_text(tmp_path) == ["中文内容"]
        finally:
            os.unlink(tmp_path)
