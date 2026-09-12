"""
API 集成测试

用 FastAPI TestClient 测试端点。
外部依赖（Neo4j/ChromaDB）通过 mock 跳过连接，不依赖 Docker。
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """构造 TestClient，mock 外部服务连接（Neo4j / ChromaDB 不在测试环境运行）。"""
    import api.main as api_module

    # Mock 服务连接，让 lifespan 不报错
    api_module.vector_store.init = AsyncMock()
    api_module.vector_store.get_stats = AsyncMock(
        return_value={"backend": "chroma", "total_vectors": 42}
    )
    api_module.knowledge_graph.init = AsyncMock()
    api_module.knowledge_graph.close = AsyncMock()
    api_module.knowledge_graph.get_stats = AsyncMock(
        return_value={"entities": 100, "relations": 250}
    )
    api_module.knowledge_graph.execute_cypher = AsyncMock(
        return_value=[{"1": 1}]
    )

    with TestClient(api_module.app) as c:
        yield c


class TestHealthEndpoint:
    """健康检查 — 验证响应结构和状态码"""

    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "AgentKnowledgeHub"
        assert "neo4j" in data["services"]
        assert "vector_store" in data["services"]

    def test_has_status_field(self, client):
        r = client.get("/api/health")
        assert r.json()["status"] in ("ok", "degraded")


class TestQAEndpoint:
    """问答端点 — 验证路由和参数校验"""

    def test_missing_question_returns_422(self, client):
        r = client.post("/api/qa/ask", json={})
        assert r.status_code == 422

    def test_forwards_thread_id_to_qa_agent(self, client, monkeypatch):
        """API 不能再把不同会话折叠到 qa-default。"""
        import api.main as api_module
        from agents.qa_agent import QAResult, QueryIntent

        fake_qa = MagicMock()
        fake_qa.answer = AsyncMock(return_value=QAResult(
            question="张三是谁？",
            answer="张三是员工。",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.0,
        ))
        monkeypatch.setitem(api_module.workflows, "qa", fake_qa)

        response = client.post(
            "/api/qa/ask",
            json={"question": "张三是谁？", "thread_id": "thread-from-client"},
        )

        assert response.status_code == 200
        fake_qa.answer.assert_awaited_once_with(
            "张三是谁？", thread_id="thread-from-client"
        )
        assert response.json()["thread_id"] == "thread-from-client"

    def test_generates_and_returns_thread_id_for_anonymous_question(self, client, monkeypatch):
        """客户端需要拿到服务端生成的 ID，才可在下一轮继续同一段对话。"""
        import api.main as api_module
        from agents.qa_agent import QAResult, QueryIntent

        fake_qa = MagicMock()
        fake_qa.answer = AsyncMock(return_value=QAResult(
            question="第一问",
            answer="回答",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.0,
        ))
        monkeypatch.setitem(api_module.workflows, "qa", fake_qa)

        response = client.post("/api/qa/ask", json={"question": "第一问"})

        assert response.status_code == 200
        generated_thread_id = response.json()["thread_id"]
        assert generated_thread_id and generated_thread_id != "qa-default"
        fake_qa.answer.assert_awaited_once_with("第一问", thread_id=generated_thread_id)

    def test_qa_openapi_example_omits_optional_thread_id(self, client):
        """Swagger 首问示例不应提交占位字符串 'string' 作为会话 ID。"""
        schema = client.get("/openapi.json").json()
        example = schema["components"]["schemas"]["QuestionRequest"]["examples"][0]

        assert example == {"question": "北京地区住宿标准是多少？报销期限是多久？"}

    def test_qa_response_hides_internal_document_paths(self, client, monkeypatch):
        """对外 evidence 只暴露逻辑文件名，内部 provenance 不应成为 API 路径泄露。"""
        import api.main as api_module
        from agents.qa_agent import QAResult, QueryIntent, RetrievedContext

        fake_qa = MagicMock()
        fake_qa.answer = AsyncMock(return_value=QAResult(
            question="制度是什么？",
            answer="见制度文件。",
            contexts=[RetrievedContext(
                content="制度原文",
                source="/app/uploads/interview_test.txt",
                score=0.8,
                retrieval_type="vector",
                metadata={
                    "source": "/app/uploads/interview_test.txt",
                    "chunk_index": 0,
                },
            )],
            intent=QueryIntent.FACTOID,
            confidence=0.8,
        ))
        monkeypatch.setitem(api_module.workflows, "qa", fake_qa)

        response = client.post("/api/qa/ask", json={"question": "制度是什么？"})

        assert response.status_code == 200
        source = response.json()["sources"][0]
        assert source["source"] == "interview_test.txt"
        assert source["metadata"]["source"] == "interview_test.txt"


class TestIngestEndpoint:
    """文档入库端点"""

    def test_upload_txt(self, client):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                r = client.post(
                    "/api/ingest/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                )
            # 接受 200 或 500（取决于 LLM 是否可用）
            assert r.status_code in (200, 500)
        finally:
            os.unlink(tmp_path)


class TestUploadSafety:
    """上传接口必须在落盘和调用工作流之前收紧不可信输入。"""

    @pytest.mark.parametrize("filename", ["../escape.txt", r"..\escape.txt", r"C:\temp\escape.txt"])
    def test_rejects_path_like_filename(self, filename):
        import api.main as api_module
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            api_module._validate_filename(filename)

        assert exc_info.value.status_code == 400

    def test_rejects_unsupported_extension_before_workflow(self, client, monkeypatch, tmp_path):
        import api.main as api_module

        monkeypatch.setattr(api_module.settings, "upload_dir", str(tmp_path))
        response = client.post(
            "/api/ingest/upload",
            files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
        )

        assert response.status_code == 415
        assert list(tmp_path.iterdir()) == []

    def test_rejects_oversized_replace_without_changing_existing_file(self, client, monkeypatch, tmp_path):
        import api.main as api_module

        monkeypatch.setattr(api_module.settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(api_module.settings, "max_upload_size_mb", 1)
        target = tmp_path / "report.txt"
        target.write_bytes(b"old version")

        response = client.put(
            "/api/files/report.txt",
            files={"file": ("report.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )

        assert response.status_code == 413
        assert target.read_bytes() == b"old version"
        assert [path.name for path in tmp_path.iterdir()] == ["report.txt"]

    def test_rejects_mismatched_put_file_extension(self, client, monkeypatch, tmp_path):
        import api.main as api_module

        monkeypatch.setattr(api_module.settings, "upload_dir", str(tmp_path))
        response = client.put(
            "/api/files/report.txt",
            files={"file": ("report.pdf", b"not really pdf", "application/pdf")},
        )

        assert response.status_code == 400
        assert list(tmp_path.iterdir()) == []

    def test_failed_index_update_restores_file_and_hides_internal_error(self, client, monkeypatch, tmp_path):
        import api.main as api_module

        monkeypatch.setattr(api_module.settings, "upload_dir", str(tmp_path))
        target = tmp_path / "report.txt"
        target.write_bytes(b"old version")
        update_workflow = MagicMock()
        update_workflow.ainvoke = AsyncMock(side_effect=RuntimeError("database password leaked"))
        monkeypatch.setitem(api_module.workflows, "update", update_workflow)

        response = client.put(
            "/api/files/report.txt",
            files={"file": ("report.txt", b"new version", "text/plain")},
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "文件更新失败，请稍后重试"
        assert target.read_bytes() == b"old version"
        assert [path.name for path in tmp_path.iterdir()] == ["report.txt"]

    def test_limits_batch_file_count_before_processing(self, client, monkeypatch):
        import api.main as api_module

        monkeypatch.setattr(api_module.settings, "max_batch_upload_files", 1)
        response = client.post(
            "/api/ingest/batch",
            files=[
                ("files", ("first.txt", b"first", "text/plain")),
                ("files", ("second.txt", b"second", "text/plain")),
            ],
        )

        assert response.status_code == 413

    def test_admin_update_cannot_target_arbitrary_local_path(self, client):
        response = client.post(
            "/api/admin/update",
            json={"file_path": r"..\outside.txt", "change_type": "modified"},
        )

        assert response.status_code == 400

    def test_admin_update_rejects_unknown_change_type(self, client):
        response = client.post(
            "/api/admin/update",
            json={"file_path": "report.txt", "change_type": "overwrite_everything"},
        )

        assert response.status_code == 422


class TestDocumentLifecycleEndpoints:
    """修改接口不得再按不完整的文本行 diff 伪增量写入。"""

    def test_update_uses_full_replace_result(self, client, monkeypatch, tmp_path):
        import api.main as api_module
        from agents.knowledge_update_agent import ChangeType, DocumentChange, UpdateResult

        monkeypatch.setattr(api_module.settings, "upload_dir", str(tmp_path))
        (tmp_path / "report.txt").write_text("old report content", encoding="utf-8")
        change = DocumentChange(
            file_path=str(tmp_path / "report.txt"),
            change_type=ChangeType.MODIFIED,
        )
        update_result = UpdateResult(
            change=change,
            chunks_processed=2,
            vectors_added=2,
            vectors_deleted=3,
            graph_records_deleted=4,
            entities_added=1,
            relations_added=1,
            events_added=1,
            update_mode="full_replace",
        )
        update_workflow = MagicMock()
        update_workflow.ainvoke = AsyncMock(return_value={"results": [update_result]})
        monkeypatch.setitem(api_module.workflows, "update", update_workflow)

        response = client.put(
            "/api/files/report.txt",
            files={"file": ("report.txt", b"new report content", "text/plain")},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "full_replace"
        assert response.json()["chunks_count"] == 2
        sent_change = update_workflow.ainvoke.await_args.args[0]["changes"][0]
        assert sent_change.change_type == ChangeType.MODIFIED
        assert sent_change.diff_chunks == []


class TestAdminStats:
    """管理统计端点"""

    def test_stats(self, client):
        r = client.get("/api/admin/stats")
        assert r.status_code == 200
        data = r.json()
        assert "vector_store" in data
        assert "knowledge_graph" in data
