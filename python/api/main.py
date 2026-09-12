"""
FastAPI 入口 — 企业知识管理系统 REST API

提供三组接口:
  1. /api/ingest   — 文档上传 & 入库
  2. /api/qa       — 智能问答 (ReAct Agent)
  3. /api/admin    — 管理（统计、更新触发）

改进点 (vs 原版):
  - loguru 结构化日志，按模块分流
  - 启动异常不再静默吞掉，记录到日志 + 暴露到 /health
  - /health 真实检测 Neo4j / ChromaDB 连通性
  - HTTP 请求日志中间件
"""

from __future__ import annotations

import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge_update_agent import ChangeType, DocumentChange
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.table_store import TableStore
from services.vector_store import VectorStoreService

# ── 日志初始化 ──────────────────────────────────────────────

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "<level>{message}</level>"
    ),
)

# ── 全局服务状态（用于健康检查） ────────────────────────────

_service_status: dict[str, str] = {
    "neo4j": "unknown",
    "vector_store": "unknown",
    "workflows": "unknown",
}

# ── 服务实例 ────────────────────────────────────────────────

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
table_store = TableStore()
workflows: dict[str, Any] = {}


TABLE_EXTENSIONS = frozenset({".xlsx", ".xls", ".csv"})
SUPPORTED_UPLOAD_EXTENSIONS = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".xls", ".txt", ".md",
})
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_PATH_METADATA_KEYS = frozenset({"source", "file_path", "path", "document_path"})


def _display_source(source: str) -> str:
    """将文档来源转为 API 可展示名称，避免泄露容器或主机文件路径。"""
    source_text = str(source)
    normalized = source_text.replace("\\", "/")
    if Path(normalized).suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS:
        return PureWindowsPath(normalized).name
    return source_text


def _public_metadata(value: Any, key: str | None = None) -> Any:
    """递归清理 metadata 中的文档路径，不修改内部检索对象。"""
    if isinstance(value, dict):
        return {str(item_key): _public_metadata(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_public_metadata(item, key) for item in value]
    if isinstance(value, str) and key in _PATH_METADATA_KEYS:
        return _display_source(value)
    return value


def _validate_filename(filename: str | None) -> str:
    """校验外部传入的文件名，只允许上传目录的一层普通文件。"""
    if not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="缺少文件名")

    name = filename
    windows_path = PureWindowsPath(name)
    if (
        not name
        or not name.strip()
        or "\x00" in name
        or any(ord(char) < 32 for char in name)
        or name in {".", ".."}
        or name != Path(name).name
        or name != windows_path.name
        or windows_path.drive
        or windows_path.root
        or name.endswith((".", " "))
    ):
        raise HTTPException(status_code=400, detail="文件名非法：不允许路径、盘符或保留结尾字符")

    if Path(name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"不支持的文件类型。仅支持：{allowed}")
    return name


def _resolve_upload_path(filename: str) -> Path:
    """将已验证文件名解析到上传根目录，并再次验证解析结果未逃逸。"""
    upload_root = Path(settings.upload_dir).resolve()
    destination = (upload_root / filename).resolve()
    if destination.parent != upload_root:
        # 防御纵深：即使未来放宽文件名规则，也不允许写出上传根目录。
        raise HTTPException(status_code=400, detail="文件路径超出上传目录")
    return destination


def _make_backup(save_path: Path) -> Path | None:
    """为覆盖写创建同目录备份，避免把旧文件完整读入内存。"""
    if not save_path.exists():
        return None
    if not save_path.is_file():
        raise HTTPException(status_code=409, detail="同名上传目标不是普通文件")

    backup_path = save_path.with_name(f".{save_path.name}.{uuid4().hex}.backup")
    try:
        shutil.copy2(save_path, backup_path)
    except OSError as exc:
        logger.error("创建文件备份失败: {} → {}", save_path, exc)
        raise HTTPException(status_code=500, detail="创建文件备份失败，请稍后重试") from exc
    return backup_path


def _cleanup_file(path: Path | None) -> None:
    """尽力清理临时文件；清理失败不掩盖主流程错误。"""
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError as exc:
        logger.warning("清理临时文件失败: {} → {}", path, exc)


def _rollback_file_replace(save_path: Path, backup_path: Path | None) -> None:
    """索引或表格处理失败时恢复旧磁盘版本；新文件则删除。"""
    _cleanup_file(save_path)
    if backup_path is not None and backup_path.is_file():
        try:
            backup_path.replace(save_path)
        except OSError as exc:
            logger.error("上传回滚失败: {} → {}", save_path, exc)


async def _save_upload_file(file: UploadFile, save_path: Path) -> int:
    """以临时文件流式接收上传，达到限额前不触碰正式文件。"""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_name(f".{save_path.name}.{uuid4().hex}.upload")
    received = 0

    try:
        with temp_path.open("xb") as target:
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大：单文件上限为 {settings.max_upload_size_mb} MiB",
                    )
                target.write(chunk)
        if received == 0:
            raise HTTPException(status_code=400, detail="不接受空文件")
        temp_path.replace(save_path)
        return received
    except Exception:
        _cleanup_file(temp_path)
        raise


# ── Lifespan ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化 → 运行 → 清理"""
    log = logger.bind(module="lifespan")

    # 1. 上传目录
    os.makedirs(settings.upload_dir, exist_ok=True)
    log.info(f"上传目录已就绪: {settings.upload_dir}")

    # 2. 向量库
    try:
        await vector_store.init()
        _service_status["vector_store"] = "connected"
        log.info(f"向量库已连接 ({settings.vector_store_type})")
    except Exception as e:
        _service_status["vector_store"] = f"disconnected: {e}"
        log.error(f"向量库初始化失败: {e}")

    # 3. 知识图谱
    try:
        await knowledge_graph.init()
        _service_status["neo4j"] = "connected"
        log.info(f"Neo4j 已连接 ({settings.neo4j_uri})")
    except Exception as e:
        _service_status["neo4j"] = f"disconnected: {e}"
        log.error(f"Neo4j 初始化失败: {e}")

    # 4. 编译 LangGraph 工作流
    try:
        workflows.update(
            build_knowledge_graph_workflow(
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
                table_store=table_store,
            )
        )
        _service_status["workflows"] = "compiled"
        log.info("LangGraph 工作流编译完成 (ingest / qa / update)")
    except Exception as e:
        _service_status["workflows"] = f"failed: {e}"
        log.error(f"工作流编译失败: {e}")

    log.info("服务启动完成，监听 {}:{}", settings.api_host, settings.api_port)
    yield

    # 清理
    await knowledge_graph.close()
    log.info("服务已关闭")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="AgentKnowledgeHub — 多Agent企业知识管理系统",
    description="支持多模态RAG、GraphRAG混合检索、CDC增量更新、ReAct Agent的企业级知识管理 API",
    version="2.0.0",
    lifespan=lifespan,
)


# ── 请求日志中间件 ──────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的方法、路径、耗时、状态码"""
    log = logger.bind(module="http")
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    log.info(
        "{} {} → {} ({:.0f}ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ── Request / Response Models ────────────────────────────────

class QuestionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"question": "北京地区住宿标准是多少？报销期限是多久？"},
        ]
    })

    question: str
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选会话 ID；省略时由服务端创建并在响应中返回",
    )


class QuestionResponse(BaseModel):
    question: str
    thread_id: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    events_count: int = 0
    extraction_failures: list[dict[str, str]] = Field(default_factory=list)
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    # 字段名为兼容已有客户端保留；只接受上传目录内的一层文件名，而不是任意 OS 路径。
    file_path: str = Field(description="上传目录内的文件名")
    change_type: ChangeType = ChangeType.MODIFIED


class UpdateResponse(BaseModel):
    file_path: str
    chunks_processed: int = 0
    vectors_added: int
    vectors_deleted: int
    graph_records_deleted: int = 0
    entities_added: int
    relations_added: int
    events_added: int = 0
    update_mode: str = ""
    extraction_failures: list[dict[str, str]] = Field(default_factory=list)
    success: bool
    processing_time_ms: float


# ═══════════════════════════════════════════════════════════════
# 健康检查（真实检测依赖服务）
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["系统管理"])
async def health():
    """健康检查：检测所有依赖服务的真实连通性"""
    log = logger.bind(module="health")
    services_ok = True

    # -- Neo4j --
    neo4j_ok = _service_status["neo4j"] == "connected"
    if neo4j_ok:
        try:
            await knowledge_graph.execute_cypher("RETURN 1")
        except Exception:
            neo4j_ok = False
            log.warning("Neo4j 健康检查失败：连接已断开")

    # -- 向量库 --
    vs_ok = _service_status["vector_store"] == "connected"
    if vs_ok:
        try:
            await vector_store.get_stats()
        except Exception:
            vs_ok = False
            log.warning("向量库健康检查失败：连接已断开")

    # -- 工作流 --
    wf_ok = _service_status["workflows"] == "compiled"

    all_ok = neo4j_ok and vs_ok and wf_ok
    if not all_ok:
        services_ok = False

    return {
        "status": "ok" if all_ok else "degraded",
        "service": "AgentKnowledgeHub",
        "version": app.version,
        "services": {
            "neo4j": "connected" if neo4j_ok else _service_status["neo4j"],
            "vector_store": "connected" if vs_ok else _service_status["vector_store"],
            "workflows": "compiled" if wf_ok else _service_status["workflows"],
        },
    }


# ═══════════════════════════════════════════════════════════════
# 文档入库
# ═══════════════════════════════════════════════════════════════

@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    """上传文档；同名非表格文件按稳定 doc_id 做安全替换。"""
    log = logger.bind(module="ingest")
    filename = _validate_filename(file.filename)
    save_path = _resolve_upload_path(filename)
    existed_before = save_path.is_file()
    backup_path = _make_backup(save_path)

    try:
        await _save_upload_file(file, save_path)
        log.info("文件已安全保存: {}", save_path)
        ext = save_path.suffix.lower()

        # 表格文件：只走结构化存储，不经过 RAG 管道。
        if ext in TABLE_EXTENSIONS:
            import pandas as pd
            if ext == ".csv":
                try:
                    df = pd.read_csv(save_path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(save_path, encoding="gb18030")
            else:
                df = pd.read_excel(save_path)
            info = table_store.add_table(filename, df, source=str(save_path))
            response = IngestResponse(
                file_name=filename,
                chunks_count=0,
                entities_count=0,
                relations_count=0,
                status=f"table_stored: {info['row_count']} rows, {len(info['columns'])} cols",
            )

        # 同名上传不是“再入库”：否则文档缩短时旧 Chunk/Event 会残留。
        # 复用更新工作流，使 POST 与 PUT 具有同一份清理语义。
        elif existed_before:
            update_wf = workflows.get("update")
            if not update_wf:
                raise HTTPException(status_code=503, detail="更新工作流未就绪")
            state = await update_wf.ainvoke(
                {"changes": [DocumentChange(file_path=str(save_path), change_type=ChangeType.MODIFIED)]},
                config={"configurable": {"thread_id": f"replace-{filename}"}},
            )
            results = state.get("results", [])
            if not results or not results[0].success:
                error = results[0].error if results else "更新未返回结果"
                raise RuntimeError(error)
            update_result = results[0]
            response = IngestResponse(
                file_name=filename,
                chunks_count=update_result.chunks_processed,
                entities_count=update_result.entities_added,
                relations_count=update_result.relations_added,
                events_count=update_result.events_added,
                extraction_failures=update_result.extraction_failures,
                status="partial_success" if update_result.extraction_failures else update_result.update_mode,
            )

        else:
            ingest_wf = workflows.get("ingest")
            if not ingest_wf:
                raise HTTPException(status_code=503, detail="入库工作流未就绪")
            result = await ingest_wf.ainvoke(
                {"file_paths": [str(save_path)]},
                config={"configurable": {"thread_id": f"ingest-{filename}"}},
            )
            chunks = result.get("chunks", [])
            extractions = result.get("extractions", [])
            # 唯一计数：跨 chunk 去重已移除（血缘优先），sum 会因重复提及虚高。
            total_entities = len({ent.name for e in extractions for ent in e.entities})
            total_relations = len({
                (r.head, r.relation, r.tail)
                for e in extractions for r in e.relations
            })
            total_events = sum(len(item.events) for item in extractions)
            extraction_failures = result.get("extraction_failures", [])
            response = IngestResponse(
                file_name=filename,
                chunks_count=len(chunks),
                entities_count=total_entities,
                relations_count=total_relations,
                events_count=total_events,
                extraction_failures=extraction_failures,
                status="partial_success" if extraction_failures else "success",
            )
    except HTTPException:
        _rollback_file_replace(save_path, backup_path)
        raise
    except Exception as exc:
        _rollback_file_replace(save_path, backup_path)
        log.error("文档入库失败: {} → {}", filename, exc)
        raise HTTPException(status_code=500, detail="文档入库失败，请稍后重试") from exc
    else:
        _cleanup_file(backup_path)
        log.info("入库完成: {} ({})", filename, response.status)
        return response


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    log = logger.bind(module="ingest")
    if len(files) > settings.max_batch_upload_files:
        raise HTTPException(
            status_code=413,
            detail=f"批量上传文件数不能超过 {settings.max_batch_upload_files}",
        )
    log.info("批量上传 {} 个文件", len(files))
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


# ═══════════════════════════════════════════════════════════════
# 文件管理
# ═══════════════════════════════════════════════════════════════

@app.get("/api/files", tags=["文件管理"])
async def list_files():
    """列出所有已上传的文件（包含来源统计）"""
    log = logger.bind(module="files")
    files = []

    # 磁盘上的上传文件
    upload_dir = settings.upload_dir
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            path = os.path.join(upload_dir, f)
            if os.path.isfile(path):
                files.append({
                    "name": f,
                    "size_kb": round(os.path.getsize(path) / 1024, 1),
                    "source": "uploads",
                })

    # 表格存储
    for t in table_store.list_tables():
        files.append({
            "name": t["name"],
            "size_kb": 0,
            "source": "table_store",
            "rows": t["row_count"],
            "columns": t["columns"],
        })

    log.info("列出 {} 个文件", len(files))
    return {"total": len(files), "files": files}


@app.delete("/api/files/{filename}", tags=["文件管理"])
async def delete_file(filename: str):
    """通过统一文档生命周期删除磁盘、向量和图谱数据。"""
    log = logger.bind(module="files")
    filename = _validate_filename(filename)
    deleted_from = []
    path = _resolve_upload_path(filename)

    # 表格不进入向量/图谱管道，保持其独立的生命周期。
    if path.suffix.lower() in TABLE_EXTENSIONS:
        if table_store.remove_table(filename):
            deleted_from.append("table_store")
        if path.is_file():
            path.unlink()
            deleted_from.append("disk")
        if not deleted_from:
            raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未找到")
        return {"deleted": filename, "from": deleted_from}

    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="更新工作流未就绪")

    # 先清理检索后端。若发生故障，保留磁盘文件，便于调用方重试。
    try:
        state = await update_wf.ainvoke(
            {"changes": [DocumentChange(file_path=str(path), change_type=ChangeType.DELETED)]},
            config={"configurable": {"thread_id": f"delete-{filename}"}},
        )
        results = state.get("results", [])
        if not results or not results[0].success:
            error = results[0].error if results else "更新未返回结果"
            raise RuntimeError(error)
        result = results[0]
    except Exception as e:
        log.error("删除知识库记录失败: {} → {}", filename, e)
        raise HTTPException(status_code=500, detail="删除知识库记录失败，请稍后重试") from e

    if result.vectors_deleted:
        deleted_from.append("vector_store")
    if result.graph_records_deleted:
        deleted_from.append("knowledge_graph")

    # 后端清理成功后再删除本地文件。
    if path.is_file():
        path.unlink()
        deleted_from.append("disk")

    if table_store.remove_table(filename):
        deleted_from.append("table_store")

    if not deleted_from:
        raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未找到")

    log.info("已删除: {} (from: {})", filename, ", ".join(deleted_from))
    return {"deleted": filename, "from": deleted_from}


@app.put("/api/files/{filename}", response_model=IngestResponse, tags=["文件管理"])
async def update_file(filename: str, file: UploadFile = File(...)):
    """
    更新文件并执行安全的文档级替换。

    现有 chunk_id 由位置决定，文本行级 diff 无法可靠地反映 Chunk 边界变化；
    因此修改会先完成新版本解析/抽取，再按稳定 doc_id 清理旧向量和图谱子图。
    """
    log = logger.bind(module="files")
    filename = _validate_filename(filename)
    save_path = _resolve_upload_path(filename)
    ext = save_path.suffix.lower()
    if file.filename:
        uploaded_filename = _validate_filename(file.filename)
        if Path(uploaded_filename).suffix.lower() != ext:
            raise HTTPException(status_code=400, detail="URL 中的文件类型必须与上传文件类型一致")

    existed_before = save_path.is_file()
    backup_path = _make_backup(save_path)
    try:
        await _save_upload_file(file, save_path)

        # 表格文件：直接替换 TableStore 中的旧表，不走 RAG 管道。
        if ext in TABLE_EXTENSIONS:
            import pandas as pd
            if ext == ".csv":
                try:
                    df = pd.read_csv(save_path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(save_path, encoding="gb18030")
            else:
                df = pd.read_excel(save_path)
            info = table_store.add_table(filename, df, source=str(save_path))
            response = IngestResponse(
                file_name=filename,
                chunks_count=0,
                entities_count=0,
                relations_count=0,
                status=f"table_updated: {info['row_count']} rows, {len(info['columns'])} cols",
            )

        else:
            update_wf = workflows.get("update")
            if not update_wf:
                raise HTTPException(status_code=503, detail="更新工作流未就绪")
            change_type = ChangeType.MODIFIED if existed_before else ChangeType.CREATED
            state = await update_wf.ainvoke(
                {"changes": [DocumentChange(file_path=str(save_path), change_type=change_type)]},
                config={"configurable": {"thread_id": f"update-{filename}"}},
            )
            results = state.get("results", [])
            if not results or not results[0].success:
                error = results[0].error if results else "更新未返回结果"
                raise RuntimeError(error)
            result = results[0]
            response = IngestResponse(
                file_name=filename,
                chunks_count=result.chunks_processed,
                entities_count=result.entities_added,
                relations_count=result.relations_added,
                events_count=result.events_added,
                extraction_failures=result.extraction_failures,
                status="partial_success" if result.extraction_failures else result.update_mode,
            )
    except HTTPException:
        _rollback_file_replace(save_path, backup_path)
        raise
    except Exception as exc:
        _rollback_file_replace(save_path, backup_path)
        log.error("文件更新失败: {} → {}", filename, exc)
        raise HTTPException(status_code=500, detail="文件更新失败，请稍后重试") from exc
    else:
        _cleanup_file(backup_path)
        log.info("文件更新完成: {} → {}", filename, response.status)
        return response


# ═══════════════════════════════════════════════════════════════
# 智能问答
# ═══════════════════════════════════════════════════════════════

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — ReAct Agent 自主调用 vector_search / cypher_query / entity_lookup"""
    log = logger.bind(module="qa")

    qa_agent = workflows.get("qa")
    if not qa_agent:
        raise HTTPException(status_code=503, detail="问答工作流未就绪")

    # 匿名请求必须拥有独立的 checkpoint key；不能把所有人折叠到 qa-default。
    thread_id = req.thread_id.strip() if req.thread_id else uuid4().hex

    try:
        qa_result = await qa_agent.answer(
            req.question,
            thread_id=thread_id,
        )
    except Exception as e:
        log.error("问答失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")

    log.info(
        "问答完成: intent={}, confidence={:.2f}, contexts={}",
        qa_result.intent.value, qa_result.confidence, len(qa_result.contexts),
    )

    return QuestionResponse(
        question=qa_result.question,
        thread_id=thread_id,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {
                "content": c.content[:200],
                "source": _display_source(c.source),
                "score": c.score,
                "type": c.retrieval_type,
                "metadata": _public_metadata(c.metadata),
            }
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


# ═══════════════════════════════════════════════════════════════
# 系统管理
# ═══════════════════════════════════════════════════════════════

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取向量库和知识图谱统计信息"""
    log = logger.bind(module="admin")
    try:
        vs_stats = await vector_store.get_stats()
        kg_stats = await knowledge_graph.get_stats()
        log.info("统计查询完成")
        return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)
    except Exception as e:
        log.error("统计查询失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"统计查询失败: {e}")


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest):
    """手动触发上传目录中某个文件的知识库更新。"""
    log = logger.bind(module="admin")

    filename = _validate_filename(req.file_path)
    file_path = _resolve_upload_path(filename)
    if req.change_type != ChangeType.DELETED and not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未找到")

    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="更新工作流未就绪")

    change = DocumentChange(
        file_path=str(file_path),
        change_type=req.change_type,
    )

    try:
        result = await update_wf.ainvoke(
            {"changes": [change]},
            config={"configurable": {"thread_id": f"update-{filename}"}},
        )
    except Exception as e:
        log.error("更新失败: {}", str(e))
        raise HTTPException(status_code=500, detail="更新失败，请稍后重试") from e

    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="更新未返回结果")

    r = results[0]
    log.info(
        "更新完成: {} → +{} vectors, +{} entities",
        r.change.file_path, r.vectors_added, r.entities_added,
    )

    return UpdateResponse(
        file_path=filename,
        chunks_processed=r.chunks_processed,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        graph_records_deleted=r.graph_records_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        events_added=r.events_added,
        update_mode=r.update_mode,
        extraction_failures=r.extraction_failures,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
