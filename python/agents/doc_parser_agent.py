"""
文档解析 Agent — 多模态文档解析，支持 PDF / 图片 / 纯文本

核心能力:
  1. PDF 混合解析（文字页取文本，图表页走视觉，按页码组装）
  2. 图片解析（OCR 探针成本路由：扫描件 0 成本，图表走视觉）
  3. 文档分块（Chunking）与元数据标注

边界说明:
  表格文件 (Excel/CSV) 不经过本模块——由 API 层直接路由到
  services/table_store.py 做结构化存储与精确查询。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

log = logger.bind(module="doc_parser")


class DocType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """一个文档块，携带内容 + 元数据"""
    content: str
    doc_id: str
    chunk_index: int
    doc_type: DocType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#chunk-{self.chunk_index}"


@dataclass(frozen=True)
class ParsedPage:
    """PDF 解析中保留物理页码，供分块与图谱溯源使用。"""
    content: str
    page_number: int | None


class DocParserAgent:
    """
    文档解析 Agent

    工作流:
      classify → parse → chunk → enrich_metadata → output
    """

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
    }

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    VISION_RETRY_ATTEMPTS = 3

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    # ── public API ───────────────────────────────────────────

    async def parse(self, file_path: str) -> list[DocumentChunk]:
        """解析单个文件，返回文档块列表"""
        doc_type = self._classify(file_path)
        doc_id = self._make_doc_id(file_path)
        log.info("解析文件: {} (type={})", file_path, doc_type.value)

        # 边界定义：表格文件不进 RAG 管道，由 API 层路由到 TableStore 做结构化查询
        if doc_type == DocType.TABLE:
            log.info("表格文件跳过 RAG 解析（应走 TableStore 结构化通道）: {}", file_path)
            return []

        revision_id = self._make_revision_id(file_path)
        raw_texts: list[str] = []
        page_numbers: list[int | None] | None = None
        if doc_type == DocType.PDF:
            parsed_pages = await self._parse_pdf(file_path)
            raw_texts = [page.content for page in parsed_pages]
            page_numbers = [page.page_number for page in parsed_pages]
        elif doc_type == DocType.IMAGE:
            raw_texts = await self._parse_image(file_path)
        else:
            raw_texts = self._parse_text(file_path)

        chunks = self._chunk_texts(
            raw_texts,
            doc_id,
            doc_type,
            file_path,
            revision_id=revision_id,
            page_numbers=page_numbers,
        )
        log.info("解析完成: {} → {} chunks", os.path.basename(file_path), len(chunks))
        return chunks

    async def parse_batch(self, file_paths: list[str]) -> list[DocumentChunk]:
        """批量解析多个文件"""
        all_chunks: list[DocumentChunk] = []
        for fp in file_paths:
            all_chunks.extend(await self.parse(fp))
        return all_chunks

    # ── classification ───────────────────────────────────────

    def _classify(self, file_path: str) -> DocType:
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        """文档身份只由规范化路径决定，内容变化由 revision_id 表示。"""
        normalized_path = os.path.normcase(os.path.abspath(file_path))
        return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _make_revision_id(file_path: str) -> str:
        """以文件内容生成版本指纹，不改变稳定的 doc_id。"""
        digest = hashlib.sha256()
        with open(file_path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    # ── PDF parsing ──────────────────────────────────────────

    VISION_MAX_PAGES = 5
    MIN_TEXT_LENGTH = 50  # 文字少于 50 字符的页视为图片页
    MIN_OCR_LENGTH = 30   # OCR 少于 30 字符的图片视为图表，走视觉解析

    async def _parse_pdf(self, file_path: str) -> list[ParsedPage]:
        """
        PDF 混合解析（按页码组装，带成本截断）:
          1. 逐页判定：文字充分的页直接取文本，图片页标记走视觉
          2. 全量无文本 → 走带截断警告的 Fallback
          3. 图文混排 → 视觉页数超过上限时截断，控制 LLM 调用成本
          4. 最终按物理页码顺序组装，保证内容顺序不乱
        """
        page_contents: dict[int, ParsedPage] = {}
        vision_pages: list[int] = []

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            log.info("PDF 共 {} 页，开始混合解析", total_pages)

            for i, page in enumerate(reader.pages):
                page_num = i + 1
                page_text = page.extract_text() or ""
                if len(page_text.strip()) >= self.MIN_TEXT_LENGTH:
                    page_contents[page_num] = ParsedPage(
                        content=f"[第 {page_num} 页]\n{page_text.strip()}",
                        page_number=page_num,
                    )
                else:
                    vision_pages.append(page_num)
        except Exception as e:
            log.error("PDF 文本提取失败: {} → {}", file_path, e)
            return await self._pdf_vision_fallback(file_path, max_pages=self.VISION_MAX_PAGES)

        # 全量扫描件：直接走带截断警告的 Fallback
        if len(vision_pages) == total_pages:
            log.warning("检测到全量无文本 PDF，直接交由 Fallback 处理")
            return await self._pdf_vision_fallback(file_path, max_pages=self.VISION_MAX_PAGES)

        # 图文混排：图片页数量截断，防止成本失控
        if vision_pages:
            if len(vision_pages) > self.VISION_MAX_PAGES:
                log.warning(
                    "图文混排的图片页过多 ({} 页)，已截断至前 {} 页",
                    len(vision_pages), self.VISION_MAX_PAGES,
                )
                vision_pages = vision_pages[:self.VISION_MAX_PAGES]

            log.info("准备对以下页码进行视觉解析: {}", vision_pages)
            try:
                vision_texts = await self._vision_parse_pages(file_path, vision_pages)
                for parsed_page in vision_texts:
                    page_contents[parsed_page.page_number] = parsed_page
            except Exception as e:
                log.error("PDF 混合视觉解析局部失败: {}", e)
                for page_num in vision_pages:
                    page_contents[page_num] = ParsedPage(
                        content=f"[PDF 视觉解析失败 - 第 {page_num} 页] {str(e)}",
                        page_number=page_num,
                    )

        # 按物理页码顺序组装
        return [page_contents[num] for num in sorted(page_contents.keys())]

    async def _vision_parse_pages(self, file_path: str, pages: list[int]) -> list[ParsedPage]:
        """将指定页渲染为图片并用 LLM 视觉解析（PyMuPDF 渲染，无系统依赖）"""
        import io
        import fitz
        from PIL import Image

        texts: list[ParsedPage] = []
        doc = fitz.open(file_path)
        try:
            for p in pages:
                page = doc[p - 1]  # fitz 页码从 0 开始
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                description = await self._describe_image_with_llm(img)
                texts.append(ParsedPage(
                    content=f"[第 {p} 页 视觉]\n{description}",
                    page_number=p,
                ))
        finally:
            doc.close()
        return texts

    async def _pdf_vision_fallback(self, file_path: str, max_pages: int = 5) -> list[ParsedPage]:
        """当 PDF 纯文本提取失败时，使用 LLM 视觉能力"""
        try:
            import io
            import fitz
            from PIL import Image

            doc = fitz.open(file_path)
            try:
                total_pages = doc.page_count
                end_page = min(total_pages, max_pages)

                texts: list[ParsedPage] = []
                for i in range(end_page):
                    pix = doc[i].get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    description = await self._describe_image_with_llm(img)
                    texts.append(ParsedPage(
                        content=f"[第 {i + 1} 页 视觉]\n{description}",
                        page_number=i + 1,
                    ))
            finally:
                doc.close()

            # 触发截断时追加警告，诚实告知用户
            if total_pages > max_pages:
                texts.append(ParsedPage(
                    content=(
                        f"[系统警告] 文档共 {total_pages} 页，"
                        f"受限于系统策略，视觉解析已在第 {max_pages} 页截断。"
                    ),
                    page_number=None,
                ))
            return texts
        except Exception as e:
            log.error("PDF 视觉回退失败: {} → {}", file_path, e)
            return [ParsedPage(
                content=f"[PDF 视觉解析失败] {file_path} - 错误信息: {str(e)}",
                page_number=None,
            )]

    # ── image parsing ────────────────────────────────────────

    async def _parse_image(self, file_path: str) -> list[str]:
        """
        图片混合解析 (成本路由版)：OCR 探针 + LLM 按需兜底

        工程考量与妥协：
          1. 优先使用本地极低成本的 OCR 提取文本。
          2. 若 OCR 提取字符数 < MIN_OCR_LENGTH，推断为数据图表或架构图，
             动态拉起大模型进行视觉解析。
          3. 妥协盲区：刻意舍弃"长文本包围小图表"场景的图表召回率，
             以换取极佳的响应速度和 API 成本控制。
        """
        texts: list[str] = []

        # 1. 本地 OCR（既是基础提取器，也是免费探针）
        ocr_text = self._ocr(file_path)
        if ocr_text.strip():
            texts.append(f"[OCR 提取]\n{ocr_text.strip()}")

        # 2. 基于字符长度的成本路由判定
        if len(ocr_text.strip()) < self.MIN_OCR_LENGTH:
            try:
                from PIL import Image
                # 图片 IO 后置加载：纯文字扫描件不消耗内存
                img = Image.open(file_path)
                description = await self._describe_image_with_llm(img)
                texts.append(f"[图表语义解析]\n{description}")
            except Exception as e:
                # 防止图片损坏或 API 超时导致整个 LangGraph 节点崩溃
                log.error("图片视觉解析失败: {} → {}", file_path, e)
                texts.append(f"[图表视觉解析失败] 发生异常: {str(e)}")

        # 3. 数据完整性兜底（绝对禁止返回空列表）
        return texts or [f"[系统警告] 未能从图像提取任何有效特征 (请检查文件是否损坏): {file_path}"]

    @staticmethod
    def _ocr(file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(file_path), lang="chi_sim+eng")
        except Exception:
            return ""

    @retry(
        stop=stop_after_attempt(VISION_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            httpx.TransportError,
        )),
        reraise=True,
    )
    async def _describe_image_with_llm(self, image: Any) -> str:
        """调用 LLM 多模态能力描述图片内容（先压缩，避免大图传输失败）"""
        import base64
        import io

        # 压缩：最长边限 1024px，JPEG 质量 75（LLM 内部本来就会缩图，不影响理解）
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        if max(img.size) > 1024:
            ratio = 1024 / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("图片压缩: {}x{} → {} KB", img.width, img.height, len(b64) // 1024)

        messages = [
            SystemMessage(content="你是一个专业的文档分析助手，请详细描述图片中的内容，包括文字、表格、图表信息。"),
            HumanMessage(content=[
                {"type": "text", "text": "请描述这张图片的所有内容："},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    # ── text / markdown ──────────────────────────────────────

    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        with open(file_path, "rb") as file:
            raw = file.read()

        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return [raw.decode(encoding)]
            except UnicodeDecodeError:
                continue

        log.warning("文本编码无法可靠识别，使用 UTF-8 替换非法字符: {}", file_path)
        return [raw.decode("utf-8", errors="replace")]

    # ── chunking ─────────────────────────────────────────────

    def _chunk_texts(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
        revision_id: str = "",
        page_numbers: list[int | None] | None = None,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        idx = 0
        for text_index, text in enumerate(texts):
            page_number = (
                page_numbers[text_index]
                if page_numbers is not None and text_index < len(page_numbers)
                else None
            )
            start = 0
            while start < len(text):
                end = start + self.CHUNK_SIZE
                content = text[start:end]
                if content.strip():
                    metadata = {
                        "source": source,
                        "char_start": start,
                        "char_end": end,
                    }
                    if revision_id:
                        metadata["revision_id"] = revision_id
                    if page_number is not None:
                        metadata["page"] = page_number
                    chunks.append(DocumentChunk(
                        content=content.strip(),
                        doc_id=doc_id,
                        chunk_index=idx,
                        doc_type=doc_type,
                        metadata=metadata,
                    ))
                    idx += 1
                start = end - self.CHUNK_OVERLAP
        return chunks
