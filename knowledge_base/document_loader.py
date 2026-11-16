import os
import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.common import clean_text


# 中文数字（用于「第X条/章/节/编/款/项」的出处提取）
_CN_NUM = "零一二三四五六七八九十百千万"
_REF_PAT = re.compile(r"第[%s0-9]+(?:条|章|节|编|款|项)" % _CN_NUM)
# markdown 标题（# 或 ## 等，最多 6 级）
_MD_HEADING_PAT = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
# markdown 代码块围栏（``` 或 ~~~）
_MD_FENCE_PAT = re.compile(r"^```|^~~~", re.MULTILINE)


def extract_refs(text: str) -> list:
    """从一段文本里提取条文/章节出处（如「第四十八条」「第三章」），去重保序。

    供切分后的每个分块标注出处；提取不到返回空列表（文档不分章节条款时降级）。
    """
    seen = []
    for m in _REF_PAT.findall(text or ""):
        if m not in seen:
            seen.append(m)
    return seen


def extract_md_heading(text: str) -> str:
    """提取一段文本里最靠前的 markdown 标题作为出处（无标题返回空串）。"""
    m = _MD_HEADING_PAT.search(text or "")
    return m.group(1).strip() if m else ""


class DocumentLoaderService:
    # 支持的文件格式
    SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

    def __init__(self):
        # 通用兜底切分器：仍按句号/换行切，供无法识别结构的文档使用
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        # 条文型切分器：块略大（条文通常比普通句长），优先按「条」边界切
        self.clause_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        # 技术型切分器：块更大（函数/章节语义单元更长）
        self.tech_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    # ---------------- 类型探测 ----------------
    @staticmethod
    def _detect_doc_type(text: str, has_table: bool = False) -> str:
        """探测文档类型，返回 'clause' | 'tech' | 'table' | 'plain'。

        依据优先级：表格 > 条文 > 技术(markdown) > 普通。
        """
        if has_table:
            return "table"
        # 条文型：正文中出现较多「第X条/章/节」这类编号
        refs = extract_refs(text)
        if len(refs) >= 3:
            return "clause"
        # 技术型：有 markdown 标题或代码围栏
        if _MD_HEADING_PAT.search(text) or _MD_FENCE_PAT.search(text):
            return "tech"
        return "plain"

    # ---------------- 读取 ----------------
    @staticmethod
    def _read_text_file(file_path: str) -> str:
        """读取文本文件，自动兼容 utf-8 / gbk 等编码"""
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

    @staticmethod
    def _read_docx(file_path: str):
        """解析 Word，返回 (正文段落列表, 表格列表)。表格单独保留结构，避免被拍平切散。"""
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return paragraphs, tables

    # ---------------- 加载 ----------------
    def load_file(self, file_path: str) -> list:
        """加载单个文档文件，返回 Document 列表（保持接口兼容，不做分块）。"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(file_path).load()
                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)
                return docs
            elif ext in (".txt", ".md", ".markdown"):
                text = self._read_text_file(file_path)
                return [Document(page_content=text, metadata={"source": file_path})]
            elif ext == ".docx":
                paragraphs, tables = self._read_docx(file_path)
                # 正文与表格合并：表格以「表头|表头2 + 行」结构化文本形式保留
                parts = list(paragraphs)
                for tbl in tables:
                    header = tbl[0]
                    parts.append("表格：" + " | ".join(header))
                    for row in tbl[1:]:
                        parts.append(" | ".join(row))
                text = "\n".join(parts)
                return [Document(page_content=text, metadata={"source": file_path})]
            else:
                raise ValueError(f"不支持的文件格式: {ext}（支持 PDF/TXT/MD/DOCX）")
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return []

    def split_documents(self, documents: list) -> list:
        """对文档分块（按文档类型自适应），并给每块标注出处 refs。

        返回的 Document 保留 page_content 与 metadata（含 source、refs、doc_type）。
        """
        out = []
        for doc in documents:
            text = doc.page_content
            if not text:
                continue
            doc_type = self._detect_doc_type(text)
            splitter = {
                "tech": self.tech_splitter,
                "plain": self.fallback_splitter,
                "clause": self.clause_splitter,
                "table": self.clause_splitter,
            }[doc_type]

            chunks = splitter.split_text(text)
            for c in chunks:
                c = clean_text(c)
                if not c:
                    continue
                refs = extract_refs(c)
                heading = extract_md_heading(c) if doc_type == "tech" else ""
                meta = dict(doc.metadata)
                meta["doc_type"] = doc_type
                # Chroma 要求 metadata 的 list 值非空，空 refs 会导致 upsert 报错
                # （ValueError: Expected metadata list value for key 'refs' to be non-empty）
                if refs:
                    meta["refs"] = refs
                if heading:
                    meta["heading"] = heading
                out.append(Document(page_content=c, metadata=meta))
        return out

    def load_and_split(self, file_path: str) -> list:
        """加载 + 自适应分块一站式处理（保持既有调用签名不变）。"""
        docs = self.load_file(file_path)
        if not docs:
            return []
        return self.split_documents(docs)


# 全局单例
doc_loader = DocumentLoaderService()
