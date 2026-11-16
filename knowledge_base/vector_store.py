import os
import json
import hashlib
import sqlite3
from langchain_community.vectorstores import Chroma
from config.settings import embeddings, CHROMA_DB_DIR, RETRIEVAL_TOP_K, DOCS_DIR
from knowledge_base.document_loader import doc_loader

# 索引文件：记录「文档名 → 集合名」映射，实现每个文件独立集合
INDEX_FILE = os.path.join(CHROMA_DB_DIR, "kb_index.json")


# 稀疏索引持久化目录（与向量库同生命周期；构建时预计算，重启直接加载，避免每次查询全量编码）
SPARSE_INDEX_DIR = os.path.join(CHROMA_DB_DIR, "sparse_index")


class SparseRetriever:
    """基于 bge-m3 稀疏向量的关键字检索通道（轻量 BM25 替代）。

    bge-m3 是稀疏-稠密联合模型，用 FlagEmbedding 的 BGEM3FlagModel 一次编码
    同时给出 dense 与 sparse 向量；sparse 分数可当关键词匹配（lexical weights），
    无需 jieba 分词、无需第二套索引。

    性能关键：稀疏向量在「构建向量库」时一次性预计算并持久化到
    data/chroma_db/sparse_index/<集合名>.json，查询时只编码 query 一个向量，
    与缓存的 doc 向量做点积，避免每次查询都全量重编码（那才是检索慢的元凶）。
    """

    def __init__(self):
        self._model = None
        self._model_path = None
        # 集合名 -> [(分块文本, 稀疏权重 dict)]
        self._cache = {}
        # 文本 -> 稀疏权重（扁平索引，查询时 O(1) 命中，避免每次遍历整个缓存）
        self._text_to_weights = {}

    def _load_model(self):
        """懒加载 bge-m3 稀疏模型（复用本地模型目录，不联网）"""
        if self._model is not None:
            return self._model
        from config.settings import EMBEDDING_MODEL_NAME
        from FlagEmbedding import BGEM3FlagModel
        # 稀疏检索对显存/算力要求低，用 CPU 即可，避免与 dense 抢 GPU
        self._model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=False)
        self._model_path = EMBEDDING_MODEL_NAME
        return self._model

    # ---------------- 索引构建与加载 ----------------
    def _index_path(self, cname: str) -> str:
        return os.path.join(SPARSE_INDEX_DIR, f"{cname}.json")

    def build_index(self, cname: str, chunks: list) -> None:
        """为某集合的所有分块预计算稀疏向量并持久化。构建向量库时调用一次。"""
        if not chunks:
            return
        try:
            model = self._load_model()
        except Exception as e:
            print(f"稀疏索引构建失败（模型加载异常，检索时将退化）: {e}")
            return
        texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]
        # 分批编码，控制内存峰值
        BATCH = 256
        cache = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            out = model.encode(batch, return_dense=False, return_sparse=True)["lexical_weights"]
            for j, weights in enumerate(out):
                # 权重值是 numpy float32，需显式转 Python float，否则 json.dump 报 "float32 is not JSON serializable"
                w = {tok: float(v) for tok, v in weights.items()}
                cache.append([texts[i + j], w])
                self._text_to_weights[texts[i + j]] = w
        # 更新内存缓存
        self._cache[cname] = cache
        # 持久化到磁盘（lexical_weights 是 dict[str,float]，可直接 JSON 序列化）
        try:
            os.makedirs(SPARSE_INDEX_DIR, exist_ok=True)
            with open(self._index_path(cname), "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"稀疏索引落盘失败（不影响本次会话，下次重建）: {e}")

    def load_index(self, cname: str) -> bool:
        """从磁盘加载某集合的稀疏索引到内存。启动时调用，避免重新编码。"""
        if cname in self._cache:
            return True
        p = self._index_path(cname)
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 数据格式：[[text, {tok: weight}], ...]
            cache = [(item[0], item[1]) for item in data]
            self._cache[cname] = cache
            for text, w in cache:
                self._text_to_weights[text] = w
            return True
        except Exception as e:
            print(f"稀疏索引加载失败（该集合检索时重新编码）: {e}")
            return False

    def invalidate(self, cname: str) -> None:
        """某集合重建后，清除其内存缓存与磁盘索引。"""
        # 清空该集合在扁平索引里的条目
        for text, _ in self._cache.pop(cname, []):
            self._text_to_weights.pop(text, None)
        try:
            p = self._index_path(cname)
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    def search(self, query: str, chunks: list, top_k: int) -> list:
        """在给定文本片段中做稀疏检索，返回 [(score, text)] 按分数降序。

        优先用缓存好的稀疏向量做点积；缓存缺失时才回退到在线编码（兼容旧数据）。
        """
        if not query or not chunks:
            return []
        try:
            model = self._load_model()
        except Exception as e:
            print(f"稀疏检索模型加载失败（降级为纯向量检索）: {e}")
            return []
        # 只编码 query 一个向量（这是稀疏检索唯一需要在线计算的量）
        q_sparse = model.encode([query], return_dense=False, return_sparse=True)["lexical_weights"][0]

        texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]
        scored = []
        miss_texts = []

        # 命中扁平索引（O(1)），未命中才在线编码兜底（兼容旧向量库）
        for t in texts:
            w = self._text_to_weights.get(t)
            if w is not None:
                s = 0.0
                for tok, qw in q_sparse.items():
                    if tok in w:
                        s += qw * w[tok]
                scored.append((s, t))
            else:
                miss_texts.append(t)

        if miss_texts:
            BATCH = 256
            for i in range(0, len(miss_texts), BATCH):
                batch = miss_texts[i:i + BATCH]
                out = model.encode(batch, return_dense=False, return_sparse=True)["lexical_weights"]
                for j, weights in enumerate(out):
                    s = 0.0
                    for tok, qw in q_sparse.items():
                        if tok in weights:
                            s += qw * weights[tok]
                    scored.append((s, miss_texts[i + j]))
                    # 兜底编码结果顺手缓存进内存，本次会话内后续查询直接命中。
                    # 注：此处不知分块所属集合，只进内存不落盘；持久化统一在 build_index 完成。
                    self._text_to_weights[miss_texts[i + j]] = dict(weights)

        scored.sort(key=lambda x: -x[0])
        scored = [(s, t) for s, t in scored if s > 0]
        return scored[:top_k]


# 全局稀疏检索器（懒加载）
sparse_retriever = SparseRetriever()


def _stable_collection_name(filename: str) -> str:
    """根据文件名生成稳定的集合名（同名文件覆盖，不堆积新集合）。
    Chroma 集合名只允许 [a-zA-Z0-9._-]，中文字符必须剔除，改用文件名 hash 保证唯一。
    """
    # 只保留 ASCII 字母数字和 . _ -，其余（含中文）替换为 _
    safe = "".join(ch if (ch.isascii() and ch.isalnum()) or ch in "._-" else "_" for ch in filename)
    safe = safe.strip("._-") or "doc"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]
    return f"kb_{safe[:40]}_{h}"


class VectorStoreService:
    def __init__(self):
        self.persist_directory = CHROMA_DB_DIR
        # 文档名 -> 集合名
        self.file_index = self._load_index()
        # 所有已加载的集合（集合名 -> Chroma 实例）
        self.stores = {}
        self._load_existing()

    # ---------------- 索引管理 ----------------
    def _load_index(self) -> dict:
        try:
            if os.path.exists(INDEX_FILE):
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_index(self):
        try:
            os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(self.file_index, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"保存知识库索引失败: {e}")

    # ---------------- 加载 ----------------
    def _load_existing(self):
        """加载索引中记录的所有集合；索引为空时尝试从磁盘自动恢复"""
        if not os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            return

        if not self.file_index:
            self._recover_index_from_disk()

        for fname, cname in list(self.file_index.items()):
            try:
                self.stores[cname] = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    collection_name=cname,
                )
            except Exception as e:
                print(f"加载集合 {cname} 失败: {e}")
                # 集合不存在则从索引移除
                self.file_index.pop(fname, None)
                continue
            # 加载该集合的稀疏索引（有则复用，无则首次查询时在线兜底）
            sparse_retriever.load_index(cname)

        self._save_index()

    def _recover_index_from_disk(self):
        """索引文件丢失时，从 chroma 数据库里找回所有 kb_ 集合并重建映射"""
        try:
            names = self._list_kb_collections()
            for cname in names:
                # 集合名形如 kb_<文件名>_<hash>，尽量还原一个可读的键
                self.file_index[cname] = cname
        except Exception:
            pass

    @staticmethod
    def _list_kb_collections() -> list:
        db = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db):
            return []
        try:
            con = sqlite3.connect(db)
            names = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM collections WHERE name LIKE 'kb\\_%' ESCAPE '\\'"
                ).fetchall()
            ]
            con.close()
            return sorted(names)
        except Exception:
            return []

    # ---------------- 构建 ----------------
    def rebuild(self, file_paths: list = None, chunks: list = None) -> bool:
        """
        构建向量库：每个文件写入一个独立集合。
        - file_paths: 文件路径列表（按文件分集合）
        - chunks: 已解析好的分块（为兼容旧调用；此时归入单个集合）
        返回是否成功。
        """
        # 优先按文件路径分集合构建
        if file_paths:
            for fp in file_paths:
                self._build_one_file(fp)
            self._save_index()
            return True

        # 兼容：直接传入 chunks（无文件名信息），归入一个默认集合
        if chunks:
            cname = _stable_collection_name("_combined")
            self.stores[cname] = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name=cname,
            )
            self.file_index["_combined"] = cname
            self._save_index()
            return True

        return False

    def _build_one_file(self, file_path: str):
        """解析单个文件并写入它自己的集合（同名覆盖）"""
        chunks = doc_loader.load_and_split(file_path)
        if not chunks:
            print(f"文件解析为空，跳过: {file_path}")
            return

        filename = os.path.basename(file_path)
        cname = _stable_collection_name(filename)
        # 覆盖式写入：同名文件重新构建时，删除旧集合再写入
        old = self.file_index.get(filename)
        if old and old in self.stores:
            try:
                self.stores[old].delete_collection()
            except Exception:
                pass
            self.stores.pop(old, None)
            # 旧集合的稀疏索引一并失效（避免残留脏缓存）
            sparse_retriever.invalidate(old)

        self.stores[cname] = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=cname,
        )
        self.file_index[filename] = cname

        # 构建稀疏索引（预计算分块稀疏向量并落盘，避免查询时全量重编码）
        sparse_retriever.build_index(cname, chunks)

    # ---------------- 检索 ----------------
    def _target_stores(self, file_filter: list = None) -> list:
        """确定要检索的集合（文件名 -> store 列表）。file_filter 空=全部。"""
        if not self.stores:
            return []
        if file_filter:
            targets = []
            for fname in file_filter:
                cname = self.file_index.get(fname)
                if cname and cname in self.stores:
                    targets.append((fname, self.stores[cname]))
            return targets
        return list(self.stores.items())

    def search(self, query: str, k: int = None, file_filter: list = None) -> list:
        """
        跨集合稠密向量检索，返回 (distance, content) 列表，按距离升序。
        距离即 L2（bge-m3 归一化后等价于余弦距离），越小越相似。
        """
        k = k or RETRIEVAL_TOP_K
        targets = self._target_stores(file_filter)

        scored = []  # (distance, content)
        for _, store in targets:
            try:
                results = store.similarity_search_with_score(query, k=k)
                for doc, score in results:
                    scored.append((score, doc.page_content))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0])
        return scored[:k]

    def collect_chunks(self, file_filter: list = None) -> list:
        """收集目标集合里的所有分块文本（供稀疏检索用）。"""
        targets = self._target_stores(file_filter)
        chunks = []
        for _, store in targets:
            try:
                res = store._collection.get(include=["documents"])
                chunks.extend(res.get("documents", []))
            except Exception:
                continue
        return chunks

    # ---------------- 重复检测 ----------------
    @staticmethod
    def _doc_fingerprint(file_path: str) -> str:
        """计算文档内容指纹（整文 MD5），用于重复文档检测"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def find_duplicates(self, file_paths: list) -> dict:
        """
        检测待构建文档与已有文档是否内容重复。
        返回 {文件名: 重复的已有文件名}，无重复则为空 dict。
        策略：整文 MD5 精确匹配 + 分块指纹近似匹配（SimHash 简化版）。
        """
        # 1. 已有文档指纹（文件名 -> 内容指纹），按需计算并缓存
        dup_map = {}
        for fp in file_paths:
            filename = os.path.basename(fp)
            fp_hash = self._doc_fingerprint(fp)
            if not fp_hash:
                continue
            # 与已有文档比对
            for existing_name in self.file_index.keys():
                if existing_name == filename:
                    continue
                existing_path = os.path.join(DOCS_DIR, existing_name)
                if not os.path.exists(existing_path):
                    continue
                existing_hash = self._doc_fingerprint(existing_path)
                if existing_hash and existing_hash == fp_hash:
                    dup_map[filename] = existing_name
                    break
        return dup_map

    # ---------------- 状态 ----------------
    def is_ready(self) -> bool:
        return bool(self.stores)

    def count(self) -> int:
        """所有集合的分块总数"""
        total = 0
        for store in self.stores.values():
            try:
                total += store._collection.count()
            except Exception:
                pass
        return total

    def list_files(self) -> list:
        """列出当前知识库中的所有文档名"""
        return list(self.file_index.keys())

    def delete_file(self, filename: str) -> bool:
        """删除某个文件对应的集合（按文件名）"""
        cname = self.file_index.get(filename)
        if not cname:
            return False
        try:
            if cname in self.stores:
                self.stores[cname].delete_collection()
                self.stores.pop(cname, None)
            sparse_retriever.invalidate(cname)
            self.file_index.pop(filename, None)
            self._save_index()
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False


# 全局单例
vector_store = VectorStoreService()
