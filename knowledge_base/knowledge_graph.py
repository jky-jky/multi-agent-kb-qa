import os
import json
import time
import hashlib
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import kg_llm, llm, KG_DIR, KG_EXTRACT_CONCURRENCY, KG_SAVE_EVERY, KG_EXTRACT_MAX_RETRY, KG_MAX_ENTITIES, KG_MAX_RELATIONS
from langchain_core.prompts import ChatPromptTemplate
from knowledge_base.document_loader import extract_refs


# 按文档隔离的图谱存储目录与索引文件
GRAPHS_DIR = os.path.join(KG_DIR, "graphs")
INDEX_FILE = os.path.join(KG_DIR, "kg_index.json")


def _stable_doc_key(filename: str) -> str:
    """根据文件名生成稳定的文档键（与向量库集合命名规则同源，便于一一对应）。

    只保留 ASCII 字母数字和 . _ -，其余（含中文）替换为 _；结尾加 md5 前 12 位保证唯一。
    """
    safe = "".join(ch if (ch.isascii() and ch.isalnum()) or ch in "._-" else "_" for ch in filename)
    safe = safe.strip("._-") or "doc"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:40]}_{h}"


class KnowledgeGraphService:
    """按文档隔离的知识图谱服务。

    每个文档一张独立的 DiGraph，持久化到 data/kg/graphs/<doc_key>.json；
    增量抽取的已处理分块记录（md5）也按文档拆分，避免换存储路径后
    全局 hash 把新图误判为"全部已处理"而建成空图。
    """

    def __init__(self):
        # doc_key -> nx.DiGraph
        self.graphs = {}
        # doc_key -> set(已处理分块 md5)
        self.processed = {}
        # 致命错误（余额不足/鉴权失败）时中止整轮抽取
        self._abort = False
        self.fatal_error = ""

        # 文档抽取Prompt：抽实体和关系，并约束实体类型与输出规模（防止无节制枚举导致输出Token失控）
        self.extract_prompt = ChatPromptTemplate.from_template("""
从以下文档片段中抽取【最重要】的实体和它们之间的关系，只输出严格JSON格式，不要其他任何文字。

实体要求：
- 实体是具体的概念、模块、组件、事物、名词、机构、法条编号等。
- 禁止把邮箱、电话、网址、纯数字、日期、金额等字面值作为实体。
- 实体名应简洁、归一（同一概念用同一写法，不重复造节点）。
- 只抽取最能代表本片段核心含义的实体，舍弃次要、边缘的实体。

关系要求：两个实体之间明确的动作、依赖、对比、因果等联系。

【出处（ref）标注要求，必须遵守】：
- 如果原文片段里能看出该关系对应的条文/章节编号（如「第X条」「第X章」「第X节」），
  在该关系的 "ref" 字段里填**该关系真正对应的那一个编号**（如「第四十八条」），
  不要把这个片段里出现的所有编号都堆上去。
- 一条关系通常对应一个编号；只有当该关系确实横跨多个编号时才填多个（用数组）。
- 无法确定该关系对应哪个编号时，"ref" 填空数组 []，不要编造编号。
- 若片段里根本没有「第X条」这类编号（如普通文章、技术文档），所有关系的 ref 都填空数组。

【硬性数量上限，必须遵守】：
- 实体最多 {max_entities} 个。
- 关系最多 {max_relations} 条。
- 宁缺毋滥：只保留最有价值的实体与关系，不要为凑数而抽取。

输出格式示例：
{{"entities": ["实体1", "实体2"], "relations": [{{"head": "实体1", "relation": "依赖", "tail": "实体2", "ref": ["第X条"]}}]}}

文档片段：
{content}
""")

        # 问题意图分类与实体抽取合并：一次调用同时产出意图与实体（省一次 NER 调用）
        self.intent_prompt = ChatPromptTemplate.from_template("""
你是知识库问答系统的意图识别器。请分析用户问题，输出严格JSON，不要其他文字。

1. 判断问题意图，从以下枚举中选一个：
   - answer：事实查询/解释（"X是什么""为什么"）
   - summarize：总结/提炼/概括（"总结一下""归纳"）
   - compare：对比/分析（"区别""异同""对比"）
   - plan：规划/生成（"制定计划""写方案""安排"）
   - review：审核/校验（"检查""审核""校验"）
   - rewrite：改写/润色（"改写""润色""扩写""缩写"）
   - extract：提取/结构化（"提取""列出所有""整理成表"）
   - memory：记忆指令（"记住""记一下"等，要求系统记住偏好）

2. 从问题中提取核心实体名词（用于知识图谱检索），没有则返回空数组。

输出格式：
{{"intent": "answer", "entities": ["实体1", "实体2"]}}

用户问题：
{query}
""")

        # 预编译链
        self.chain = self.extract_prompt | kg_llm
        self.intent_chain = self.intent_prompt | llm

        self._load()

    # ---------------- 持久化 ----------------
    def _graph_path(self, doc_key: str) -> str:
        return os.path.join(GRAPHS_DIR, f"{doc_key}.json")

    def _hash_path(self, doc_key: str) -> str:
        return os.path.join(GRAPHS_DIR, f"{doc_key}.hashes.json")

    def _load(self):
        """从磁盘加载所有文档的图谱与已处理记录"""
        try:
            if os.path.exists(INDEX_FILE):
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {}
        except Exception:
            index = {}

        os.makedirs(GRAPHS_DIR, exist_ok=True)
        # 若索引为空，尝试从 graphs 目录自动恢复
        if not index:
            for name in os.listdir(GRAPHS_DIR):
                if name.endswith(".json") and not name.endswith(".hashes.json"):
                    index[name[:-5]] = name[:-5]

        for doc_key in index:
            gpath = self._graph_path(doc_key)
            try:
                g = nx.DiGraph()
                if os.path.exists(gpath):
                    with open(gpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for node in data.get("nodes", []):
                        g.add_node(node)
                    for edge in data.get("edges", []):
                        src, dst = edge.get("source"), edge.get("target")
                        if src and dst:
                            g.add_edge(src, dst, relation=edge.get("relation", ""),
                                       ref=edge.get("ref", []))
                self.graphs[doc_key] = g
            except Exception as e:
                print(f"加载图谱 {doc_key} 失败: {e}")

            # 加载已处理分块记录
            hpath = self._hash_path(doc_key)
            try:
                if os.path.exists(hpath):
                    with open(hpath, "r", encoding="utf-8") as f:
                        self.processed[doc_key] = set(json.load(f))
            except Exception:
                self.processed[doc_key] = set()

        self._save_index()

    def _save_index(self):
        try:
            os.makedirs(GRAPHS_DIR, exist_ok=True)
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted(self.graphs.keys()), f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"保存图谱索引失败: {e}")

    def _save_graph(self, doc_key: str):
        try:
            os.makedirs(GRAPHS_DIR, exist_ok=True)
            g = self.graphs[doc_key]
            payload = {
                "nodes": list(g.nodes),
                "edges": [
                    {"source": u, "target": v, "relation": d.get("relation", ""), "ref": d.get("ref", [])}
                    for u, v, d in g.edges(data=True)
                ],
            }
            with open(self._graph_path(doc_key), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            with open(self._hash_path(doc_key), "w", encoding="utf-8") as f:
                json.dump(sorted(self.processed.get(doc_key, set())), f)
        except Exception as e:
            print(f"保存图谱 {doc_key} 失败: {e}")

    # ---------------- 状态 ----------------
    @property
    def node_count(self) -> int:
        return sum(len(g.nodes) for g in self.graphs.values())

    @property
    def edge_count(self) -> int:
        return sum(len(g.edges) for g in self.graphs.values())

    @property
    def doc_count(self) -> int:
        return len(self.graphs)

    def count_pending(self, doc_key: str, chunks: list) -> int:
        """统计某文档还有多少分块未抽取（用于预估费用）"""
        done = self.processed.get(doc_key, set())
        n = 0
        for chunk in chunks:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            if self._chunk_hash(text) not in done:
                n += 1
        return n

    # ---------------- 构建 ----------------
    @staticmethod
    def _chunk_hash(text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_json(text: str):
        """容错解析模型返回的JSON"""
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return json.loads(text)

    def build_from_documents(self, doc_key: str, chunks: list, max_chunks: int = None,
                             progress_cb=None, concurrency: int = None) -> dict:
        """对单个文档并发增量构建知识图谱，返回统计信息。"""
        stats = {"total": len(chunks), "skipped": 0, "processed": 0, "failed": 0, "fatal": ""}

        self._abort = False
        self.fatal_error = ""

        if doc_key not in self.graphs:
            self.graphs[doc_key] = nx.DiGraph()
        done_hashes = self.processed.setdefault(doc_key, set())

        todo = []
        for chunk in chunks:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            h = self._chunk_hash(text)
            if h in done_hashes:
                stats["skipped"] += 1
            else:
                todo.append((h, text))

        if max_chunks and len(todo) > max_chunks:
            todo = todo[:max_chunks]

        total = len(todo)
        if total == 0:
            return stats

        workers = concurrency or KG_EXTRACT_CONCURRENCY
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._extract_once, text): (h, text) for h, text in todo}
            for fut in as_completed(futures):
                h, text = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    # 线程内未捕获的异常透传到这里，记录下来便于排查
                    res = None
                    if not self.fatal_error:
                        self.fatal_error = f"抽取线程异常：{str(e)[:200]}"

                done += 1
                if res is None:
                    stats["failed"] += 1
                else:
                    entities, relations = res
                    # 整块出处兜底：仅当关系未自带 ref 时使用，且只取第一条
                    # （一个分块通常以某条为主，避免把无关条文全堆到每条边上）
                    block_refs = extract_refs(text)
                    g = self.graphs[doc_key]
                    for ent in entities:
                        g.add_node(ent)
                    for head, relation, tail, rel_ref in relations:
                        g.add_node(head)
                        g.add_node(tail)
                        # 关系自带出处优先；否则用整块第一条兜底；都没有则空
                        ref = rel_ref if rel_ref else (block_refs[:1] if block_refs else [])
                        g.add_edge(head, tail, relation=relation, ref=ref)
                    done_hashes.add(h)
                    stats["processed"] += 1

                if progress_cb:
                    progress_cb(done, total)
                if done % KG_SAVE_EVERY == 0:
                    self._save_graph(doc_key)

        self._save_graph(doc_key)
        self._save_index()
        stats["fatal"] = self.fatal_error
        return stats

    def _extract_once(self, text: str):
        """调用一次模型抽取实体关系（带少量重试），不改动图谱，线程安全。"""
        if self._abort:
            return None

        last_err = ""
        for attempt in range(KG_EXTRACT_MAX_RETRY + 1):
            try:
                result = self.chain.invoke({
                    "content": text,
                    "max_entities": KG_MAX_ENTITIES,
                    "max_relations": KG_MAX_RELATIONS,
                })
                data = self._parse_json(result.content)

                entities = [
                    e.strip()
                    for e in data.get("entities", [])
                    if isinstance(e, str) and e.strip()
                ]
                relations = []
                for rel in data.get("relations", []):
                    if not isinstance(rel, dict):
                        continue
                    head = (rel.get("head") or "").strip()
                    tail = (rel.get("tail") or "").strip()
                    relation = (rel.get("relation") or "").strip()
                    if head and tail and relation:
                        # 出处：优先用模型为该关系标注的 ref，其次才是整块兜底
                        ref = rel.get("ref") or []
                        if not isinstance(ref, list):
                            ref = [ref]
                        ref = [r for r in ref if isinstance(r, str) and r.strip()]
                        relations.append((head, relation, tail, ref))
                return entities, relations
            except Exception as e:
                message = str(e)
                last_err = message
                # 打印真实异常到控制台，便于定位（Gradio 界面只显示统计，不显示细节）
                print(f"[图谱抽取失败] {type(e).__name__}: {message[:300]}", flush=True)
                if "Insufficient Balance" in message or "402" in message:
                    self.fatal_error = "API 账户余额不足（402 Insufficient Balance），请先充值后重试"
                    self._abort = True
                    return None
                if "401" in message or "Unauthorized" in message or "invalid_api_key" in message:
                    self.fatal_error = "API Key 无效或未授权（401），请检查 .env 中的配置"
                    self._abort = True
                    return None
                # 429 限流：并发过高时 DeepSeek 会限流，属于可恢复的瞬时错误，
                # 但重试仍频繁失败时应提醒用户降低并发，而非静默记 failed。
                if "429" in message or "rate limit" in message.lower() or "too many" in message.lower():
                    self.fatal_error = (
                        f"触发 API 限流（429），建议降低图谱抽取并发（KG_EXTRACT_CONCURRENCY，当前 {KG_EXTRACT_CONCURRENCY}）。"
                        f"部分分块已成功落盘，降低并发后重试会自动跳过已抽取块，不重复计费。"
                    )
                    self._abort = True
                    return None
                if attempt < KG_EXTRACT_MAX_RETRY:
                    time.sleep(1.5 * (attempt + 1))
        # 重试耗尽仍未成功：记录最后一次错误，便于排查（而非完全静默）
        if last_err:
            self.fatal_error = f"抽取失败（重试 {KG_EXTRACT_MAX_RETRY} 次仍失败）：{last_err[:200]}"
        return None

    @staticmethod
    def filenames_to_doc_keys(filenames: list) -> list:
        """把文档名列表映射为 doc_key 列表（用于检索时隔离）。"""
        return [_stable_doc_key(f) for f in filenames] if filenames else []

    # ---------------- 意图与实体抽取 ----------------
    def classify_intent(self, query: str) -> dict:
        """一次 LLM 调用同时产出意图标签与核心实体。失败时回退规则判断。"""
        try:
            result = self.intent_chain.invoke({"query": query})
            data = self._parse_json(result.content)
            intent = (data.get("intent") or "").strip()
            entities = [e for e in data.get("entities", []) if isinstance(e, str) and e.strip()]
            if intent in {"answer", "summarize", "compare", "plan", "review", "rewrite", "extract", "memory"}:
                return {"intent": intent, "entities": entities}
        except Exception:
            pass
        # 回退：默认 answer，实体为空
        return {"intent": "answer", "entities": []}

    # ---------------- 图谱检索 ----------------
    @staticmethod
    def _format_ref(ref: list) -> str:
        """把边的出处列表格式化为「出处：第X条、第Y条」；无出处返回空串。"""
        if not ref:
            return ""
        return "（出处：" + "、".join(ref) + "）"

    def query_relations(self, entities: list, doc_keys: list = None) -> str:
        """在指定文档的图谱（默认全部）中查询实体关系，返回结构化文本（附出处）。"""
        targets = doc_keys if doc_keys else list(self.graphs.keys())
        # 只在有对应图谱的文档键里查
        targets = [k for k in targets if k in self.graphs]

        # 过滤：只保留在图谱中存在的实体
        present = []
        for e in entities:
            if any(self.graphs[k].has_node(e) for k in targets):
                present.append(e)
        if not present:
            return ""

        parts = []
        for doc_key in targets:
            g = self.graphs[doc_key]
            # 单实体：同时查后继与前置（出边+入边），修复原实现只看出边导致入边丢失的问题
            if len(present) == 1:
                entity = present[0]
                if not g.has_node(entity):
                    continue
                for neighbor in set(g.successors(entity)) | set(g.predecessors(entity)):
                    if g.has_edge(entity, neighbor):
                        ed = g.get_edge_data(entity, neighbor)
                        rel = ed.get("relation", "")
                        ref = self._format_ref(ed.get("ref", []))
                        parts.append(f"- {entity} {rel} {neighbor}{ref}")
                    if g.has_edge(neighbor, entity):
                        ed = g.get_edge_data(neighbor, entity)
                        rel = ed.get("relation", "")
                        ref = self._format_ref(ed.get("ref", []))
                        parts.append(f"- {neighbor} {rel} {entity}{ref}")
            else:
                # 多实体：两两查最短路径
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        if not (g.has_node(present[i]) and g.has_node(present[j])):
                            continue
                        try:
                            path = nx.shortest_path(g, present[i], present[j])
                            path_desc = []
                            for k in range(len(path) - 1):
                                edge_data = g.get_edge_data(path[k], path[k + 1])
                                path_desc.append(f"{path[k]} →[{edge_data['relation']}]→ {path[k + 1]}")
                            # 取路径上最后一条边的出处作为整条路径的出处
                            last_ref = self._format_ref(g.get_edge_data(path[-2], path[-1]).get("ref", []))
                            parts.append("关系路径：" + " ；".join(path_desc) + last_ref)
                        except (nx.NetworkXNoPath, KeyError):
                            continue

        return "\n".join(dict.fromkeys(parts))  # 去重保序

    # ---------------- 清理 ----------------
    def clear(self):
        """清空所有文档图谱与记录。"""
        for doc_key in list(self.graphs.keys()):
            gpath = self._graph_path(doc_key)
            hpath = self._hash_path(doc_key)
            for p in (gpath, hpath):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        self.graphs.clear()
        self.processed.clear()
        self._save_index()

    def delete_doc(self, doc_key: str):
        """删除某文档的图谱与记录。"""
        self.graphs.pop(doc_key, None)
        self.processed.pop(doc_key, None)
        for p in (self._graph_path(doc_key), self._hash_path(doc_key)):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self._save_index()

    def is_empty(self) -> bool:
        return self.node_count == 0


# 全局单例
kg_service = KnowledgeGraphService()
