from graph.state import AgentState
from config.settings import RETRIEVAL_TOP_K, SPARSE_TOP_K, RRF_K, RELEVANCE_THRESHOLD
from knowledge_base.vector_store import vector_store, sparse_retriever
from knowledge_base.knowledge_graph import kg_service


def _rrf_fuse(ranked_lists: list, k: int = RRF_K) -> list:
    """RRF（Reciprocal Rank Fusion）融合多路检索结果。

    输入：多路结果列表，每路是 [(doc_id, content)]，已按相关度降序。
    返回：按 RRF 分数降序的 content 列表（取前 k）。
    用「名次倒数」而非原始分数融合，因为稠密/稀疏/图谱三路分数不可直接比较。
    """
    scores = {}       # content -> rrf 分数
    order = []        # 记录首次出现顺序，保证稳定
    for ranked in ranked_lists:
        for rank, (_, content) in enumerate(ranked):
            if content not in scores:
                scores[content] = 0.0
                order.append(content)
            scores[content] += 1.0 / (k + rank + 1)

    order.sort(key=lambda c: -scores[c])
    return order[:RETRIEVAL_TOP_K]


def retrieval_agent(state: AgentState) -> AgentState:
    """
    输入：user_query, intent, entities, active_kb_files
    输出：retrieval_context（三通道召回 + RRF 融合）
    """
    query = state.get("user_query", "")
    if not query:
        return {**state, "retrieval_context": "", "error_msg": "无有效查询内容"}

    active_files = state.get("active_kb_files") or []

    # 1. 稠密向量检索：跨集合召回，返回 (distance, content)，按距离升序
    dense_hits = vector_store.search(query, k=RETRIEVAL_TOP_K, file_filter=active_files)

    # 相关性阈值过滤：距离超阈值的片段丢弃，避免硬塞不相关内容（阈值对归一化后的 L2 距离有效）
    dense_hits = [(d, c) for d, c in dense_hits if d <= RELEVANCE_THRESHOLD]

    # 2. 稀疏（关键字）检索：bge-m3 稀疏向量，返回 (score, content)，按分数降序
    sparse_hits = []
    if dense_hits or active_files:  # 有活跃知识库才做稀疏检索
        chunks = vector_store.collect_chunks(file_filter=active_files)
        sparse_hits = sparse_retriever.search(query, chunks, top_k=SPARSE_TOP_K)

    # 3. 知识图谱检索：intent 节点已抽出实体，命中图谱节点即取关系（尽可能触发）
    kg_context = ""
    entities = state.get("entities") or []
    doc_keys = kg_service.filenames_to_doc_keys(active_files)
    if entities and not kg_service.is_empty():
        try:
            kg_result = kg_service.query_relations(entities, doc_keys=doc_keys)
            if kg_result:
                kg_context = f"【知识图谱结构化信息】\n{kg_result}"
        except Exception as e:
            print(f"知识图谱检索失败: {e}")
            kg_context = ""

    # 4. RRF 融合向量与稀疏两路（图谱是结构化三元组，单独作为一段前置文本，不参与 RRF）
    dense_ranked = [(i, c) for i, (_, c) in enumerate(dense_hits)]
    sparse_ranked = [(i, c) for i, (_, c) in enumerate(sparse_hits)]
    fused_texts = _rrf_fuse([dense_ranked, sparse_ranked])

    vector_context = "\n\n".join(fused_texts)

    # 5. 附「本次参考文档」元信息：生成阶段依据标注无法定位到具体出处时，
    #    至少能写出「依据：<文档名>」，而非笼统的「参考文档」。
    doc_meta = ""
    source_names = active_files if active_files else vector_store.list_files()
    if source_names:
        doc_meta = "【本次参考文档】\n" + "、".join(source_names)

    full_context = "\n\n".join(part for part in [doc_meta, kg_context, vector_context] if part)

    if not full_context:
        return {**state, "retrieval_context": "", "error_msg": "知识库中未找到相关内容"}

    return {**state, "retrieval_context": full_context, "error_msg": ""}
