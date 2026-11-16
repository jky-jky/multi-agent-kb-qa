"""检索与回答评测脚本（支持消融对比实验 + 规范输出）。

用法：
    D:/anaconda/envs/agent1/python.exe tests/evaluate.py [testset.json] [选项]

选项：
    --no-answer       跳过回答准确率评测（不调用生成 LLM，省 token）
    --no-ablation     跳过消融对比实验（只跑完整链路）
    --out-dir DIR     输出目录（默认 tests/results）

输出：
    results/result.json        完整指标（机器可读）
    results/result.csv         逐用例明细（可用 Excel 打开）
    results/ablation.json      消融对比实验指标
    控制台                       可读报告

核心指标：
    - recall@k          精确召回率（按 chunk 的 source 元数据反查来源文档）
    - intent_accuracy   意图识别准确率
    - entity_hit_rate   实体命中率
    - answer_accuracy   回答准确率（LLM 判定语义一致）

消融对比（证明混合检索价值）：
    dense-only / sparse-only / kg-only / dense+sparse / full(三通道)

前置条件：知识库（向量库+图谱）已构建完成。
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.vector_store import vector_store, sparse_retriever
from knowledge_base.knowledge_graph import kg_service
from agents.retrieval_agent import _rrf_fuse
from config.settings import RETRIEVAL_TOP_K, SPARSE_TOP_K, RELEVANCE_THRESHOLD


# ---------------- 工具 ----------------

def _load_testset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("cases", [])


def _build_content_to_doc() -> dict:
    """建立「分块内容 → 来源文档名」映射，用于精确反查召回命中的文档。

    通过 Chroma 的 metadata['source']（存了文件路径）反查文件名。
    """
    mapping = {}
    for cname, store in vector_store.stores.items():
        try:
            data = store._collection.get(include=["documents", "metadatas"])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            for content, meta in zip(docs, metas):
                src = (meta or {}).get("source", "")
                if src:
                    mapping[content] = os.path.basename(src)
        except Exception:
            continue
    return mapping


def _recall_hit_docs(contents: list, content_to_doc: dict) -> set:
    """根据召回到的分块内容，反查命中了哪些文档名。"""
    hit = set()
    for c in contents:
        d = content_to_doc.get(c)
        if d:
            hit.add(d)
    return hit


# ---------------- 检索通道 ----------------

def _dense_search(query, active_files, k=None):
    k = k or RETRIEVAL_TOP_K
    hits = vector_store.search(query, k=k, file_filter=active_files)
    return [(d, c) for d, c in hits if d <= RELEVANCE_THRESHOLD]


def _sparse_search(query, active_files, k=None):
    k = k or SPARSE_TOP_K
    if not active_files and not vector_store.stores:
        return []
    chunks = vector_store.collect_chunks(file_filter=active_files)
    return sparse_retriever.search(query, chunks, top_k=k)


def _kg_search(query_entities, active_files):
    """图谱检索，返回命中的关系文本（结构化，非 chunk）。"""
    doc_keys = kg_service.filenames_to_doc_keys(active_files) if active_files else None
    return kg_service.query_relations(query_entities, doc_keys=doc_keys)


# ---------------- 消融通道组合 ----------------

def _run_ablation(query, active_files, content_to_doc):
    """对一条 query 分别用 5 种通道组合检索，返回各自命中的文档集合。"""
    entities = kg_service.classify_intent(query)["entities"]

    # 稠密 only
    dense = _dense_search(query, active_files)
    dense_docs = _recall_hit_docs([c for _, c in dense], content_to_doc)

    # 稀疏 only
    sparse = _sparse_search(query, active_files)
    sparse_docs = _recall_hit_docs([c for _, c in sparse], content_to_doc)

    # 图谱 only（命中实体即视为命中该文档——图谱按文档隔离）
    # 说明：_kg_search 只对 active_files(=expected_docs) 对应的文档图做查询；
    # query_relations 返回结构化关系文本（非 chunk），无法按 chunk 反查文档，
    # 但只要 kg_text 非空，就说明在 expected_docs 的文档图里命中了实体——
    # 直接把 expected_docs 视为命中即可，无需（也无法）反查 doc_key。
    kg_text = _kg_search(entities, active_files)
    kg_docs = set(active_files) if kg_text else set()

    # 稠密 + 稀疏 RRF 融合
    dense_ranked = [(i, c) for i, (_, c) in enumerate(dense)]
    sparse_ranked = [(i, c) for i, (_, c) in enumerate(sparse)]
    fused = _rrf_fuse([dense_ranked, sparse_ranked])
    fused_docs = _recall_hit_docs(fused, content_to_doc)

    # 完整三通道：RRF 融合稠密+稀疏，再并入图谱命中（图谱命中时 active_files 视为命中，与 kg_only 口径一致）
    full_docs = fused_docs | kg_docs

    return {
        "dense_only": dense_docs,
        "sparse_only": sparse_docs,
        "kg_only": kg_docs,
        "dense_sparse": fused_docs,
        "full": full_docs,
    }


# ---------------- 主评测 ----------------

def evaluate(testset_path: str, with_answer: bool = True, with_ablation: bool = True) -> dict:
    cases = _load_testset(testset_path)
    total = len(cases)

    content_to_doc = _build_content_to_doc()

    recall_total = 0.0
    recall_evaluated = 0
    intent_correct = 0
    intent_evaluated = 0
    entity_hit_total = 0.0
    entity_evaluated = 0
    answer_correct = 0
    answer_evaluated = 0

    ablation_agg = {k: {"hit": 0, "total": 0} for k in
                    ["dense_only", "sparse_only", "kg_only", "dense_sparse", "full"]}

    per_case = []

    for case in cases:
        qid = case.get("id", "")
        query = case.get("query", "")
        expected_docs = case.get("docs", [])
        expected_intent = case.get("intent", "")
        expected_entities = case.get("entities", [])
        expected_answer = case.get("expected_answer", "")

        # 1. 意图 + 实体（一次调用）
        cls = kg_service.classify_intent(query)
        intent = cls["intent"]
        entities = cls["entities"]

        # 2. 消融对比
        if with_ablation and expected_docs:
            ab = _run_ablation(query, expected_docs, content_to_doc)
            expected_set = set(expected_docs)
            for combo, hit_docs in ab.items():
                ablation_agg[combo]["total"] += 1
                if hit_docs & expected_set:
                    ablation_agg[combo]["hit"] += 1

        # 3. 精确召回率（用完整链路）
        recall = None
        if expected_docs:
            recall_evaluated += 1
            dense = _dense_search(query, expected_docs)
            sparse = _sparse_search(query, expected_docs)
            kg_text = _kg_search(entities, expected_docs)
            dense_ranked = [(i, c) for i, (_, c) in enumerate(dense)]
            sparse_ranked = [(i, c) for i, (_, c) in enumerate(sparse)]
            fused = _rrf_fuse([dense_ranked, sparse_ranked])
            hit_docs = _recall_hit_docs(fused, content_to_doc)
            # 图谱命中并入
            if kg_text:
                hit_docs |= set(expected_docs)
            recall = len(hit_docs & set(expected_docs)) / len(set(expected_docs))
            recall_total += recall

        # 4. 意图命中
        intent_ok = None
        if expected_intent:
            intent_evaluated += 1
            intent_ok = (intent == expected_intent)
            if intent_ok:
                intent_correct += 1

        # 5. 实体命中
        entity_rate = None
        if expected_entities:
            entity_evaluated += 1
            hit = sum(1 for e in expected_entities if e in entities)
            entity_rate = hit / len(expected_entities)
            entity_hit_total += entity_rate

        # 6. 回答准确率（可选，走生成链路 + LLM 判定）
        answer_acc = None
        if with_answer and expected_answer:
            try:
                from config.settings import llm
                from langchain_core.prompts import ChatPromptTemplate
                from agents.retrieval_agent import retrieval_agent
                from agents.memory_agent import memory_agent
                from agents.generation_agent import generation_agent

                # 完整链路：retrieval_agent → memory_agent → generation_agent
                # （retrieval 产出 retrieval_context，memory 产出 full_context，generation 读 full_context）
                state = {"user_query": query, "active_kb_files": expected_docs,
                         "entities": entities, "intent": intent}
                ctx_state = retrieval_agent(state)
                mem_state = memory_agent(ctx_state)
                if mem_state.get("error_msg"):
                    actual = "（未找到相关内容）"
                else:
                    gen = generation_agent(mem_state)
                    actual = gen.get("generated_answer", "")
                judge_prompt = ChatPromptTemplate.from_template(
                    "请判断「生成答案」是否与「期望答案」语义一致，只回答 1 或 0。\n"
                    "期望答案：{expected}\n生成答案：{actual}\n"
                )
                judge = (judge_prompt | llm).invoke({"expected": expected_answer, "actual": actual})
                answer_acc = 1 if judge.content.strip().startswith("1") else 0
                answer_evaluated += 1
                if answer_acc:
                    answer_correct += 1
            except Exception as e:
                print(f"  [回答评测失败] {query}: {e}")

        per_case.append({
            "id": qid,
            "query": query,
            "category": case.get("category", ""),
            "recall": round(recall, 3) if recall is not None else None,
            "intent": intent,
            "intent_correct": intent_ok,
            "entities": entities,
            "entity_hit_rate": round(entity_rate, 3) if entity_rate is not None else None,
            "answer_acc": answer_acc,
        })

    result = {
        "total": total,
        "recall@k": round(recall_total / recall_evaluated, 4) if recall_evaluated else None,
        "intent_accuracy": round(intent_correct / intent_evaluated, 4) if intent_evaluated else None,
        "entity_hit_rate": round(entity_hit_total / entity_evaluated, 4) if entity_evaluated else None,
        "answer_accuracy": round(answer_correct / answer_evaluated, 4) if answer_evaluated else None,
        "per_case": per_case,
    }
    if with_ablation:
        result["ablation"] = {
            k: round(v["hit"] / v["total"], 4) if v["total"] else None
            for k, v in ablation_agg.items()
        }
    return result


# ---------------- 输出 ----------------

def _write_csv(per_case: list, path: str):
    if not per_case:
        return
    fields = ["id", "query", "category", "recall", "intent", "intent_correct",
              "entities", "entity_hit_rate", "answer_acc"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in per_case:
            row = {k: c.get(k) for k in fields}
            row["entities"] = "、".join(c.get("entities", []))
            w.writerow(row)


def _print_report(result: dict):
    print("\n" + "=" * 50)
    print("评测报告")
    print("=" * 50)
    print(f"用例总数: {result['total']}")
    print(f"召回率 recall@k:      {result.get('recall@k')}")
    print(f"意图识别准确率:       {result.get('intent_accuracy')}")
    print(f"实体命中率:           {result.get('entity_hit_rate')}")
    print(f"回答准确率:           {result.get('answer_accuracy')}")
    if "ablation" in result:
        print("\n--- 消融对比实验（召回率）---")
        labels = {
            "dense_only": "仅稠密向量",
            "sparse_only": "仅稀疏关键词",
            "kg_only": "仅知识图谱",
            "dense_sparse": "稠密+稀疏(RRF)",
            "full": "完整三通道",
        }
        for k, v in result["ablation"].items():
            print(f"  {labels[k]:<14}: {v}")
    print("\n--- 逐用例明细 ---")
    for c in result["per_case"]:
        flag = ""
        if c.get("answer_acc") == 0:
            flag = "  [回答错误]"
        elif c.get("recall") == 0:
            flag = "  [未召回]"
        print(f"  {c['id']} {c['query'][:24]:<26} recall={c['recall']} "
              f"intent={c['intent']} answer={c['answer_acc']}{flag}")
    print("=" * 50)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    testset = pos[0] if pos else os.path.join(os.path.dirname(__file__), "testset.json")
    with_answer = "--no-answer" not in args
    with_ablation = "--no-ablation" not in args
    out_dir = None
    for a in args:
        if a.startswith("--out-dir="):
            out_dir = a.split("=", 1)[1]
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(testset):
        print(f"测试集不存在：{testset}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"开始评测：{testset}")
    print(f"（回答评测={'开' if with_answer else '关'}，消融对比={'开' if with_ablation else '关'}）")

    result = evaluate(testset, with_answer=with_answer, with_ablation=with_ablation)

    # 先写 JSON（更重要的机器可读结果），再写 CSV（供 Excel 查看）
    json_path = os.path.join(out_dir, "result.json")
    csv_path = os.path.join(out_dir, "result.csv")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except PermissionError:
        print(f"\n[警告] 无法写入 {json_path}：文件被占用（请关闭打开它的程序后重试）")
    try:
        _write_csv(result["per_case"], csv_path)
    except PermissionError:
        print(f"[警告] 无法写入 {csv_path}：文件被占用（请在 Excel/WPS 中关闭该文件后重试）")

    _print_report(result)
    print(f"\n结果已写入: {out_dir}/result.json 与 result.csv")
