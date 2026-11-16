from graph.state import AgentState
from utils.common import clean_text
from memory.memory_manager import long_term_memory
from knowledge_base.knowledge_graph import kg_service


# 记忆指令强关键词（"记住xxx"这类，用于在意图分类前做精确短路，零 token 成本）。
# 注意：只保留"明确要求记忆/强调"的动词性触发词；时间副词（下次/以后/今后/之后）
# 只表示"将来"，不代表"要记住"，容易把普通提问误判成记忆指令，故已移除。
REMEMBER_KEYWORDS = ["记住", "记下来", "别忘了", "要记得", "帮我记住", "记一下", "记得", "请记"]

# 弱触发词：单独出现不足以下结论，需交给 LLM 精判是否真是记忆指令
AMBIGUOUS_KEYWORDS = ["以后", "今后", "之后", "下次", "往后", "从今往后"]


def _detect_remember(query: str) -> str:
    """检测记忆指令，返回要记住的内容（空串表示非记忆指令）。"""
    for kw in REMEMBER_KEYWORDS:
        if kw in query:
            content = query.split(kw, 1)[1].strip("，。！？：；、 \t")
            if content:
                return content
    return ""


def _is_ambiguous_remember(query: str) -> bool:
    """判断这句话是否"像记忆指令但仅靠时间副词"——需 LLM 精判。"""
    return any(kw in query for kw in AMBIGUOUS_KEYWORDS)


def _llm_judge_remember(query: str) -> str:
    """歧义时用 LLM 精判：这句是否在要求系统记住一个偏好，若是则返回应记住的内容，否则返回空串。"""
    from langchain_core.prompts import ChatPromptTemplate
    from config.settings import llm

    prompt = ChatPromptTemplate.from_template(
        "判断用户这句话是否是在要求你「记住」一个长期的回答偏好（如回答详略、称呼、关注领域等）。\n"
        "若是，请提取出应记住的偏好内容（保留原文措辞）；若不是（只是普通提问或临时要求），请只输出一个字：否。\n"
        "用户的话：{query}\n"
    )
    result = (prompt | llm).invoke({"query": query})
    text = result.content.strip()
    if text and text != "否":
        return text
    return ""


def intent_agent(state: AgentState) -> AgentState:
    """
    输入：asr_text
    输出：user_query, intent, entities, error_msg, memory_command, memory_feedback

    职责：清洗文本 → 记忆指令短路 → 意图分类 + 实体抽取（合并为一次 LLM 调用）。
    """
    asr_text = state.get("asr_text", "").strip()
    if not asr_text:
        return {**state, "error_msg": "未识别到语音内容", "user_query": ""}

    query = clean_text(asr_text)
    if len(query) < 2:
        return {**state, "error_msg": "输入内容过短，请重新提问", "user_query": ""}

    # —— 记忆指令短路：写长期记忆槽位，不走检索/生成 ——
    # 两级识别：强关键词直接短路（零成本）；仅时间副词等歧义句交给 LLM 精判（低频，可接受）
    memory_content = _detect_remember(query)
    if not memory_content and _is_ambiguous_remember(query):
        memory_content = _llm_judge_remember(query)
    if memory_content:
        feedback = long_term_memory.add_memory(memory_content)
        if feedback:
            return {
                **state,
                "user_query": query,
                "intent": "memory",
                "entities": [],
                "memory_command": memory_content,
                "memory_feedback": feedback,
                "error_msg": "",
            }

    # —— 清除偏好短路：识别「取消/清除/清空/重置」类指令，直接执行，不走检索/生成 ——
    clear_feedback = long_term_memory.clear_preference(query)
    if clear_feedback:
        return {
            **state,
            "user_query": query,
            "intent": "memory",
            "entities": [],
            "memory_command": query,
            "memory_feedback": clear_feedback,
            "error_msg": "",
        }

    # —— 意图分类 + 实体抽取：一次 LLM 调用 ——
    result = kg_service.classify_intent(query)
    intent = result["intent"]
    entities = result["entities"]

    return {
        **state,
        "user_query": query,
        "intent": intent,
        "entities": entities,
        "error_msg": "",
    }
