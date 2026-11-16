from graph.state import AgentState
from config.settings import llm
from langchain_core.prompts import ChatPromptTemplate
from utils.token_monitor import token_monitor


# 意图 → 输出形态说明。生成时据此选择对应的回答方式。
# 检索与记忆对全部意图开放，意图只决定「下游用什么结构输出」。
_INTENT_GUIDE = {
    "answer": "事实查询/解释：直接简洁作答，分点清晰，附文档依据。",
    "summarize": "总结/提炼：直接提炼核心要点，分点列出，不赘述原文。",
    "compare": "对比/分析：按「相同点」「不同点」分块逐条对比；文档只提到部分对象时如实说明。",
    "plan": "规划/生成：给出「目标→步骤→时间/顺序→要点提示」的可执行计划，每步一句话。",
    "review": "审核/校验：对照文档检查给定内容是否准确，指出不符或缺失之处，给出修正。",
    "rewrite": "改写/润色：在保留原意与文档依据的前提下，按用户要求改写或润色。",
    "extract": "提取/结构化：从材料中提取指定信息，用分点或表格结构化呈现。",
}

GENERATION_PROMPT = ChatPromptTemplate.from_template("""
你是一个严谨的知识库问答助手，只能根据提供的参考文档回答，禁止编造文档中不存在的内容。

本次任务的输出方式要求（请据此组织回答，但**不要**在回答里说出"本任务属于XX类"这类元描述）：
{intent_guide}

通用要求：
- 只依据参考文档，不确定或文档缺失的内容明确说明，不猜测。
- 回答简洁、分点清晰、重点突出。
- 如果参考文档中完全没有相关信息，直接回答"文档中未找到相关内容"。

【关于"依据"的标注规则，必须遵守】：
- 上下文里「知识图谱结构化信息」部分的三元组（如"死刑 只适用于 严重罪行"）只是定位线索，**不能当作依据原样复述**。
- 真正的依据来自「知识库内容」里的原文分块。标注依据时，只写出处（如"刑法第四十八条"），不要贴出整句原文，也不要复述三元组。
- 依据的分级标注，按可靠程度从高到低：
  1. 若原文分块里能看出明确的条文编号（如"第X条""第X章""第X节"），依据就写该编号（如"刑法第四十八条"）。
  2. 若定位不到条文编号，但能从「知识图谱结构化信息」里的出处（如"（出处：第三百八十三条）"）或「本次参考文档」确认来源，依据就写对应文档名（如"依据：GitHub十大开源AI项目文档"），不要笼统写"参考文档"。
  3. 确实无法确定任何来源时，才写"参考文档"，且不要编造编号或文档名。
- 一条依据对应一个结论点；同一结论点不要重复堆多条依据；不要把三元组里的词拼成伪出处。

参考文档：
{context}

用户问题：
{query}

请回答：
""")


def generation_agent(state: AgentState) -> AgentState:
    """
    输入：user_query, intent, full_context
    输出：generated_answer
    """
    query = state.get("user_query", "")
    context = state.get("full_context", "")
    intent = state.get("intent", "answer")

    # 记忆指令短路：识别到"记住xxx"时，直接返回槽位写入反馈，不走检索/生成
    memory_content = state.get("memory_command", "")
    if memory_content:
        feedback = state.get("memory_feedback", "")
        answer = feedback if feedback else f"好的，我已经记住了：「{memory_content}」"
        return {**state, "generated_answer": answer, "error_msg": ""}

    if not context:
        return {**state, "generated_answer": "文档中未找到相关内容"}

    intent_guide = _INTENT_GUIDE.get(intent, _INTENT_GUIDE["answer"])

    try:
        chain = GENERATION_PROMPT | llm
        result = chain.invoke({"query": query, "context": context, "intent_guide": intent_guide})
        answer = result.content.strip()
    except Exception as e:
        message = str(e)
        if "Insufficient Balance" in message or "402" in message:
            error = "⚠️ API 账户余额不足（402），无法生成答案，请先充值"
        elif "401" in message or "Unauthorized" in message:
            error = "⚠️ API Key 无效（401），请检查 .env 配置"
        else:
            error = f"⚠️ 生成答案失败：{message[:200]}"
        return {**state, "generated_answer": "", "error_msg": error}

    token_monitor.record_node(
        "generation_agent",
        GENERATION_PROMPT.format(query=query, context=context, intent_guide=intent_guide),
        answer,
    )

    return {**state, "generated_answer": answer, "error_msg": ""}
