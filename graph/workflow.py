from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.intent_agent import intent_agent
from agents.retrieval_agent import retrieval_agent
from agents.memory_agent import memory_agent
from agents.generation_agent import generation_agent
from agents.memory_updater import memory_updater_node
from voice.asr import asr_service
from voice.tts import tts_service


def _asr_node(state: AgentState) -> AgentState:
    """ASR节点：语音转文字。文字输入直接透传。"""
    audio_path = state.get("audio_path", "")
    if not audio_path:
        if state.get("asr_text", "").strip():
            return state
        return {**state, "asr_text": "", "error_msg": "无音频文件"}

    result = asr_service.transcribe(audio_path)
    if result["success"]:
        return {**state, "asr_text": result["text"]}
    return {**state, "asr_text": "", "error_msg": result.get("error", "语音识别失败")}


def _tts_node(state: AgentState) -> AgentState:
    """TTS节点（语音工具，非智能体）：答案转语音。失败静默降级为纯文字。"""
    answer = state.get("generated_answer", "")
    if not answer:
        return {**state, "tts_audio_path": ""}

    # 限制合成长度，避免语音文件过大
    synth_text = answer[:500]

    try:
        audio_path = tts_service.synthesize(synth_text)
    except Exception:
        audio_path = ""

    return {**state, "tts_audio_path": audio_path or ""}


def _route_after_intent(state: AgentState) -> str:
    """intent 之后路由：记忆指令直接进生成（短路），其余走检索主链路。"""
    if state.get("error_msg"):
        return "memory_updater"
    if state.get("memory_command", ""):
        return "generation"
    return "retrieval"


def _route_after_retrieval(state: AgentState) -> str:
    """retrieval 之后路由：非记忆指令统一走记忆融合，再生成。"""
    if state.get("error_msg") and not state.get("memory_command", ""):
        # 检索无结果且非记忆指令：直接结束（error_msg 会作为答案回显）
        return "generation"
    return "memory_agent"


def build_workflow():
    """构建多 Agent 工作流（统一主链路）。

    asr(语音工具) → intent(意图+实体) → retrieval(混合检索) → memory_agent(记忆融合) → generation(按意图模板) → tts(语音工具) → memory_updater
    记忆指令在 intent 处短路，直接进 generation 返回反馈。
    注：asr/tts 为语音工具节点（非智能体）；真正的智能体是 intent/retrieval/memory_agent/generation/memory_updater 五个。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("asr", _asr_node)
    workflow.add_node("intent", intent_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("memory_agent", memory_agent)
    workflow.add_node("generation", generation_agent)
    workflow.add_node("tts", _tts_node)
    workflow.add_node("memory_updater", memory_updater_node)

    workflow.set_entry_point("asr")
    workflow.add_edge("asr", "intent")

    workflow.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"retrieval": "retrieval", "generation": "generation", "memory_updater": "memory_updater"},
    )

    workflow.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {"memory_agent": "memory_agent", "generation": "generation"},
    )

    workflow.add_edge("memory_agent", "generation")

    # 生成后：有错误则结束，否则 TTS（默认开启，失败静默降级）
    workflow.add_conditional_edges(
        "generation",
        lambda s: "memory_updater" if s.get("error_msg") else "tts",
        {"tts": "tts", "memory_updater": "memory_updater"},
    )

    workflow.add_edge("tts", "memory_updater")
    workflow.add_edge("memory_updater", END)

    return workflow.compile()


agent_workflow = build_workflow()
