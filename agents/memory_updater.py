from graph.state import AgentState
from memory.memory_manager import short_term_memory
from utils.token_monitor import token_monitor

def memory_updater_node(state: AgentState) -> AgentState:
    """
    工作流末尾节点：更新短期记忆。
    长期记忆（偏好槽位）只在 intent 阶段通过显式指令写入，此处不重复处理。
    """
    query = state.get("user_query", "")
    answer = state.get("generated_answer", "")

    if not query or not answer or state.get("error_msg"):
        return state

    # 更新短期会话记忆（记忆指令也照常记录本轮对话）
    short_term_memory.add_round(query, answer)

    # 生成Token统计摘要
    token_summary = token_monitor.get_round_summary()

    return {**state, "token_summary": token_summary}
