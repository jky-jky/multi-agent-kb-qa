from graph.state import AgentState
from memory.memory_manager import memory_context
from utils.token_monitor import token_monitor


def memory_agent(state: AgentState) -> AgentState:
    """
    融合短期记忆+长期记忆(偏好槽位)+知识库检索结果，生成完整上下文
    输入：user_query, retrieval_context
    输出：full_context

    说明：本节点只做读取与拼接，全程没有大模型调用，因此不产生费用。
    token_monitor 记录的是这份上下文在生成阶段将消耗多少输入 token，属于记账。
    """
    query = state.get("user_query", "")
    kb_context = state.get("retrieval_context", "")

    # 记忆上下文（短期会话 + 长期偏好槽位），与规划节点共用同一实现
    mem_context = memory_context()

    # 按优先级拼接上下文：记忆在前，知识库原文在后（紧邻用户问题）
    context_parts = []
    if mem_context:
        context_parts.append(mem_context)
    if kb_context:
        context_parts.append(f"【知识库内容】\n{kb_context}")

    full_context = "\n\n".join(context_parts)

    token_monitor.record_node("memory_agent", query + kb_context, full_context)

    return {
        **state,
        "full_context": full_context,
        "error_msg": "" if full_context else "未找到相关内容",
    }
