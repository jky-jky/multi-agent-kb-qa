from typing import TypedDict


class AgentState(TypedDict):
    # 语音输入
    audio_path: str
    asr_text: str

    # 用户查询
    user_query: str
    # 意图标签：answer / summarize / compare / plan / review / rewrite / extract / memory
    # 只决定下游生成用哪个模板，不影响检索与记忆（二者对所有意图开放）
    intent: str
    # 意图节点顺带抽出的核心实体（供图谱检索复用，避免二次 NER）
    entities: list

    # 检索与上下文
    retrieval_context: str   # 混合检索结果（向量+稀疏+图谱，RRF 融合）
    full_context: str        # 记忆 + 检索融合后的完整上下文
    # 知识库隔离：本次检索限定在哪些文档（文件名列表）；空列表=检索全部
    active_kb_files: list

    # 答案生成
    generated_answer: str

    # 记忆指令短路：识别到"记住xxx"等指令时直接处理，不走检索/生成
    memory_command: str      # 非空表示本次是记忆指令，值为要记住的内容
    memory_feedback: str     # 记忆指令写入后的反馈文本

    # 输出控制（TTS 默认开启，无需开关）
    tts_audio_path: str

    # Token统计
    token_summary: dict

    # 错误信息
    error_msg: str
