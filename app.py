import os
import shutil
import gradio as gr
from config.settings import DOCS_DIR, KG_EXTRACT_CONCURRENCY
from knowledge_base.vector_store import vector_store
from knowledge_base.document_loader import doc_loader
from knowledge_base.knowledge_graph import kg_service
from graph.workflow import agent_workflow
from memory.memory_manager import short_term_memory, long_term_memory
from utils.token_monitor import token_monitor
from voice.asr import asr_service

# 支持的文件格式
SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

def _extract_path(item):
    """兼容不同版本 Gradio 传入的文件对象（str / NamedString / dict / 文件对象）"""
    if isinstance(item, (str, os.PathLike)):
        return str(item)
    if isinstance(item, dict):
        return item.get("path") or item.get("name") or ""
    if hasattr(item, "name"):
        return str(item.name)
    return ""

def upload_file(files):
    """上传文件到本地目录"""
    if not files:
        return "请先选择文件", []

    if not isinstance(files, (list, tuple)):
        files = [files]

    file_paths, failed = [], []
    for file in files:
        src = _extract_path(file)
        if not src or not os.path.isfile(src):
            failed.append(src or "未知文件")
            continue
        filename = os.path.basename(src)
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTS:
            failed.append(f"{filename}（格式不支持）")
            continue
        save_path = os.path.join(DOCS_DIR, filename)
        try:
            shutil.copyfile(src, save_path)
            file_paths.append(save_path)
        except Exception as e:
            failed.append(f"{filename}（{e}）")

    if not file_paths:
        return "上传失败：" + ("；".join(failed) if failed else "未获取到有效文件"), []

    msg = f"已上传 {len(file_paths)} 个文件：\n" + "\n".join(os.path.basename(p) for p in file_paths)
    if failed:
        msg += "\n以下文件未上传：" + "；".join(failed)
    return msg, file_paths

def build_kb(file_paths, build_graph, progress=gr.Progress()):
    """构建知识库：向量库（本地免费，按文件分集合）+ 可选知识图谱（调用大模型，产生费用，按文档隔离）"""
    if not file_paths:
        return "请先上传文档（点击上方上传区域选择文件）"

    # 加载并分块文档（按文件分组，用于统计与知识图谱按文档隔离）
    file_chunks = {}  # 文件名 -> chunks
    detail = []
    for file_path in file_paths:
        chunks = doc_loader.load_and_split(file_path)
        file_chunks[os.path.basename(file_path)] = chunks
        detail.append(f"{os.path.basename(file_path)}: {len(chunks)} 块")

    if not any(file_chunks.values()):
        return (
            "文档解析失败，请检查文件格式（支持 PDF/TXT/MD/DOCX）\n"
            + "\n".join(detail)
        )

    try:
        # 第零步：重复文档检测（整文 MD5 指纹比对已有文档）
        dup_map = vector_store.find_duplicates(file_paths)
        # 第一步：构建向量库（本地 bge-m3 模型，不产生 API 费用；每个文件独立集合）
        progress(0.05, desc="正在向量化文档（本地模型，免费）...")
        vector_success = vector_store.rebuild(file_paths=file_paths)
    except Exception as e:
        return f"知识库构建失败：{type(e).__name__}: {e}"

    if not vector_success:
        return "知识库构建失败（向量库为空）"

    # 第二步：可选构建知识图谱（每块调一次大模型，按量计费；按文档独立建图，互不污染）
    if build_graph:
        from knowledge_base.knowledge_graph import _stable_doc_key
        total_pending = 0
        for fname, chunks in file_chunks.items():
            doc_key = _stable_doc_key(fname)
            total_pending += kg_service.count_pending(doc_key, chunks)

        if total_pending == 0:
            kg_msg = (
                f"\n知识图谱：全部 {sum(len(c) for c in file_chunks.values())} 块此前已抽取过，本次无需重复调用模型（未产生新费用）。\n"
                f"图谱实体数：{kg_service.node_count}，关系数：{kg_service.edge_count}"
            )
        else:
            est = total_pending * 0.004
            mins = total_pending / max(KG_EXTRACT_CONCURRENCY, 1) * 15 / 60
            progress(
                0.1,
                desc=f"知识图谱抽取：{total_pending} 块待处理（约 ¥{est:.2f}，并发 {KG_EXTRACT_CONCURRENCY}，预计 {mins:.0f} 分钟）",
            )

            total_done = 0
            per_doc_stats = []
            for fname, chunks in file_chunks.items():
                doc_key = _stable_doc_key(fname)

                def cb(done, total, _fname=fname):
                    # 预留 10% 给向量化，剩余 90% 给图谱（按文档整体进度折算）
                    nonlocal total_done
                    progress(0.1 + 0.9 * done / max(total, 1), desc=f"知识图谱抽取 [{_fname}] {done}/{total} 块")

                stats = kg_service.build_from_documents(doc_key, chunks, progress_cb=cb)
                per_doc_stats.append((fname, stats))
                if stats.get("fatal"):
                    break

            any_fatal = any(s.get("fatal") for _, s in per_doc_stats)
            if any_fatal:
                fatal_msg = next(s["fatal"] for _, s in per_doc_stats if s.get("fatal"))
                kg_msg = (
                    f"\n⚠️ 知识图谱抽取已中止 —— {fatal_msg}\n"
                    f"本次成功 {sum(s['processed'] for _, s in per_doc_stats)} 块（已保存，充值后重试会自动跳过，不重复收费），"
                    f"失败 {sum(s['failed'] for _, s in per_doc_stats)} 块。\n"
                    f"当前图谱实体数：{kg_service.node_count}，关系数：{kg_service.edge_count}\n"
                    f"提示：向量检索不受影响，不做知识图谱也能正常问答。"
                )
            else:
                kg_msg = (
                    f"\n知识图谱：新增 {sum(s['processed'] for _, s in per_doc_stats)} 块，"
                    f"跳过 {sum(s['skipped'] for _, s in per_doc_stats)} 块（已抽取过，未重复计费），"
                    f"失败 {sum(s['failed'] for _, s in per_doc_stats)} 块\n"
                    f"图谱实体数：{kg_service.node_count}，关系数：{kg_service.edge_count}"
                )
    else:
        kg_msg = (
            f"\n知识图谱未更新（你未勾选）。当前已有实体 {kg_service.node_count} 个，"
            f"向量检索功能不受影响。"
        )

    # 重复检测提示
    dup_msg = ""
    if dup_map:
        dup_msg = "\n⚠️ 重复检测：以下文档与已有文档内容完全相同，可考虑删除重复项：\n" + "\n".join(
            f"  · {k} 与 {v} 内容重复" for k, v in dup_map.items()
        )

    return (
        f"知识库构建成功！\n向量分块数：{vector_store.count()}\n"
        + " | ".join(detail)
        + kg_msg
        + dup_msg
    )

def kb_status():
    """首页展示当前知识库加载状态"""
    if vector_store.is_ready():
        files = vector_store.list_files()
        v = f"✅ 向量库已加载（{vector_store.count()} 个分块，{len(files)} 个文档）"
        if files:
            v += "\n📄 " + "、".join(files)
    else:
        v = "⚠️ 向量库未加载：请先在「知识库管理」上传文档并构建"
    if kg_service.node_count:
        k = f"✅ 知识图谱已加载（实体 {kg_service.node_count} 个，关系 {kg_service.edge_count} 条）"
    else:
        k = "⚪ 知识图谱为空（构建时勾选「同时构建知识图谱」即可生成，也可不建，不影响问答）"
    return f"{v}\n{k}"

def chat_reply(message, history, active_files):
    """聊天式问答：接收新消息和完整历史，返回历史更新和清空的输入框"""
    # message 可能是字符串（文本框）或 dict（多模态，含语音）
    if isinstance(message, dict):
        text = message.get("text", "")
        audio_path = message.get("files", [None])[0] if message.get("files") else None
    else:
        text = message or ""
        audio_path = None

    # history 是 list[dict]，每条 {"role": "user"/"assistant", "content": str 或 list}
    history = history or []
    active_files = active_files or []

    if not vector_store.is_ready():
        history = history + [
            {"role": "user", "content": text or "（语音）"},
            {"role": "assistant", "content": "请先到「知识库管理」上传文档并构建知识库"},
        ]
        return "", history, ""

    # 长期记忆查询：用户询问系统记住了什么偏好时，直接列出槽位，不走检索
    if text.strip() and _is_memory_query(text.strip()):
        slots_text = _render_memory_slots()
        history = history + [
            {"role": "user", "content": text.strip()},
            {"role": "assistant", "content": slots_text},
        ]
        return "", history, "本次：0 tokens｜会话累计：0 tokens"

    if audio_path and text.strip():
        # 带音频 + 带文字：用户在确认前编辑过识别文字，直接以编辑后文字为准（跳过重复 ASR），
        # 但仍保留音频条用于回听。
        initial_state = {"asr_text": text.strip(), "active_kb_files": active_files}
        user_msg = text.strip()
    elif audio_path and not text.strip():
        # 纯语音提问
        initial_state = {"audio_path": audio_path, "active_kb_files": active_files}
        user_msg = "🎤（语音提问）"
    elif text.strip():
        initial_state = {"asr_text": text.strip(), "active_kb_files": active_files}
        user_msg = text.strip()
    else:
        history = history + [
            {"role": "assistant", "content": "请输入问题或录制语音"},
        ]
        return "", history, ""

    try:
        result = agent_workflow.invoke(initial_state)
    except Exception as e:
        answer = f"处理失败：{type(e).__name__}: {e}"
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
        return "", history, ""

    error_msg = result.get("error_msg", "")
    answer = result.get("generated_answer") or ""
    if error_msg and not answer:
        answer = error_msg
    elif not answer and not error_msg:
        answer = "处理失败：未获取到有效答案"

    # 语音提问时，用户气泡直接显示识别文字（而非占位符）
    if audio_path and result.get("user_query"):
        user_msg = result["user_query"]

    # TTS 默认开启：答案自动合成语音（edge-tts 免费），失败时静默降级为纯文字
    tts_path = result.get("tts_audio_path") or None

    # 组装消息：assistant 气泡可内嵌语音条（content 为 list 结构）。
    # 用 gr.Audio 实例（而非 {"type":"file"}）嵌入，才能通过 buttons=[] 从源头隐藏分享/下载按钮。
    assistant_content = answer
    if tts_path:
        assistant_content = [
            {"type": "text", "text": answer},
            gr.Audio(value=tts_path, label=None, show_label=False, buttons=[], interactive=False),
        ]

    # 语音提问时，用户气泡内也嵌入原始语音条（在识别文字下方），方便回听
    if audio_path:
        user_content = [
            {"type": "text", "text": user_msg},
            gr.Audio(value=audio_path, label=None, show_label=False, buttons=[], interactive=False),
        ]
    else:
        user_content = user_msg

    history = history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Token 统计回显
    token_text = _format_token_summary(result.get("token_summary"))

    return "", history, token_text


def _is_memory_query(text: str) -> bool:
    """判断用户是否在询问系统记住了什么偏好"""
    return any(k in text for k in ["你记得我什么", "你记得什么", "记得我什么", "记住我什么", "我的偏好", "我的喜好", "你记得哪些", "记住了什么"])

def _render_memory_slots() -> str:
    """把长期记忆槽位渲染为可读文本"""
    slots = long_term_memory.get_slots()
    if not slots:
        return "我目前还没有记住你的任何偏好。你可以说「记住我主要关注刑法」「以后回答简洁一点」这类指令来告诉我。"
    lines = []
    for slot, val in slots.items():
        if isinstance(val, list):
            lines.append(f"· {slot}：{'、'.join(val)}")
        else:
            lines.append(f"· {slot}：{val}")
    return "我记住的关于你的偏好：\n" + "\n".join(lines)

def _format_token_summary(token_summary) -> str:
    if not token_summary:
        return "本次：0 tokens｜会话累计：0 tokens"
    round_total = token_summary.get("round_total", 0)
    session_total = token_summary.get("session_total", 0)
    details = token_summary.get("node_details", [])
    if details:
        parts = "、".join(f"{d['node']}:{d['total_tokens']}" for d in details)
        return f"本次 {round_total} tokens（{parts}）｜会话累计 {session_total}"
    return f"本次 {round_total} tokens｜会话累计 {session_total}"

def clear_memory():
    """清空所有记忆（含长期偏好槽位）。仅在需要彻底重置时使用。"""
    short_term_memory.clear()
    long_term_memory.clear_all()
    token_monitor.reset_session()
    return "记忆已清空，会话已重置"

# 构建Gradio界面
with gr.Blocks(title="多Agent语音知识库助手") as demo:
    gr.Markdown("## 🎙️ 多 Agent 知识库助手", elem_id="main-title")
    gr.Markdown("支持文档问答 · 语音交互 · 长期记忆", elem_id="sub-title")
    
    with gr.Tab("📚 知识库管理"):
        kb_status_box = gr.Textbox(label="当前知识库状态", interactive=False, lines=3)
        demo.load(kb_status, None, kb_status_box)

        file_output = gr.Textbox(label="上传状态")
        file_paths_state = gr.State([])
        
        upload_btn = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".txt", ".md", ".markdown", ".docx"],
            label="上传 PDF / TXT / MD / DOCX 文档",
        )
        upload_btn.upload(upload_file, upload_btn, [file_output, file_paths_state])

        build_graph_cb = gr.Checkbox(
            value=False,
            label="同时构建知识图谱（调用大模型抽取实体关系，按量计费约 ¥0.004/块；已抽取过的内容会自动跳过，不重复收费）",
            elem_id="toggle-graph",
        )
        build_btn = gr.Button("🔨 构建知识库", variant="primary")
        build_result = gr.Textbox(label="构建结果", lines=6)
        build_btn.click(build_kb, [file_paths_state, build_graph_cb], [build_result])
    
    with gr.Tab("💬 智能问答"):
        # 知识库选择（多选；不选=检索全部文档）
        kb_selector = gr.Dropdown(
            choices=[],
            multiselect=True,
            label="📚 检索范围（可多选，不选则检索全部文档）",
            interactive=True,
            elem_id="kb-selector",
        )

        # 聊天记录区（主体，占最大空间）
        # 用固定 height 保证聊天区不因消息多少而伸缩
        chatbot = gr.Chatbot(
            label="对话",
            height=520,
            placeholder="知识库助手已就绪，开始提问吧",
            show_label=False,
            elem_id="chatbot-box",
            buttons=[],
            allow_file_downloads=False,
        )

        # 底部输入区
        with gr.Row(elem_id="input-row"):
            msg = gr.Textbox(
                label="",
                placeholder="在这里输入问题，回车发送…",
                scale=9,
                lines=1,
                show_label=False,
                elem_id="msg-box",
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)

        # 次要功能收进折叠面板，做成小按钮
        with gr.Accordion("🎤 语音输入", open=False):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="按住录音，或上传音频文件",
                elem_id="audio-input-box",
                buttons=[],
            )
            gr.Markdown(
                "> 麦克风需在系统浏览器（Chrome/Edge）打开 **http://localhost:7860** 才能使用；"
                "Windows 请先开启「设置→隐私→麦克风」权限。远程部署时请改用「上传」音频。\n\n"
                "> **录音结束后**，识别出的文字会自动填入下方输入框，"
                "你可以确认无误后点「发送」，或直接修改；"
                "如果录错了，直接重新录制即可覆盖。"
            )

        with gr.Accordion("⚙️ 更多选项", open=False):
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", size="sm")
            # Token 消耗监控（每轮问答后自动刷新）
            token_box = gr.Textbox(
                label="📊 Token 消耗监控",
                interactive=False,
                lines=1,
                value="本次：0 tokens｜会话累计：0 tokens",
                elem_id="token-box",
            )

        # 提交：回车或点击按钮都触发
        submit_btn.click(
            chat_reply,
            [msg, chatbot, kb_selector],
            [msg, chatbot, token_box],
        )
        msg.submit(
            chat_reply,
            [msg, chatbot, kb_selector],
            [msg, chatbot, token_box],
        )

        # 语音输入：录音/上传结束后，本地识别文字并自动填入输入框；
        # 用户确认后手动点「发送」或回车（录错直接重新录即可覆盖，无需额外按钮）。
        def preview_audio(audio_path):
            """录音或上传结束后：识别文字并回显到输入框（不自动发送）"""
            if not audio_path:
                return gr.update(value=""), gr.update()
            try:
                result = asr_service.transcribe(audio_path)
            except Exception as e:
                return gr.update(value=f"⚠️ 语音识别失败：{type(e).__name__}: {e}"), gr.update(value=None)

            if not isinstance(result, dict):
                return gr.update(value="⚠️ 语音识别异常，请重新录制"), gr.update(value=None)

            if not result.get("success"):
                err = result.get("error", "").strip()
                return gr.update(value=f"⚠️ 语音识别失败：{err}" if err else "⚠️ 未识别到语音内容，请重新录制"), \
                       gr.update(value=None)

            text = (result.get("text") or "").strip()
            if not text:
                return gr.update(value="⚠️ 未识别到有效内容（可能是静音或环境噪声），请重新录制"), \
                       gr.update(value=None)

            # 识别成功：文字回显到输入框；音频组件清空（不保留语音条，纯文字交互）
            return gr.update(value=text), gr.update(value=None)

        audio_input.stop_recording(
            preview_audio,
            [audio_input],
            [msg, audio_input],
        )
        # 上传音频用 upload 事件（而非 change）：避免与 stop_recording 在录音结束时重复触发
        audio_input.upload(
            preview_audio,
            [audio_input],
            [msg, audio_input],
        )

        # 进入问答页时刷新知识库选项
        def refresh_kb_choices():
            files = vector_store.list_files() if vector_store.is_ready() else []
            return gr.update(choices=files)
        demo.load(refresh_kb_choices, None, kb_selector)

        # 清空对话：只清空界面聊天记录和短期记忆（会话上下文），
        # 长期记忆（偏好槽位）跨会话保留，不在此处清除。
        def clear_chat():
            short_term_memory.clear()
            token_monitor.reset_session()
            return [], "本次：0 tokens｜会话累计：0 tokens"
        clear_btn.click(clear_chat, outputs=[chatbot, token_box])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css="""
        .gradio-container { max-width: 960px !important; margin: auto !important; }
        #main-title { text-align: center; margin-bottom: 0.2em; }
        #sub-title { text-align: center; color: #888; margin-top: 0; }
        footer { display: none !important; }

        /* 聊天区固定布局，避免消息增减导致宽度跳动 */
        #chatbot-box { width: 100% !important; min-width: 0 !important; }
        #input-row { width: 100% !important; }
        #input-row > div { flex-shrink: 1 !important; }
        #msg-box textarea { min-height: 44px !important; }

        /* 聊天气泡加宽：让文字充分展开，不因几个字就换行 */
        #chatbot-box .message-item { max-width: 100% !important; min-width: 40% !important; }
        #chatbot-box .bot-message { margin-right: 8px !important; }
        #chatbot-box .user-message { margin-left: 8px !important; padding-right: 16px !important; }
        #chatbot-box .message-text { max-width: none !important; }

        /* 语音录音按钮放大 */
        #audio-input-box { min-height: 120px !important; }
        #audio-input-box button { transform: scale(1.3) !important; margin: 12px !important; }

        /* 气泡内嵌语音条：限制宽度，与文字和谐 */
        #chatbot-box audio { width: 100% !important; max-width: 360px !important; height: 44px !important; display: block; margin-top: 8px; }
        /* 气泡内嵌音频的分享/下载按钮：双保险隐藏（buttons=[] 已从源头隐藏，这里兜底）。
           注意 class 是 icon_button（单数）/ icon-buttons（复数）都可能出现，一并覆盖。 */
        #chatbot-box .icon-buttons, #chatbot-box .icon-button,
        #chatbot-box [class*="share"], #chatbot-box [class*="download"] {
            display: none !important;
        }

        /* Checkbox 美化为滑动开关（off/on 滑块，去掉勾选框） */
        #toggle-graph { display: inline-flex !important; align-items: center; gap: 10px; }
        #toggle-graph label { position: relative; display: inline-flex !important; align-items: center; gap: 10px; cursor: pointer; }
        /* 隐藏原生勾选框，改用自定义滑块 */
        #toggle-graph input[type="checkbox"] {
            -webkit-appearance: none; appearance: none; -moz-appearance: none;
            width: 44px; height: 24px; border-radius: 24px;
            background: #ccc; position: relative; outline: none; cursor: pointer;
            transition: background 0.25s; flex-shrink: 0; border: none;
        }
        #toggle-graph input[type="checkbox"]::before {
            content: ""; position: absolute; top: 2px; left: 2px;
            width: 20px; height: 20px; border-radius: 50%;
            background: #fff; transition: transform 0.25s; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        #toggle-graph input[type="checkbox"]:checked {
            background: #4caf50;
        }
        #toggle-graph input[type="checkbox"]:checked::before {
            transform: translateX(20px);
        }
        """,
    )