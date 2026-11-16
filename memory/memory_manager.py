import time
import json
import os
from typing import Dict, Optional
from config.settings import llm, DATA_DIR
from langchain_core.prompts import ChatPromptTemplate


class ShortTermMemory:
    """短期会话记忆：1段历史摘要 + 最近5轮完整对话"""
    def __init__(self, max_full_rounds: int = 5):
        self.max_full_rounds = max_full_rounds
        self.full_history = []  # 完整对话轮次：[{"user": "...", "assistant": "..."}]
        self.history_summary = ""  # 更早对话的浓缩摘要

    def add_round(self, user_query: str, assistant_answer: str):
        """添加一轮对话，超出阈值自动滚动压缩"""
        self.full_history.append({"user": user_query, "assistant": assistant_answer})

        # 超出最大完整轮次，将最早的一轮压缩进摘要
        if len(self.full_history) > self.max_full_rounds:
            oldest_round = self.full_history.pop(0)
            self._merge_into_summary(oldest_round)

    def _merge_into_summary(self, new_round: Dict):
        """将最早的一轮对话合并到历史摘要中"""
        if not self.history_summary:
            prompt = ChatPromptTemplate.from_template("""
            请将以下对话浓缩成核心摘要，保留主题、关键实体和结论，不超过50字。
            用户：{user}
            助手：{assistant}
            摘要：
            """)
        else:
            prompt = ChatPromptTemplate.from_template("""
            将新对话合并到原有摘要中，生成新摘要，保留核心信息，不超过80字。
            原有摘要：{summary}
            新对话：
            用户：{user}
            助手：{assistant}
            新摘要：
            """)

        chain = prompt | llm
        result = chain.invoke({
            "summary": self.history_summary,
            "user": new_round["user"],
            "assistant": new_round["assistant"]
        })
        self.history_summary = result.content.strip()

    def get_context(self) -> str:
        """获取拼接好的完整对话上下文"""
        context_parts = []
        if self.history_summary:
            context_parts.append(f"【历史对话摘要】\n{self.history_summary}")

        if self.full_history:
            history_text = "\n".join([
                f"用户：{round_data['user']}\n助手：{round_data['assistant']}"
                for round_data in self.full_history
            ])
            context_parts.append(f"【最近对话】\n{history_text}")

        return "\n\n".join(context_parts)

    def clear(self):
        """清空短期记忆"""
        self.full_history.clear()
        self.history_summary = ""


# ============ 长期记忆：结构化偏好槽位 ============
# 设计定位：长期记忆不做"语义自动提取"（那需要额外 LLM 调用、且会跑偏成聊天机器人），
# 而是收敛为「记住会改变系统后续行为的少数事实」——通过显式指令 + 关键词归一化写入有限槽位。
# 写入零成本，读取零成本，行为确定。

# 槽位名 -> 配置（enum=枚举值；max_items 用于自由文本槽位）
SLOT_DETAIL = {
    "回答详略": {"enum": ["详细", "简洁"]},
    "结论方式": {"enum": ["先结论后展开", "直接列要点"]},
    "关注领域": {"enum": None, "max_items": 3},  # 自由文本，最多保留 3 个
    "身份称呼": {"enum": None, "max_items": 1},  # 自由文本，单值覆盖
}

# 各槽位的触发关键词（用于判断"用户这句话是不是在设这个槽"）
_SLOT_KEYWORDS = {
    "回答详略": ["详细", "详尽", "具体", "展开说", "简洁", "简短", "简要", "概括", "简略", "精炼", "详略"],
    "结论方式": ["结论", "要点", "重点", "直接", "先给", "优先", "展开", "罗列", "列出", "给出"],
    "身份称呼": ["我是", "我叫", "我的身份", "称呼我", "叫我"],
    "关注领域": ["关注", "主要关注", "感兴趣", "领域", "研究方向", "关心", "侧重"],
}

# 归一化：把"结论方式"映射到标准枚举值
_CONCLUSION_MAPPING = {
    "直接列要点": ["要点", "重点", "列出", "罗列", "直接列", "直接给", "直接说"],
    "先结论后展开": ["先给结论", "先结论", "优先结论", "结论先行", "先给", "先下结论"],
}

# 详略归一化（注意：简洁词典只放"简洁"语义词，否定式如"不用详细"由否定词逻辑统一处理，避免自相矛盾）
_DETAIL_MAPPING = {
    "简洁": ["简洁", "简短", "简要", "简略", "精炼", "概括", "简明", "简要点", "短一点"],
    "详细": ["详细", "详尽", "具体", "展开说", "详细一点", "详细点", "详细描述", "说详细"],
}

# 否定词：句子含这些词时，其后的详略描述要取反（"不用详细" = 简洁）
_NEGATION_WORDS = ["不用", "不要", "别", "无需", "不必", "不想", "别太", "不要太"]


class LongTermMemory:
    """长期记忆：结构化偏好槽位，跨会话持久化。

    与旧版（向量库 + 权重衰减）不同：本版只存有限的、会改变系统行为的偏好槽位，
    写入靠显式指令 + 关键词归一化，读取直接拼进 prompt，全程零额外 LLM 调用。
    """

    SLOTS_FILE = os.path.join(DATA_DIR, "user_prefs.json")

    def __init__(self):
        self.slots: Dict[str, object] = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        try:
            if os.path.exists(self.SLOTS_FILE):
                with open(self.SLOTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.slots = data if isinstance(data, dict) else {}
        except Exception:
            self.slots = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.SLOTS_FILE), exist_ok=True)
            with open(self.SLOTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.slots, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"保存用户偏好失败: {e}")

    # ---------- 归一化与识别 ----------
    def _normalize_slot_value(self, slot: str, raw_text: str) -> Optional[str]:
        """把用户原话归一化到槽位的标准值；无法归一化返回 None（由调用方追问）。"""
        if slot == "回答详略":
            # 强信号：明确要求"只给结论/直接给结论"，直接判简洁（优先级最高）
            if any(k in raw_text for k in ["只给结论", "直接给结论", "直接说结论", "只要结论"]):
                return "简洁"
            # 否定句：出现否定词时，语义取反（"不用详细"=简洁，"别太简洁"=详细）
            negated = any(k in raw_text for k in _NEGATION_WORDS)
            has_brief = any(k in raw_text for k in _DETAIL_MAPPING["简洁"])
            has_detail = any(k in raw_text for k in _DETAIL_MAPPING["详细"])
            if negated:
                if has_detail and not has_brief:
                    return "简洁"
                if has_brief and not has_detail:
                    return "详细"
                return None
            if has_brief and not has_detail:
                return "简洁"
            if has_detail and not has_brief:
                return "详细"
            return None  # 同时出现或无关键词 => 冲突/无法判定
        if slot == "结论方式":
            has_points = any(k in raw_text for k in _CONCLUSION_MAPPING["直接列要点"])
            has_conclusion = any(k in raw_text for k in _CONCLUSION_MAPPING["先结论后展开"])
            if has_points and not has_conclusion:
                return "直接列要点"
            if has_conclusion and not has_points:
                return "先结论后展开"
            return None
        return None

    def _detect_slot(self, text: str) -> Optional[str]:
        """判断用户这句话在设置哪个槽位；无匹配返回 None。"""
        for slot, kws in _SLOT_KEYWORDS.items():
            if any(k in text for k in kws):
                return slot
        return None

    def _detect_clear(self, text: str) -> Optional[str]:
        """检测"取消/清除偏好"类指令，命中则删除对应槽位并返回反馈，否则返回 None。"""
        if not any(k in text for k in ["取消", "清除", "清空", "忘掉", "不用记", "别记", "重置", "恢复默认"]):
            return None
        slot = self._detect_slot(text)
        if not slot:
            # 未指明具体槽位，但明确要清空 => 清空全部偏好
            self.clear_all()
            return "已清空所有偏好设置"
        if slot in self.slots:
            del self.slots[slot]
            self._save()
            return f"已取消「{slot}」的偏好设置"
        return f"「{slot}」本来就没有设置偏好"

    def _extract_free_value(self, text: str, slot: str) -> Optional[str]:
        """从指令中抽取自由文本槽位的值（身份称呼/关注领域）。"""
        if slot == "身份称呼":
            for marker in ["我是", "我叫", "我的身份是", "叫我", "称呼我"]:
                if marker in text:
                    val = text.split(marker, 1)[1].strip("，。！？：；、 \t\n")
                    if val:
                        return val[:20]
            return None
        if slot == "关注领域":
            for marker in ["关注", "主要关注", "感兴趣", "领域是", "研究方向是", "关心", "侧重"]:
                if marker in text:
                    val = text.split(marker, 1)[1].strip("，。！？：；、 \t\n的")
                    if val:
                        return val[:30]
            return None
        return None

    # ---------- 写入 ----------
    def add_memory(self, content: str) -> str:
        """处理一条记忆指令，返回给用户的反馈文本。

        返回：确认/覆盖提示/追问文本；无法识别返回空串（调用方据此决定是否当普通查询）。
        """
        text = content.strip()
        if not text:
            return ""

        # 清除指令：显式要求取消某偏好
        clear_hit = self._detect_clear(text)
        if clear_hit:
            return clear_hit

        slot = self._detect_slot(text)
        if not slot:
            return ""

        # 枚举槽位
        if SLOT_DETAIL[slot]["enum"]:
            new_val = self._normalize_slot_value(slot, text)
            if new_val is None:
                if slot == "结论方式":
                    return "你的意思是要我「先给结论再展开」还是「直接列要点」？"
                if slot == "回答详略":
                    return "你是希望我回答「详细」一点还是「简洁」一点？"
                return ""
            old_val = self.slots.get(slot)
            self.slots[slot] = new_val
            self._save()
            if old_val and old_val != new_val:
                return f"已把「{slot}」从「{old_val}」改为「{new_val}」"
            return f"好的，已记住「{slot}」为「{new_val}」"

        # 自由文本槽位
        free_val = self._extract_free_value(text, slot)
        if not free_val:
            return ""
        if slot == "身份称呼":
            old_val = self.slots.get(slot)
            self.slots[slot] = free_val
            self._save()
            if old_val and old_val != free_val:
                return f"已把「{slot}」从「{old_val}」改为「{free_val}」"
            return f"好的，已记住你的「{slot}」是「{free_val}」"
        else:  # 关注领域：追加，满 3 淘汰最旧
            items = self.slots.get(slot) or []
            if not isinstance(items, list):
                items = [items]
            items = [x for x in items if x != free_val]
            items.append(free_val)
            max_items = SLOT_DETAIL[slot].get("max_items", 3)
            if len(items) > max_items:
                items = items[-max_items:]
            self.slots[slot] = items
            self._save()
            return f"好的，已记住你关注「{free_val}」"

    # ---------- 读取 ----------
    def get_slots(self) -> Dict[str, object]:
        """返回当前所有已设置的槽位（供查询与注入）。"""
        return dict(self.slots)

    def get_context(self) -> str:
        """把槽位渲染为可注入 prompt 的文本；无槽位返回空串。"""
        if not self.slots:
            return ""
        lines = []
        for slot, val in self.slots.items():
            if slot == "关注领域" and isinstance(val, list):
                if val:
                    lines.append(f"用户关注领域：{'、'.join(val)}")
            else:
                lines.append(f"用户偏好·{slot}：{val}")
        if not lines:
            return ""
        return "【用户偏好】\n" + "\n".join(lines)

    def clear_all(self):
        """清空所有长期偏好槽位。"""
        self.slots = {}
        try:
            if os.path.exists(self.SLOTS_FILE):
                os.remove(self.SLOTS_FILE)
        except Exception:
            pass

    def clear_preference(self, text: str) -> str:
        """对外清除入口：识别「取消/清除/清空/重置」类指令并执行，返回反馈文本。

        供 intent_agent 短路调用（清除指令不含「记住」等写入触发词，需单独识别）。
        返回空串表示不是清除指令。
        """
        return self._detect_clear(text.strip()) or ""


# 全局单例
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()


def memory_context() -> str:
    """统一的记忆上下文：短期会话记忆 + 长期偏好槽位。

    memory_agent 节点统一调用本函数，把记忆与知识库内容并成一份材料，
    供下游生成节点使用。所有意图（查询/总结/规划等）都走同一条记忆链路。
    """
    parts = []
    short_ctx = short_term_memory.get_context()
    long_ctx = long_term_memory.get_context()
    if short_ctx:
        parts.append(short_ctx)
    if long_ctx:
        parts.append(long_ctx)
    return "\n\n".join(parts)
