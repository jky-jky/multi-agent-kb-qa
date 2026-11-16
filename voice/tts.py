import edge_tts
import asyncio
import re
from utils.common import generate_audio_path


def _strip_markdown(text: str) -> str:
    """剥离 Markdown 语法，避免 edge-tts 把 **、#、` 等标记字面朗读出来"""
    # 移除图片/链接语法 [文本](url) → 文本
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 加粗 **x** / __x__、斜体 *x* / _x_ → x
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)
    # 行内代码 `x` → x
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # 标题 #、列表 -/*/+、引用 >
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[.、)]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    # 分隔线
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 残留的星号/下划线
    text = text.replace('*', '').replace('_', '')
    return text.strip()


class TTSService:
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
    
    def synthesize(self, text: str) -> str:
        """
        文本转语音，返回音频文件路径
        """
        output_path = generate_audio_path()
        # 合成前剥离 Markdown 标记，避免朗读出星号、井号等
        text = _strip_markdown(text)
        
        # 同步调用异步方法
        async def _synthesize():
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
        
        try:
            asyncio.run(_synthesize())
            return output_path
        except Exception as e:
            print(f"TTS生成失败: {e}")
            return ""

# 全局单例
tts_service = TTSService()