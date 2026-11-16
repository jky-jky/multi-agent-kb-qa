import re
import os
import uuid
import tempfile

def clean_text(text: str) -> str:
    """清洗文本，去除多余空格、换行符、特殊字符。
    注意：保留 @ . - _ + % 等符号，避免破坏邮箱、网址、数字小数点等内容。
    """
    text = re.sub(r'\s+', ' ', text)
    # 保留中英文标点、@ 邮箱、. 小数点/网址、- _ 连字符、% 百分号、数字等
    text = re.sub(r'[^\w\s，。？！：；、""''（）@.\-_+%#&/]', '', text)
    return text.strip()

def generate_audio_path() -> str:
    """生成临时音频文件路径（系统临时目录，用完即弃，不污染项目目录）"""
    filename = f"wb_tts_{uuid.uuid4().hex}.mp3"
    return os.path.join(tempfile.gettempdir(), filename)

def is_chinese(text: str) -> bool:
    """简单判断是否为中文文本"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))