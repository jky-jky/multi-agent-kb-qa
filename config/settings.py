import os
# ==============================================
# 必须放在最顶部！在导入任何huggingface库之前设置
# ==============================================

# ffmpeg：openai-whisper 语音识别依赖 ffmpeg CLI（whisper/audio.py 用 subprocess 调 "ffmpeg"）。
# conda 环境里其实已装好 ffmpeg（D:\anaconda\envs\agent1\Library\bin\ffmpeg.exe），
# 只是没加入 PATH。这里在导入 whisper 之前自动把它加进 PATH，无需手动配环境变量。
def _ensure_ffmpeg_in_path():
    import sys
    # 用当前解释器的父目录反推 conda env 的 Library\bin（D:\anaconda\envs\agent1\Library\bin）
    _env = os.path.dirname(sys.executable)  # D:\anaconda\envs\agent1
    _ffmpeg_dir = os.path.join(_env, "Library", "bin")
    if os.path.isdir(_ffmpeg_dir):
        cur = os.environ.get("PATH", "")
        if _ffmpeg_dir not in cur:
            os.environ["PATH"] = _ffmpeg_dir + os.pathsep + cur
        return _ffmpeg_dir
    return ""

_ensure_ffmpeg_in_path()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# hf-mirror 不支持 Xet 协议（其CAS服务器需访问 huggingface.com 会 401）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# HF 缓存目录（必须在导入 huggingface 库之前设置）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.environ.setdefault("HF_HOME", os.path.join(DATA_DIR, "hf_cache"))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# 加载环境变量
load_dotenv()

# 数据路径
DOCS_DIR = os.path.join(DATA_DIR, "docs")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
KG_DIR = os.path.join(DATA_DIR, "kg")

# 自动创建目录
for path in [DOCS_DIR, CHROMA_DB_DIR, AUDIO_DIR, KG_DIR]:
    os.makedirs(path, exist_ok=True)

# ========== LLM 配置（兼容OpenAI接口，可切换智谱、通义、Ollama等） ==========
LLM_CONFIG = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "temperature": 0.1,
}

# 初始化全局LLM实例
llm = ChatOpenAI(**LLM_CONFIG)

# 图谱抽取专用 LLM：限制最大输出 token，防止关系抽取无节制输出导致费用失控
# 每块最多 8 实体 + 8 关系，按最长实体/关系名估算，512 token 足够。
kg_llm = ChatOpenAI(
    **{**LLM_CONFIG, "max_tokens": 512, "temperature": 0.0},
)

# ========== 嵌入模型配置（自动检测GPU/CPU） ==========
# 使用本地下载好的模型目录，避免每次启动都联网检查/下载
EMBEDDING_MODEL_NAME = os.path.join(DATA_DIR, "models", "bge-m3")
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# ========== 业务阈值配置 ==========
RETRIEVAL_TOP_K = 3             # 单通道检索返回候选片段数
# 向量检索相关性阈值：bge-m3 归一化 L2 距离，真实相关片段多在 0.6~1.0 之间。
# 0.5 会把大量真实相关内容误杀（表现为"同一个意思换个问法就查不到"），
# 实测放宽到 0.8 后召回明显稳定，仍能挡住距离 >0.8 的明显无关片段。
RELEVANCE_THRESHOLD = 0.8
SPARSE_TOP_K = 8                # 稀疏检索候选数（召回更宽，交给 RRF 再收敛）
RRF_K = 60                      # RRF 融合常数（1/(60+rank)）

# ========== 知识图谱抽取配置 ==========
# 并发抽取数：单次模型调用约 14 秒，串行处理长文档极慢。
# 注意：并发过高容易触发 API 限流(429)，而限流+重试会成倍烧钱。
# 设为 4 是成本与速度的平衡点。
KG_EXTRACT_CONCURRENCY = 4
KG_SAVE_EVERY = 10              # 每处理多少块落盘一次（中断不丢进度）
KG_EXTRACT_MAX_RETRY = 1        # 单块抽取失败最多重试次数（1=只试一次，避免重复计费）
KG_MAX_ENTITIES = 8             # 每块最多抽取实体数（硬上限，防输出Token失控）
KG_MAX_RELATIONS = 8            # 每块最多抽取关系数（硬上限）