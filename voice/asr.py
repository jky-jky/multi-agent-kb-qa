import whisper
import torch

# 繁体转简体（whisper 输出偶尔带繁体，统一转简体）
try:
    from opencc import OpenCC
    _cc_t2s = OpenCC("t2s")
except Exception:
    _cc_t2s = None


def _to_simplified(text: str) -> str:
    """把文本中的繁体字统一转为简体（opencc 不可用时原样返回）"""
    if not text or _cc_t2s is None:
        return text
    try:
        return _cc_t2s.convert(text)
    except Exception:
        return text

class ASRService:
    def __init__(self, model_size: str = "base"):
        """
        model_size: tiny/base/small/medium/large，base兼顾速度和准确率
        模型采用懒加载：首次调用 transcribe 时才加载，避免启动时联网下载阻塞
        """
        self.model_size = model_size
        self.model = None

    def _get_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(self.model_size, device=device)
        return self.model

    def transcribe(self, audio_path: str) -> dict:
        """
        语音转文字
        返回: {"text": str, "confidence": float, "success": bool}
        """
        try:
            model = self._get_model()
            result = model.transcribe(audio_path, language="zh")
            text = _to_simplified(result["text"].strip())

            # 置信度：优先用 avg_logprob（whisper 对识别结果的置信度），
            # 而非 no_speech_prob（那是"检测到无语音"的概率，两者含义相反，
            # 之前的写法把 no_speech_prob 当置信度反推，导致正常语音被误判失败）。
            confidence = 1.0
            if result.get("segments"):
                confs = []
                for seg in result["segments"]:
                    # avg_logprob 通常为负值（如 -0.5），越大越自信
                    lp = seg.get("avg_logprob", 0.0)
                    # 映射到 0~1：exp(lp) 即 token 平均概率
                    confs.append(min(max(2.718 ** lp, 0.0), 1.0))
                confidence = sum(confs) / len(confs) if confs else 1.0

            # 只要识别出了非空文本，就认为成功；置信度仅作参考记录，不再一刀切拦截。
            # 之前的阈值拦截会把"识别成功但置信度略低"的正常语音误判为失败，导致返回"未找到内容"。
            success = len(text) > 0

            return {
                "text": text,
                "confidence": round(confidence, 4),
                "success": success
            }
        except Exception as e:
            return {"text": "", "confidence": 0.0, "success": False, "error": str(e)}

# 全局单例
asr_service = ASRService()
