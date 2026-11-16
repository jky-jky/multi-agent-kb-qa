import tiktoken
from typing import Dict
from config.settings import LLM_CONFIG

class TokenMonitor:
    def __init__(self, model_name: str = None):
        self.model = model_name or LLM_CONFIG["model"]
        try:
            self.encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # 模型不匹配时用通用编码兜底
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # 会话级累计统计
        self.session_total = 0
        self._node_stats = []  # 单轮各节点明细
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数"""
        if not text:
            return 0
        return len(self.encoder.encode(text))
    
    def record_node(self, node_name: str, input_text: str, output_text: str):
        """记录单个节点的token消耗"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        total = input_tokens + output_tokens
        self._node_stats.append({
            "node": node_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total
        })
        return total
    
    def get_round_summary(self) -> Dict:
        """获取单轮汇总，并累加到会话总计"""
        round_total = sum(item["total_tokens"] for item in self._node_stats)
        self.session_total += round_total
        summary = {
            "node_details": self._node_stats.copy(),
            "round_total": round_total,
            "session_total": self.session_total
        }
        # 清空单轮明细，准备下一轮
        self._node_stats.clear()
        return summary
    
    def reset_session(self):
        """重置整个会话的统计"""
        self.session_total = 0
        self._node_stats.clear()

# 全局单例
token_monitor = TokenMonitor()