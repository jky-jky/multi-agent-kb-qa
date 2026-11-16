# 多 Agent 语音知识库问答系统

基于 LangGraph 状态图编排的本地知识库问答系统。上传文档后自动构建向量库（可选构建知识图谱），支持文字与语音提问，答案可标注依据并合成为语音。

## 功能特性

- **多 Agent 编排**：意图识别、检索、记忆融合、生成、记忆更新 5 个智能体，配合语音识别、语音合成 2 个工具节点，流程走向集中在状态图中声明。
- **三通道混合检索**：稠密向量（bge-m3）负责语义匹配，稀疏关键词（bge-m3 lexical weights）负责专有名词、条文编号等精确匹配，两路用 RRF 融合；知识图谱（NetworkX）补充跨段落的实体关系与多跳问题，并提供依据出处。
- **文档级隔离**：每个文档一个独立 Chroma 集合，可按文档勾选检索范围，同名文件覆盖重建。
- **自适应切分**：按文档类型（条文型 / 技术型 / 表格 / 普通）选择切分器与块长。
- **两级记忆**：短期会话（最近 5 轮 + 滚动摘要）与长期偏好（回答详略、结论方式、关注领域、身份称呼）。
- **成本可控**：向量化、语音识别、语音合成均使用本地模型或免费服务；付费环节只有知识图谱构建与问答生成。图谱抽取做了输出上限、分块 md5 增量跳过、并发控制、402/401 熔断。
- **可评测**：`tests/testset.json` 50 条跨文档用例，`tests/evaluate.py` 输出召回率、意图识别、实体命中、回答准确率及五组消融对比。

## 系统流程

```
语音/文字输入
   ↓
asr（语音转文字，文字直接透传）
   ↓
intent（意图分类 + 实体抽取，记忆指令在此短路）
   ↓
retrieval（稠密 + 稀疏 + 知识图谱三通道召回，RRF 融合）
   ↓
memory_agent（拼接检索结果、短期对话、长期偏好）
   ↓
generation（按意图模板生成，分级标注依据）
   ↓
tts（合成为语音，失败降级为纯文字）
   ↓
memory_updater（写入短期记忆、统计 token）
```

## 目录结构

```
agents/          智能体：intent / retrieval / memory_agent / generation / memory_updater
graph/           LangGraph 状态图与共享状态定义
knowledge_base/  文档解析、切分、向量库、知识图谱
memory/          短期会话记忆与长期偏好槽位
voice/           Whisper 语音识别、edge-tts 语音合成
config/          全局配置（模型、阈值、并发等）
tests/           测试用例与评测脚本
scripts/         bge-m3 / Whisper 模型下载与格式转换脚本
app.py           Gradio 界面
```

## 快速开始

### 1. 环境

推荐 Python 3.11。建议使用 conda 独立环境：

```bash
conda create -n kbqa python=3.11 -y
conda activate kbqa
pip install -r requirements.txt
```

语音识别依赖 ffmpeg，需自行安装并加入 PATH。Windows 下 conda 环境自带 `Library/bin/ffmpeg.exe`，`config/settings.py` 会自动把它加入 PATH。

### 2. 下载模型

嵌入模型不入库（约 2.1GB），首次运行前先下载：

```bash
python scripts/download_model.py       # bge-m3 到 data/models/bge-m3
python scripts/download_whisper.py     # Whisper base.pt 到用户缓存目录（仅在用语音输入时需要）
```

### 3. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

默认调用 DeepSeek（OpenAI 兼容接口）。如需换成其他厂商，改 `DEEPSEEK_BASE_URL` 与 `DEEPSEEK_MODEL` 即可。

### 4. 启动

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。

### 5. 构建知识库

在界面「知识库管理」中上传文档（支持 PDF / TXT / MD / DOCX），点击构建。构建向量库免费；勾选「同时构建知识图谱」会按分块调用大模型抽取实体关系，产生费用（约 0.004 元/块），已抽过的分块会按 md5 跳过。

## 评测

```bash
python tests/evaluate.py                  # 回答评测 + 消融对比
python tests/evaluate.py --no-ablation    # 只看回答指标
python tests/evaluate.py --no-answer      # 只看检索与意图
```

结果写入 `tests/results/`。50 条用例的实测结果：召回率 100%、意图识别 92%、实体命中 74%、回答准确率 82%；消融中仅稠密 0.8、仅稀疏 1.0、仅图谱 0.56、稠密+稀疏 1.0。

## 说明

- `data/docs/` 仅保留少量公开文档作为示例，课程作业评分标准因含他人联系方式未上传，因此测试集中与之相关的 5 条用例在克隆后无对应文档。
- 本项目面向单机小规模知识库设计。若扩展到万级文档，检索需引入粗筛分层、向量库换分布式方案、构建过程异步化。
