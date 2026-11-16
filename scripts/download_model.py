# -*- coding: utf-8 -*-
"""直接下载 bge-m3 模型文件到本地目录（绕开 HF 缓存管理，支持断点续传）"""
import os
import time
import httpx

BASE = "https://hf-mirror.com/BAAI/bge-m3/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models", "bge-m3")

FILES = [
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sparse_linear.pt",
    "colbert_linear.pt",
    "pytorch_model.bin",
]

os.makedirs(os.path.join(OUT, "1_Pooling"), exist_ok=True)


def head_size(client, url):
    r = client.head(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def fetch(fname, max_retries=40):
    url = BASE + fname
    dest = os.path.join(OUT, fname.replace("/", os.sep))
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(30, read=60)) as c:
                total = head_size(c, url)
                if os.path.exists(dest) and os.path.getsize(dest) == total:
                    print(f"[skip] {fname} already complete ({total} bytes)", flush=True)
                    return True
                have = os.path.getsize(dest) if os.path.exists(dest) else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                with c.stream("GET", url, headers=headers, follow_redirects=True) as r:
                    if have and r.status_code == 200:
                        # 服务器不支持续传，从头开始
                        have = 0
                        headers = {}
                        r.close()
                        continue_flag = True
                    else:
                        r.raise_for_status()
                        mode = "ab" if have and r.status_code == 206 else "wb"
                        done = have
                        with open(dest, mode) as f:
                            for chunk in r.iter_bytes(1024 * 1024):
                                f.write(chunk)
                                done += len(chunk)
                        if total and done != total:
                            raise IOError(f"incomplete: {done}/{total}")
                print(f"[done] {fname} ({os.path.getsize(dest)} bytes)", flush=True)
                return True
        except Exception as e:
            print(f"[retry {attempt}] {fname}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(2)
    return False


all_ok = True
for f in FILES:
    if not fetch(f):
        print(f"[FAILED] {f}", flush=True)
        all_ok = False

print("ALL_DONE" if all_ok else "SOME_FAILED", flush=True)
