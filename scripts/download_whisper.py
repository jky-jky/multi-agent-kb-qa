# -*- coding: utf-8 -*-
"""下载 whisper base.pt（139MB）到用户缓存目录，支持断点续传"""
import os
import time
import httpx

URL = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
TOTAL = 139653961  # base.pt 固定大小
DEST = os.path.expanduser(r"~\.cache\whisper\base.pt")
TMP = DEST + ".part"

os.makedirs(os.path.dirname(DEST), exist_ok=True)

if os.path.exists(DEST) and os.path.getsize(DEST) == TOTAL:
    print("[skip] base.pt already complete", flush=True)
else:
    for attempt in range(1, 61):
        try:
            have = os.path.getsize(TMP) if os.path.exists(TMP) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with httpx.Client(timeout=httpx.Timeout(30, read=90)) as c:
                with c.stream("GET", URL, headers=headers, follow_redirects=True) as r:
                    if have and r.status_code == 200:
                        have = 0  # 不支持续传，重头下
                        with open(TMP, "wb") as f:
                            for chunk in r.iter_bytes(1024 * 1024):
                                f.write(chunk)
                    else:
                        r.raise_for_status()
                        mode = "ab" if have and r.status_code == 206 else "wb"
                        with open(TMP, mode) as f:
                            for chunk in r.iter_bytes(1024 * 1024):
                                f.write(chunk)
            size = os.path.getsize(TMP)
            print(f"[attempt {attempt}] got {size}/{TOTAL}", flush=True)
            if size == TOTAL:
                with open(TMP, "rb") as src, open(DEST, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                print("[done] base.pt installed:", DEST, flush=True)
                break
        except Exception as e:
            print(f"[retry {attempt}] {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(2)
    else:
        print("[FAILED]", flush=True)
