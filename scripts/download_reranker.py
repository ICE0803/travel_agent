#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 Rerank 模型到 data/models/（与 bge-small-zh-v1.5 保持同样的本地部署方式）

用法：
  venv\\Scripts\\python.exe scripts\\download_reranker.py

网络受限时先设镜像（当前进程有效，注意Windows下 setx 不影响已运行进程）：
  $env:HF_ENDPOINT="https://hf-mirror.com"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from huggingface_hub import snapshot_download

REPO = "BAAI/bge-reranker-base"
TARGET = ROOT / "data" / "models" / "bge-reranker-base"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"下载 {REPO}\n   → {TARGET}")
    print("  提示：网络受限时先执行 $env:HF_ENDPOINT='https://hf-mirror.com'")
    snapshot_download(
        repo_id=REPO,
        local_dir=str(TARGET),
        # 只要 PyTorch 权重和分词器，跳过 onnx/h5/msgpack
        # ⚠️ *.bin 与 *.safetensors 是**同一份权重的两种格式**（这里各约 1.06 GB）。
        #    两个都下会白白多占 1 GB；transformers 优先读 safetensors，所以只留它。
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        ignore_patterns=["*.onnx", "onnx/*", "*.h5", "*.msgpack", "*.ot"],
    )
    total = 0
    for f in sorted(TARGET.rglob("*")):
        if f.is_file():
            mb = f.stat().st_size / 1e6
            total += mb
            print(f"  {f.relative_to(TARGET)}  {mb:.1f} MB")
    print(f"完成，共 {total:.1f} MB")

    # 早期版本会同时下 .bin 和 .safetensors，手动清掉冗余的那份
    redundant = TARGET / "pytorch_model.bin"
    if redundant.exists() and (TARGET / "model.safetensors").exists():
        mb = redundant.stat().st_size / 1e6
        print(f"\n⚠ 检测到冗余权重 {redundant.name}（{mb:.1f} MB），"
              f"与 model.safetensors 内容等价，可安全删除：")
        print(f"    Remove-Item \"{redundant}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())