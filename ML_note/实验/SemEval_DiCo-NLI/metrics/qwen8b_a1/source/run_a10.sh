#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" == setup ]]; then
  # 全部走阿里云内网镜像：PyPI 上 Linux 的 torch 2.6.0 就是 CUDA 12.4 版（之前 A10 上都这样装）；
  # download.pytorch.org 在境外，从 PAI 下载很慢，会白白消耗计算时。
  MIRROR=(-i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com)
  python3 -m venv .venv
  .venv/bin/python -m pip install "${MIRROR[@]}" torch==2.6.0
  .venv/bin/python -m pip install "${MIRROR[@]}" transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 numpy==2.4.6 safetensors==0.8.0 tokenizers==0.23.3 huggingface_hub==2.2.0 modelscope==1.40.2
  .venv/bin/python - <<'PY'
import platform, torch
print('Python', platform.python_version(), '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')
assert torch.__version__ == '2.6.0+cu124' and torch.cuda.is_available(), '装出来的 torch 不是 2.6.0+cu124，或看不到 GPU；先停下，把这几行发给 Claude'
if platform.python_version() != '3.11.11':
    print('注意：Python 不是 3.11.11（10-28 验收时的版本）。v5 是新版本、会重新验收，可以继续，但结果里会记下这个差别。')
PY
  exit
fi
exec "${DICO_PYTHON:-.venv/bin/python}" a1_queue.py "$@"
