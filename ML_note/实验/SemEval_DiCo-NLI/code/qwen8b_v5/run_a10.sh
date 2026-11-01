#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" == setup ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
  .venv/bin/python -m pip install -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 numpy==2.4.6 safetensors==0.8.0 tokenizers==0.23.3 huggingface_hub==2.2.0 modelscope scikit-learn pandas
  exit
fi
exec "${DICO_PYTHON:-.venv/bin/python}" a1_queue.py "$@"
