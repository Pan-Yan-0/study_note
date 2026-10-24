#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SRC="$PWD"
ROOT="${DICO_JOINT_ROOT:-/mnt/workspace/dico_joint}"
VENV="/mnt/workspace/venv_joint"
mkdir -p "$ROOT"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.cloud.aliyuncs.com
export PYTHONUNBUFFERED=1
py() { "$VENV/bin/python" "$@"; }
lock() {
  exec 9>"$ROOT/queue.lock"
  flock -n 9 || { echo "已有流程在运行，请用 status/log 查看，不要重复启动。"; exit 1; }
}
gpu() {
  py -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0), torch.__version__); assert 'A10' in torch.cuda.get_device_name(0), '这个包按A10规划，请核对实例规格'"
}
case "${1:-}" in
 setup)
  lock
  python -c "import sys; assert sys.version_info[:2]==(3,11), '请选Python 3.11镜像，以匹配原实验'"
  python -c "import sys,time,json; from pathlib import Path; p=Path(sys.argv[1])/'session_clock.json'; p.exists() or p.write_text(json.dumps({'start':time.time()}))" "$ROOT"
  [ -x "$VENV/bin/python" ] || python -m venv "$VENV"
  py -m pip install --upgrade pip
  py -m pip install torch==2.6.0 transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 sentencepiece==0.2.2 huggingface_hub==2.2.0 modelscope==1.40.2 numpy==2.4.6 'protobuf>=4.25,<8'
  gpu
  py -c "import shutil; assert shutil.disk_usage('/mnt/workspace').free > 30*2**30, '至少预留30GiB磁盘空间'"
  py joint_queue.py prepare --root "$ROOT"
  py -m pip freeze > "$ROOT/environment.txt"
  echo "准备完成。下一步：bash run_a10.sh smoke"
  ;;
 smoke)
  lock; gpu
  py -u joint_queue.py smoke --root "$ROOT" 2>&1 | tee -a "$ROOT/smoke.log"
  ;;
 run)
  lock; gpu
  py -c "from joint_common import *; import sys; require(read(Path(sys.argv[1])/'SMOKE_OK.json')['pipeline_id']==pipeline_id(),'请先通过SMOKE')" "$ROOT"
  # Child inherits fd9; shell returns while the entire background queue retains the lock.
  nohup "$VENV/bin/python" -u "$SRC/joint_queue.py" run --root "$ROOT" >> "$ROOT/queue.log" 2>&1 &
  echo "$!" > "$ROOT/queue.pid"
  echo "已在后台开始，PID $!。查看：bash run_a10.sh log"
  ;;
 log) tail -n 50 -f "$ROOT/queue.log" ;;
 status)
  if flock -n "$ROOT/queue.lock" -c true; then echo "队列未持有运行锁"; else echo "流程正在运行"; fi
  tail -n 15 "$ROOT/queue.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
  echo "结果与权重下载目录：$ROOT/download/"
  ;;
 check|pack)
  lock
  py -u joint_queue.py "$1" --root "$ROOT"
  ;;
 *) echo "用法：bash run_a10.sh setup|smoke|run|log|status|check|pack"; exit 2 ;;
esac
