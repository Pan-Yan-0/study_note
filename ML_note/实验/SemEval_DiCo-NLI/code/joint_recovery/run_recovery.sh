#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/mnt/workspace/dico_joint
PY=/mnt/workspace/venv_joint/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec 9>"$ROOT/queue.lock"
flock -n 9 || { echo '原队列仍在运行，停止补跑。'; exit 1; }
nohup "$PY" -u "$PWD/recover_joint.py" --root "$ROOT" >> "$ROOT/recovery.log" 2>&1 &
echo "补跑已启动，PID $!；查看：tail -n 60 -f $ROOT/recovery.log"
