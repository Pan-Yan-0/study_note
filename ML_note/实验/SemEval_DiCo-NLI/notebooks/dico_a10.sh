#!/usr/bin/env bash
# 在阿里云 PAI-DSW 上跑 dico_qwen_sft_a10.py 的辅助脚本。放在 /mnt/workspace/，和 .py 放在一起。
#   bash dico_a10.sh setup   建 venv 并装好依赖（只需一次，venv 在 /mnt/workspace/venv，停机后还在）
#   bash dico_a10.sh smoke   前台跑 SMOKE（约 15 分钟，另加下载模型）
#   bash dico_a10.sh run     后台正式运行（关掉网页也不停），输出写到 dico_qwen_sft_a10.log
#   bash dico_a10.sh log     看正式运行的进度（Ctrl+C 只退出查看，不停训练）
#   bash dico_a10.sh status  看后台进程和显卡
set -euo pipefail
cd "${DICO_WS:-/mnt/workspace}"
MIRROR=https://mirrors.aliyun.com/pypi/simple/
SRC=dico_qwen_sft_a10.py
RUN=dico_qwen_sft_a10_run.py   # 正式运行用的副本：只把 SMOKE 改成 False
LOG=dico_qwen_sft_a10.log

# 浏览器下载同名文件时会改名成 "dico_qwen_sft_a10 (1).py"：用最新上传的那份覆盖 $SRC
newest=$(ls -t dico_qwen_sft_a10*.py 2>/dev/null | grep -v "_run\.py$" | head -n 1 || true)
if [ -n "$newest" ] && [ "$newest" != "$SRC" ]; then
  mv -f "$newest" "$SRC"
  echo "已用最新的 $newest 覆盖 $SRC"
fi

activate() {
  if [ ! -x venv/bin/python ]; then echo "还没有 venv，请先运行：bash dico_a10.sh setup"; exit 1; fi
  set +u
  # shellcheck disable=SC1091
  source venv/bin/activate
  set -u
}

case "${1:-}" in
  setup)
    if [ ! -x venv/bin/python ]; then
      python -m venv venv || { python -m pip install -i "$MIRROR" virtualenv && python -m virtualenv venv; }
    fi
    activate
    pip install -i "$MIRROR" --upgrade pip
    pip install -i "$MIRROR" torch==2.6.0
    pip install -i "$MIRROR" transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 "protobuf>=4.25,<8" modelscope
    python -c "import torch; print('torch', torch.__version__, 'GPU 可用:', torch.cuda.is_available())"
    echo "装好了。下一步：bash dico_a10.sh smoke"
    ;;
  smoke)
    activate
    grep -q '^SMOKE = True' "$SRC" || { echo "$SRC 里不是 SMOKE = True"; exit 1; }
    python -u "$SRC"
    ;;
  run)
    activate
    if pgrep -f "python -u $RUN" >/dev/null; then echo "正式运行已经在后台跑了：bash dico_a10.sh log 看进度"; exit 0; fi
    if [ -f "$RUN" ]; then
      echo "沿用已有的 $RUN 接着跑（不重新生成，代码哈希才不变）"
    else
      sed 's/^SMOKE = True/SMOKE = False/' "$SRC" > "$RUN"
      grep -q '^SMOKE = False' "$RUN" || { echo "没能把 SMOKE 改成 False"; rm -f "$RUN"; exit 1; }
    fi
    nohup python -u "$RUN" >> "$LOG" 2>&1 &
    echo "已在后台开始（进程 $!）。看进度：bash dico_a10.sh log。跑完记得手动停止实例。"
    ;;
  log)
    tail -n 50 -f "$LOG"
    ;;
  status)
    pgrep -af "dico_qwen_sft_a10" || echo "没有在跑的 dico_qwen_sft_a10 进程"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
    ;;
  *)
    sed -n '2,7p' "$0"
    ;;
esac
