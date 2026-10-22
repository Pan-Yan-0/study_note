#!/usr/bin/env bash
# 在阿里云 PAI-DSW 的 A10 上跑第五组（dico_mdeberta_methods.py）。上传包解压到 /mnt/workspace/ 以后：
#   bash dico_a10_mdeberta.sh setup    建 venv、装依赖，把官方仓库、老师概率、上一次的结果包放进工作目录（几分钟）
#   bash dico_a10_mdeberta.sh smoke    前台跑 SMOKE（约 10 分钟，另加下载模型），最后打印两个阶段的时间估计
#   bash dico_a10_mdeberta.sh stage1   后台跑第一阶段（关掉网页也不停）
#   bash dico_a10_mdeberta.sh stage2   后台跑第二阶段（要先有第一阶段的结果包，由上传包带来）
#   bash dico_a10_mdeberta.sh log      看后台运行的进度（Ctrl+C 只退出查看）
#   bash dico_a10_mdeberta.sh status   看进程、显卡和已经生成的结果包
# 跑完以后：下载 /mnt/workspace/dico_mdeberta/ 里最新的 dico_mdeberta_results_*.zip 发给 Claude，再释放实例（只停止不能止住扣计算时）。
set -euo pipefail
cd /mnt/workspace
W=/mnt/workspace/dico_mdeberta
SRC=/mnt/workspace/dico_mdeberta_methods.py
export PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.cloud.aliyuncs.com
export DICO_WORK="$W" DICO_PLATFORM="阿里云 PAI-DSW"
mkdir -p "$W"

activate() {
  [ -x venv/bin/python ] || { echo "还没有 venv，请先运行：bash dico_a10_mdeberta.sh setup"; exit 1; }
  set +u; source venv/bin/activate; set -u
}

start_bg() {  # $1 = stage1 / stage2
  activate
  if pgrep -f "dico_mdeberta_methods.py" >/dev/null; then echo "已经有一个在跑：bash dico_a10_mdeberta.sh log 看进度"; exit 0; fi
  LOG="$W/$1.log"
  echo "$1" > "$W/current_stage"
  DICO_STAGE="$1" DICO_LOG="$LOG" nohup python -u "$SRC" >> "$LOG" 2>&1 &
  echo "已在后台开始 $1（进程 $!）。看进度：bash dico_a10_mdeberta.sh log"
  echo "跑完会打印'结果包：…'。下载那个 zip 发给 Claude，下载完再释放实例。"
}

case "${1:-}" in
  setup)
    if [ ! -x venv/bin/python ]; then
      python -m venv venv || { python -m pip install -i "$PIP_INDEX_URL" virtualenv && python -m virtualenv venv; }
    fi
    activate
    pip install -q --upgrade pip
    pip install torch==2.6.0
    pip install transformers==5.19.0 sentencepiece "protobuf>=4.25,<8" modelscope numpy
    for f in SemEval-2027-Task-2-DiCo-NLI-main.zip dico_mdeberta_results_*.zip; do
      if [ -e "$f" ] && [ ! -e "$W/$f" ]; then cp "$f" "$W/" && echo "放进工作目录：$f"; fi
    done
    if [ -d dico_qwen_teacher_results ] && [ ! -d "$W/dico_qwen_teacher_results" ]; then
      cp -r dico_qwen_teacher_results "$W/" && echo "放进工作目录：老师概率 dico_qwen_teacher_results"
    fi
    python -c "import torch; print('torch', torch.__version__, '；GPU 可用：', torch.cuda.is_available(), '；', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    echo "装好了。下一步：bash dico_a10_mdeberta.sh smoke"
    ;;
  smoke)
    activate
    DICO_STAGE=smoke DICO_LOG="$W/smoke.log" python -u "$SRC" 2>&1 | tee -a "$W/smoke.log"
    ;;
  stage1|stage2) start_bg "$1" ;;
  log)
    s=$(cat "$W/current_stage" 2>/dev/null || echo stage1)
    tail -n 40 -f "$W/$s.log"
    ;;
  status)
    pgrep -af "dico_mdeberta_methods.py" || echo "没有在跑"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
    ls -lt "$W"/dico_mdeberta_*.zip 2>/dev/null | head -n 3 || echo "还没有结果包"
    ;;
  *) sed -n '2,9p' "$0" ;;
esac
