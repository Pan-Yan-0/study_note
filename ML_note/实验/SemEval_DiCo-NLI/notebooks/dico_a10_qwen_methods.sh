#!/usr/bin/env bash
# 在阿里云 PAI-DSW 的 A10 上跑第六组（dico_qwen_methods.py：Qwen LoRA SFT 上的 R-Drop / SCL / MoCo）。上传包解压到 /mnt/workspace/ 以后：
#   bash dico_a10_qwen_methods.sh setup    建单独的 venv_qwen、装依赖（不碰第五组用的 venv），把官方仓库 zip 放进工作目录
#   bash dico_a10_qwen_methods.sh smoke    前台依次跑 rdrop、scl、moco 的 SMOKE（每个约 5 分钟，第一次另加下载模型）
#   bash dico_a10_qwen_methods.sh chain    后台：等第五组（mDeBERTa）跑完 → 三个 SMOKE → 五个方法的正式运行 → 打一个总结果包
#   bash dico_a10_qwen_methods.sh log      看后台进度（Ctrl+C 只退出查看）
#   bash dico_a10_qwen_methods.sh status   看进程、显卡、磁盘和已经生成的结果包
# 跑完以后：下载 /mnt/workspace/dico_qwen_methods/ 里的 dico_qwen_methods_all_*.zip（发给 Claude）和
# dico_qwen_methods_weights_*.zip（自己留着），再释放实例（只停止不能止住扣计算时）。
set -euo pipefail
cd /mnt/workspace
W=/mnt/workspace/dico_qwen_methods
SRC=/mnt/workspace/dico_qwen_methods.py
VENV=/mnt/workspace/venv_qwen
SMOKE_METHODS="rdrop scl moco"            # ce、ce_drop 和第四组 A10 版走同一条交叉熵路径，rdrop 的 SMOKE 已经覆盖 dropout
FULL_METHODS="ce ce_drop rdrop scl moco"  # 先跑基线，后面的方法即使出错也不影响对照
export PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.cloud.aliyuncs.com
export DICO_WORK="$W" DICO_PLATFORM="阿里云 PAI-DSW"
mkdir -p "$W"

activate() {
  [ -x "$VENV/bin/python" ] || { echo "还没有 venv_qwen，请先运行：bash dico_a10_qwen_methods.sh setup"; exit 1; }
  set +u; source "$VENV/bin/activate"; set -u
}

need_gpu() {
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "没有可用的 GPU（torch 看不到卡），先停下。请检查实例。"; exit 1; }
}

mdeberta_running() { pgrep -f "dico_mdeberta_methods.py" >/dev/null; }

run_smoke() {  # $1 = 方法；退出码 0 表示 SMOKE 的检查全部通过
  echo "==== $(date '+%m-%d %H:%M') SMOKE：$1"
  DICO_METHOD="$1" DICO_SMOKE=1 python -u "$SRC" 2>&1 | tee -a "$W/smoke_$1.log"
  return "${PIPESTATUS[0]}"
}

pack_all() {  # 每个方法最新的结果包（SMOKE 和正式各一份）合成一个总包，权重包另合成一个
  local stamp; stamp=$(date +%Y%m%d-%H%M)
  ( cd "$W"
    res=(); wts=()
    for m in $FULL_METHODS; do
      for kind in smoke results; do
        f=$(ls -t dico_qwen_${m}_${kind}_*.zip 2>/dev/null | head -n 1 || true); [ -n "$f" ] && res+=("$f")
      done
      f=$(ls -t dico_qwen_${m}_weights_2*.zip 2>/dev/null | head -n 1 || true); [ -n "$f" ] && wts+=("$f")
    done
    logs=(); for f in chain.log smoke_*.log full_*.log; do [ -e "$f" ] && logs+=("$f"); done
    if [ ${#res[@]} -gt 0 ]; then python -m zipfile -c "dico_qwen_methods_all_$stamp.zip" "${res[@]}" "${logs[@]}"; fi
    if [ ${#wts[@]} -gt 0 ]; then python -m zipfile -c "dico_qwen_methods_weights_$stamp.zip" "${wts[@]}"; fi
    echo "总结果包（发给 Claude）：$W/dico_qwen_methods_all_$stamp.zip"
    echo "总权重包（自己留着）：$W/dico_qwen_methods_weights_$stamp.zip" )
}

chain() {  # 由 chain 命令放到后台运行
  activate
  echo "==== $(date '+%m-%d %H:%M') 开始。先等第五组（mDeBERTa）跑完"
  while true; do  # 连续两次（间隔 1 分钟）都看不到第五组的进程才算跑完，避开第一、二阶段之间的空档
    if mdeberta_running; then sleep 60; continue; fi
    sleep 60
    mdeberta_running || break
  done
  echo "==== $(date '+%m-%d %H:%M') 第五组已经不在跑了，开始第六组"
  need_gpu
  for m in $SMOKE_METHODS; do
    if ! run_smoke "$m"; then
      echo "==== $(date '+%m-%d %H:%M') $m 的 SMOKE 没通过，正式运行不开始。把 $W 里的 smoke_$m.log 和 SMOKE 结果包发给 Claude。"
      pack_all; exit 1
    fi
  done
  for m in $FULL_METHODS; do
    echo "==== $(date '+%m-%d %H:%M') 正式运行：$m"
    if DICO_METHOD="$m" DICO_SMOKE=0 python -u "$SRC" >> "$W/full_$m.log" 2>&1; then
      echo "==== $(date '+%m-%d %H:%M') $m 跑完"
    else
      echo "==== $(date '+%m-%d %H:%M') $m 出错停下了（见 full_$m.log），接着跑下一个方法；已完成的种子在结果包里，重跑会接着跑"
    fi
  done
  echo "==== $(date '+%m-%d %H:%M') 全部结束"
  pack_all
}

case "${1:-}" in
  setup)
    if [ ! -x "$VENV/bin/python" ]; then
      python -m venv "$VENV" || { python -m pip install virtualenv && python -m virtualenv "$VENV"; }
    fi
    activate
    pip install -q --upgrade pip
    pip install torch==2.6.0
    pip install transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 "protobuf>=4.25,<8" modelscope
    for f in SemEval-2027-Task-2-DiCo-NLI-main.zip dico_qwen_*_results_*.zip; do
      if [ -e "$f" ] && [ ! -e "$W/$f" ]; then cp "$f" "$W/" && echo "放进工作目录：$f"; fi
    done
    python -c "import torch; print('torch', torch.__version__, '；GPU 可用：', torch.cuda.is_available())"
    df -h /mnt/workspace | tail -n 1 | awk '{print "工作盘剩余：" $4 "（模型约 8 GB，五个方法的权重和结果约 3 GB）"}'
    echo "装好了。第五组还在跑时可以直接：bash dico_a10_qwen_methods.sh chain（它会等第五组跑完再开始）"
    ;;
  smoke)
    activate; need_gpu
    if mdeberta_running; then echo "第五组还在跑，两边抢一张卡会互相拖慢。等它跑完再跑 SMOKE，或者直接用 chain。"; exit 1; fi
    for m in ${2:-$SMOKE_METHODS}; do run_smoke "$m" || { echo "$m 的 SMOKE 没通过，停下。把上面的输出发给 Claude。"; exit 1; }; done
    echo "三个 SMOKE 都通过了。正式运行：bash dico_a10_qwen_methods.sh chain"
    ;;
  chain)
    activate
    if pgrep -f "dico_a10_qwen_methods.sh _chain" >/dev/null; then echo "已经有一条在跑：bash dico_a10_qwen_methods.sh log 看进度"; exit 0; fi
    nohup bash "$0" _chain >> "$W/chain.log" 2>&1 &
    echo "已在后台开始（进程 $!）。它会先等第五组跑完。看进度：bash dico_a10_qwen_methods.sh log"
    ;;
  _chain) chain ;;
  log)
    tail -n 5 "$W/chain.log" 2>/dev/null || true
    cur=$(ls -t "$W"/smoke_*.log "$W"/full_*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$cur" ]; then echo "---- $cur"; tail -n 30 -f "$cur"; else tail -f "$W/chain.log"; fi
    ;;
  status)
    pgrep -af "dico_qwen_methods.py|dico_a10_qwen_methods.sh _chain|dico_mdeberta_methods.py" || echo "没有在跑"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
    df -h /mnt/workspace | tail -n 1
    tail -n 3 "$W/chain.log" 2>/dev/null || true
    ls -lt "$W"/dico_qwen_*.zip 2>/dev/null | head -n 6 || echo "还没有结果包"
    ;;
  *) sed -n '2,9p' "$0" ;;
esac
