#!/usr/bin/env bash
# 一次 A10 会话的入口（只调用 run_a10.sh，不改训练逻辑）。在上传包解压出来的目录里运行：
#   bash qwen8b_v5/a10_session.sh prepare   # 解压官方数据、装环境（约 5–10 分钟）
#   bash qwen8b_v5/a10_session.sh check     # 后台：下载模型 → 最长输入探测 → SMOKE → 4B 兼容性（约 1–1.5 小时）
#   bash qwen8b_v5/a10_session.sh all       # v5.5：check 之后直接接 formal（formal 入口自己核对两个回执，没通过就不开跑）
#   bash qwen8b_v5/a10_session.sh status    # 只读：看进度和两个回执（用正式入口同一套校验）
#   bash qwen8b_v5/a10_session.sh formal    # 两个回执都通过后：后台跑 15 个正式任务（约 10 小时）
#   bash qwen8b_v5/a10_session.sh pack      # 单独重新打包（每一步结束时都会自动打包）
#   bash qwen8b_v5/a10_session.sh restore   # 换实例：把上次的三个包放进 /mnt/workspace/restore/ 后恢复
#   bash qwen8b_v5/a10_session.sh resume    # 恢复后：重新下载 8B，接着跑剩下的正式任务
set -euo pipefail
W="${DICO_WS:-/mnt/workspace}"   # 只为本地测试可改
CODE="$(cd -- "$(dirname -- "$0")" && pwd)"
UP="$(dirname -- "$CODE")"
ROOT="$W/dico_qwen8b_v5"
ADAPTER="$UP/inputs/adapter4b"
CACHE="$UP/inputs/zero4b_cache.jsonl"
LOG="$ROOT/session.log"
SLOCK="$ROOT/.session.lock"
mkdir -p "$ROOT"

find_data() { if [[ -d "$W/official" ]]; then find "$W/official" -maxdepth 3 -type d -name final_data | head -1 | xargs -r dirname; fi; }
busy() { ! flock -n "$SLOCK" true; }   # 有 check / formal 在跑时为真

# 在后台串行跑若干阶段，整个会话持有 .session.lock；每个阶段记下退出码
run_bg() {
  local label="$1"; shift
  nohup flock -n "$SLOCK" bash -c '
    lab="$1"; shift
    for step in "$@"; do
      echo "== [$lab] 开始：$step  $(date "+%F %T")"
      set +e; eval "$step"; rc=$?; set -e
      echo "== [$lab] 结束：$step  退出码 $rc  $(date "+%F %T")"
      case "$step" in *" download "*|*" formal "*) [[ $rc -eq 0 ]] || { echo "== [$lab] 这一步失败，后面不再继续"; exit $rc; } ;; esac
    done
    echo "== [$lab] 全部结束"
  ' _ "$label" "$@" >> "$LOG" 2>&1 &
}

case "${1:-}" in
  prepare)
    [[ -f "$UP/official.zip" ]] || { echo "找不到 $UP/official.zip：上传包没有解压到位"; exit 1; }
    [[ -f "$ADAPTER/adapter_model.safetensors" && -f "$ADAPTER/adapter_config.json" ]] || { echo "inputs/adapter4b/ 里缺 4B adapter 的两个文件，先停下"; exit 1; }
    [[ -f "$CACHE" ]] || { echo "缺 inputs/zero4b_cache.jsonl"; exit 1; }
    if [[ -z "$(find_data)" ]]; then mkdir -p "$W/official" && unzip -q "$UP/official.zip" -d "$W/official"; fi
    echo "官方数据：$(find_data)"
    bash "$CODE/run_a10.sh" setup
    echo "环境装好了。下一步：bash qwen8b_v5/a10_session.sh check"
    ;;
  check)
    DATA="$(find_data)"; [[ -n "$DATA" ]] || { echo "先运行 prepare"; exit 1; }
    busy && { echo "已经有任务在跑，不要重复启动。看进度：bash qwen8b_v5/a10_session.sh status"; exit 1; }
    R="bash $CODE/run_a10.sh"; A="--root $ROOT --data $DATA"
    # download 失败就停；SMOKE 失败也继续做兼容性检查，一次实例看全两类信息
    run_bg check "$R plan $A" "$R download $A" "$R smoke $A" "$R compatibility $A --adapter $ADAPTER --baseline-cache $CACHE"
    echo "已在后台开始（关掉网页也会继续）。看进度：bash qwen8b_v5/a10_session.sh status"
    ;;
  all)
    DATA="$(find_data)"; [[ -n "$DATA" ]] || { echo "先运行 prepare"; exit 1; }
    busy && { echo "已经有任务在跑，不要重复启动。看进度：bash qwen8b_v5/a10_session.sh status"; exit 1; }
    R="bash $CODE/run_a10.sh"; A="--root $ROOT --data $DATA"
    # 和 check 相同的四步；最后的 formal 入口会先用 smoke_check / compat_check 核对两个回执，任何一个没通过就拒绝启动
    run_bg all "$R plan $A" "$R download $A" "$R smoke $A" "$R compatibility $A --adapter $ADAPTER --baseline-cache $CACHE" "$R formal $A --confirm-formal"
    echo "已在后台开始：验收（约 1 小时）→ 两个回执都通过后自动跑 15 个正式任务。看进度：bash qwen8b_v5/a10_session.sh status"
    ;;
  status)
    tail -n 30 "$LOG" 2>/dev/null || echo "还没有日志"
    echo "--- 回执（只读校验，和正式入口用同一套检查）"
    DATA="$(find_data)"
    if [[ -x "$CODE/.venv/bin/python" && -n "$DATA" ]]; then
      bash "$CODE/run_a10.sh" receipts --root "$ROOT" --data "$DATA" || echo "回执校验没能运行（见上面的报错）"
    else
      echo "环境还没装好，先运行 prepare"
    fi
    busy && echo "后台：有任务正在运行" || echo "后台：没有任务在运行"
    if [[ -f "$ROOT/last_error.json" ]]; then echo "注意：有报错记录 $ROOT/last_error.json（看修改时间，可能是已经修复的旧错误）"; fi
    ;;
  formal)
    DATA="$(find_data)"; [[ -n "$DATA" ]] || { echo "先运行 prepare"; exit 1; }
    busy && { echo "已经有任务在跑，不要重复启动。看进度：bash qwen8b_v5/a10_session.sh status"; exit 1; }
    out="$(bash "$CODE/run_a10.sh" receipts --root "$ROOT" --data "$DATA")"; echo "$out"
    grep -q '"SMOKE": "通过' <<<"$out" && grep -q '"COMPATIBLE": "通过' <<<"$out" || { echo "两个回执没有都通过，先别跑正式任务；把 status 的输出发给 Claude"; exit 1; }
    run_bg formal "bash $CODE/run_a10.sh formal --root $ROOT --data $DATA --confirm-formal"
    echo "15 个正式任务已在后台开始。中途断了，重新运行这条命令会接着跑（已完成的跳过）。"
    ;;
  restore)
    # 换实例接着跑：把上次下载的三个包（results / weights / resume）放进 /mnt/workspace/restore/，先运行 prepare，再运行本命令
    RD="$W/restore"
    for k in results weights resume; do [[ -f "$RD/a1_qwen8b_v5_$k.zip" ]] || { echo "缺 $RD/a1_qwen8b_v5_$k.zip"; exit 1; }; done
    [[ -e "$ROOT/identity.json" ]] && { echo "$ROOT 里已经有运行记录，不覆盖；先确认是不是已经恢复过"; exit 1; }
    busy && { echo "有任务正在运行"; exit 1; }
    "$CODE/.venv/bin/python" - "$CODE" "$ROOT" "$RD" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
code, root, rd = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
sys.path.insert(0, str(code)); from common import pipeline
pid = pipeline(); n = 0
for k in ('results', 'weights', 'resume'):
    z = zipfile.ZipFile(rd / f'a1_qwen8b_v5_{k}.zip'); m = json.loads(z.read('MANIFEST.json'))
    assert m['pipeline'] == pid, f'{k} 包的代码身份 {m["pipeline"][:12]} 和现在的 {pid[:12]} 不同，不能接着跑'
    for rel, h in m['files'].items():
        if rel.startswith('source/'): continue
        data = z.read(rel); assert hashlib.sha256(data).hexdigest() == h, f'{k} 包里 {rel} 的哈希不对'
        dest = (root / rel).resolve(); assert dest.is_relative_to(root), rel
        if dest.exists(): assert dest.read_bytes() == data, f'{rel} 在两个包里内容不同'; continue
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data); n += 1
print(f'已恢复 {n} 个文件到 {root}，三个包的代码身份都是 {pid[:12]}…')
PY
    echo "下一步：bash qwen8b_v5/a10_session.sh resume"
    ;;
  resume)
    DATA="$(find_data)"; [[ -n "$DATA" ]] || { echo "先运行 prepare"; exit 1; }
    busy && { echo "已经有任务在跑，不要重复启动"; exit 1; }
    out="$(bash "$CODE/run_a10.sh" receipts --root "$ROOT" --data "$DATA")"; echo "$out"
    grep -q '"SMOKE": "通过' <<<"$out" && grep -q '"COMPATIBLE": "通过' <<<"$out" || { echo "恢复后的回执没有都通过，先别跑；把这段输出发给 Claude"; exit 1; }
    R="bash $CODE/run_a10.sh"; A="--root $ROOT --data $DATA"
    # 新实例要重新下载 8B（模型和 model.json 不在包里）；运行环境和上次不同时，正式任务会因 runtime.json 不一致而停下
    run_bg resume "$R download $A" "$R formal $A --confirm-formal"
    echo "已在后台开始：先下载 8B，再接着跑剩下的正式任务（已完成的会核对后跳过）。看进度：bash qwen8b_v5/a10_session.sh status"
    ;;
  pack)
    busy && { echo "有任务正在运行，等它结束（它结束时会自动打包）"; exit 1; }
    bash "$CODE/run_a10.sh" pack --root "$ROOT" --data "$(find_data)"
    ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
