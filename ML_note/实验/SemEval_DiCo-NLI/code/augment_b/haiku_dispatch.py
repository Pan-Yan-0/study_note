"""实验 B 的 Haiku 5.5 生成：Claude 用 Agent 工具调度时的准备和登记（FULL_README 第 10 节）。

  python haiku_dispatch.py prep   <批次包> <运行目录>                       写 prompts/<batch>.txt，核对哈希，打印派发说明
  python haiku_dispatch.py wrapper <运行目录> <batch> <attempt>            打印这一次尝试的派发说明（原样作为子 agent 的任务）
  python haiku_dispatch.py record <批次包> <运行目录> <batch> <attempt> <agent_id> <start> <end> [--infra 原因]
                                                                          算回复哈希、跑格式检查、追加 dispatch_log.jsonl

- 提示词不经调度方转抄：prompts/<batch>.txt 就是批次文件里的 prompt 字段，逐字节相同。
- 回复也不经调度方转抄：子 agent 自己把回复写进 outputs/<batch>/attempt_N.txt。
- 日志的 requested_model 是 Agent 工具实际收到的模型参数（haiku），source 写 claude-haiku-5-5。
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_format import check  # noqa: E402

SOURCE = "claude-haiku-5-5"
REQUESTED = "haiku"
WRAPPER = """You are a data writer. Read the file {prompt} and do exactly what it asks.
Do not open, list or search any other file, and do not use the web.
Write your complete answer (only the JSON the file asks for, nothing else) to {out} with the Write tool.
Then reply with the single word: done
"""
WRAPPER_SHA = hashlib.sha256(WRAPPER.encode("utf-8")).hexdigest()


def sha(b): return hashlib.sha256(b).hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)


def prep(pkg, run):
    os.makedirs(f"{run}/prompts", exist_ok=True)
    for x in sorted(os.listdir(f"{pkg}/batches")):
        b = load(f"{pkg}/batches/{x}")
        path = f"{run}/prompts/{b['batch_id']}.txt"
        with open(path, "w", encoding="utf-8", newline="") as f: f.write(b["prompt"])
        with open(path, "rb") as f: assert sha(f.read()) == b["prompt_sha256"], b["batch_id"]
    print(f"{len(os.listdir(f'{run}/prompts'))} 个提示词文件，哈希都等于批次文件；派发说明 {WRAPPER_SHA[:12]}…")


def wrapper(run, bid, attempt):
    os.makedirs(f"{run}/outputs/{bid}", exist_ok=True)
    return WRAPPER.format(prompt=f"{run}/prompts/{bid}.txt", out=f"{run}/outputs/{bid}/attempt_{attempt}.txt")


def record(pkg, run, bid, attempt, agent_id, start, end, infra=None):
    b = load(f"{pkg}/batches/{bid}.json")
    with open(f"{run}/prompts/{bid}.txt", "rb") as f: psha = sha(f.read())
    path = f"{run}/outputs/{bid}/attempt_{attempt}.txt"
    e = {"batch_id": bid, "attempt": int(attempt), "agent_id": agent_id, "start": start, "end": end,
         "requested_model": REQUESTED, "source": SOURCE, "visible_model": "Agent 工具未单独返回模型身份",
         "prompt_sha256": psha, "dispatch_wrapper_sha256": WRAPPER_SHA}
    if infra:
        e.update(status="infra_error", reason=infra, has_partial_output=os.path.exists(path))
        if os.path.exists(path):
            with open(path, "rb") as f: e["raw_sha256"] = sha(f.read())
    else:
        with open(path, encoding="utf-8") as f: r = check(b, f.read())
        e.update(status=r["status"], reason="；".join(r["reasons"]), raw_sha256=r["raw_sha256"])
    with open(f"{run}/dispatch_log.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(bid, attempt, e["status"], e.get("reason", ""))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "prep": prep(os.path.abspath(a[2]), os.path.abspath(a[3]))
    elif a[1] == "wrapper": print(wrapper(os.path.abspath(a[2]), a[3], a[4]), end="")
    elif a[1] == "record":
        infra = a[a.index("--infra") + 1] if "--infra" in a else None
        record(os.path.abspath(a[2]), os.path.abspath(a[3]), *a[4:9], infra=infra)
    else: raise SystemExit(__doc__)
