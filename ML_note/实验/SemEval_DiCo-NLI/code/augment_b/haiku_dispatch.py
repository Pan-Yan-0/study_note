"""实验 B 的 Haiku 5.5 生成：Claude 用 Agent 工具调度时的准备和登记（FULL_README 第 10 节）。

  python haiku_dispatch.py prep   <批次包> <运行目录>                       写 prompts/<batch>.txt，核对哈希，打印派发说明
  python haiku_dispatch.py wrapper <运行目录> <batch> <attempt>            打印这一次尝试的派发说明（原样作为子 agent 的任务）
  python haiku_dispatch.py record <批次包> <运行目录> <batch> <attempt> <agent_id> <start> <end> [--infra 原因]
                                                                          算回复哈希、跑格式检查、追加 dispatch_log.jsonl

- 提示词不经调度方转抄：prompts/<batch>.txt 就是批次文件里的 prompt 字段，逐字节相同。
- 回复也不经调度方转抄：子 agent 自己把回复写进 outputs/<batch>/attempt_N.txt。
- 日志的 requested_model 是 Agent 工具实际收到的模型参数（haiku），source 写 claude-haiku-5-5。

  python haiku_dispatch.py harvest <批次包> <运行目录> <subagents 目录>        （11-07 起用这个代替 record）
- 从每个 subagent 自己的 transcript（<subagents 目录>/agent-<id>.jsonl 和 .meta.json）读：agent_id、起止时间、
  API 实际返回的 model 字段（api_models）、调用了哪些工具和读写了哪些文件。
- 只有 api_models 恰好是 {claude-haiku-5-5}、只 Read 了本批提示词文件、只 Write 了本批输出文件，才按格式检查记 ok / format_error；
  模型不是 Haiku 记 model_mismatch（协议失败，不重派，停下）；读写了别的文件记 tool_violation（同样停下）。
- 每个 subagent 的初始 context 里有没有仓库的 CLAUDE.md / AGENTS.md（instructions 附件）也照实记下（context_files）。

  python haiku_dispatch.py verify <subagents 目录> <描述前缀> <输出.json>        事后核对已登记的批次（按 meta.json 的描述匹配）
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


HAIKU_API = "claude-haiku-5-5"


def transcript(sub_dir, agent_id):
    """读一个 subagent 的 transcript：起止时间、API 返回的 model、工具调用、context 里带的指令文件。"""
    import glob
    meta = load(f"{sub_dir}/agent-{agent_id}.meta.json")
    models, tools, times, ctx = set(), [], [], []
    with open(f"{sub_dir}/agent-{agent_id}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("timestamp"): times.append(d["timestamp"])
            m = d.get("message") or {}
            if m.get("model"): models.add(m["model"])
            if isinstance(m.get("content"), list):
                for x in m["content"]:
                    if x.get("type") == "tool_use":
                        inp = x.get("input") or {}
                        tools.append({"name": x["name"], "file": inp.get("file_path")})
            a = d.get("attachment") or {}
            if a.get("type") == "instructions": ctx += [f["path"] for f in a.get("files", [])]
    iso = lambda t: t[:19] + "Z"
    return {"agent_id": agent_id, "description": meta.get("description"), "requested_model_meta": meta.get("model"),
            "api_models": sorted(models), "tools": tools, "start": iso(min(times)), "end": iso(max(times)),
            "context_files": sorted(set(ctx)), "finished": meta.get("finished")}


def harvest(pkg, run, sub_dir):
    import glob, re
    logged = set()
    if os.path.exists(f"{run}/dispatch_log.jsonl"):
        for line in open(f"{run}/dispatch_log.jsonl", encoding="utf-8"):
            e = json.loads(line); logged.add((e["batch_id"], e["attempt"]))
    batches = {x[:-5] for x in os.listdir(f"{pkg}/batches")}
    found = []
    for mp in sorted(glob.glob(f"{sub_dir}/agent-*.meta.json")):
        meta = load(mp); m = re.fullmatch(r"Haiku (\S+) attempt (\d+)", meta.get("description") or "")
        if not m or m.group(1) not in batches: continue
        bid, att = m.group(1), int(m.group(2))
        aid = os.path.basename(mp)[len("agent-"):-len(".meta.json")]
        t = transcript(sub_dir, aid)
        if not t["finished"]: continue  # 还在跑
        first = open(f"{sub_dir}/agent-{aid}.jsonl", encoding="utf-8").readline()
        if f"{run}/prompts/{bid}.txt" not in first: continue  # 别的运行目录的同名批次
        msg = json.loads(first).get("message", {}).get("content")
        t["wrapper_exact"] = msg == wrapper(run, bid, att)  # subagent 收到的任务逐字等于固定派发说明
        found.append((t["start"], bid, att, t))
    for _, bid, att, t in sorted(found):
        if (bid, att) in logged: continue
        b = load(f"{pkg}/batches/{bid}.json")
        with open(f"{run}/prompts/{bid}.txt", "rb") as f: psha = sha(f.read())
        path = f"{run}/outputs/{bid}/attempt_{att}.txt"
        e = {"batch_id": bid, "attempt": att, "agent_id": t["agent_id"], "start": t["start"], "end": t["end"],
             "requested_model": REQUESTED, "source": SOURCE, "visible_model": ",".join(t["api_models"]) or "无",
             "api_models": t["api_models"], "model_verified": t["api_models"] == [HAIKU_API],
             "tools": t["tools"], "context_files": t["context_files"],
             "prompt_sha256": psha, "dispatch_wrapper_sha256": WRAPPER_SHA}
        reads = [x["file"] for x in t["tools"] if x["name"] == "Read"]
        writes = [x["file"] for x in t["tools"] if x["name"] in ("Write", "Edit")]
        other = [x["name"] for x in t["tools"] if x["name"] not in ("Read", "Write", "SubagentHandback", "ToolSearch")]
        e["wrapper_exact"] = t["wrapper_exact"]
        if t["api_models"] in ([], ["<synthetic>"]) and not t["tools"]:
            # API 限额等：subagent 没有拿到任何模型回复，也没调用工具；记 infra_error，可以按协议重派
            e.update(status="infra_error", reason="API 没有返回模型回复（限额），subagent 没有调用工具", has_partial_output=False)
        elif not t["wrapper_exact"]:
            e.update(status="tool_violation", reason="subagent 收到的任务和固定派发说明不一致")
        elif not e["model_verified"]:
            e.update(status="model_mismatch", reason=f"API 返回的 model 是 {t['api_models']}，不是 {HAIKU_API}")
        elif any(r != f"{run}/prompts/{bid}.txt" for r in reads) or any(w != path for w in writes) or other:
            e.update(status="tool_violation", reason=f"读 {reads}、写 {writes}、其他工具 {other}")
        elif not os.path.exists(path):
            e.update(status="infra_error", reason="subagent 结束但没有写出输出文件", has_partial_output=False)
        else:
            with open(path, encoding="utf-8") as f: r = check(b, f.read())
            e.update(status=r["status"], reason="；".join(r["reasons"]), raw_sha256=r["raw_sha256"])
        with open(f"{run}/dispatch_log.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(bid, att, e["status"], e["visible_model"], e.get("reason", "")[:80])


def verify(sub_dir, prefix, dest):
    import glob, re
    rows = []
    for mp in sorted(glob.glob(f"{sub_dir}/agent-*.meta.json")):
        meta = load(mp)
        if not (meta.get("description") or "").startswith(prefix): continue
        aid = os.path.basename(mp)[len("agent-"):-len(".meta.json")]
        t = transcript(sub_dir, aid)
        rows.append({k: t[k] for k in ("agent_id", "description", "requested_model_meta", "api_models", "start", "end", "context_files")}
                    | {"tools": [x["name"] for x in t["tools"]]})
    out = {"n": len(rows), "api_models": dict(__import__("collections").Counter(",".join(r["api_models"]) for r in rows)),
           "all_haiku": all(r["api_models"] == [HAIKU_API] for r in rows),
           "context_files": dict(__import__("collections").Counter(";".join(r["context_files"]) for r in rows)), "agents": rows}
    with open(dest, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=1)
    print({k: v for k, v in out.items() if k != "agents"})


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "prep": prep(os.path.abspath(a[2]), os.path.abspath(a[3]))
    elif a[1] == "wrapper": print(wrapper(os.path.abspath(a[2]), a[3], a[4]), end="")
    elif a[1] == "record":
        infra = a[a.index("--infra") + 1] if "--infra" in a else None
        record(os.path.abspath(a[2]), os.path.abspath(a[3]), *a[4:9], infra=infra)
    elif a[1] == "harvest": harvest(os.path.abspath(a[2]), os.path.abspath(a[3]), os.path.abspath(a[4]))
    elif a[1] == "verify": verify(os.path.abspath(a[2]), a[3], os.path.abspath(a[4]))
    else: raise SystemExit(__doc__)
