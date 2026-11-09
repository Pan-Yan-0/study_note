"""Independent audit of Haiku dispatch runs. Does not import haiku_dispatch / process_outputs.

usage: python3 -I audit_haiku.py <subagents dir> <out.json> <stage>=<pkg>:<run>[:<build_out>] ...
"""
import csv, glob, hashlib, json, os, re, sys
from collections import Counter, defaultdict

WRAPPER = ("You are a data writer. Read the file {prompt} and do exactly what it asks.\n"
           "Do not open, list or search any other file, and do not use the web.\n"
           "Write your complete answer (only the JSON the file asks for, nothing else) to {out} with the Write tool.\n"
           "Then reply with the single word: done\n")
WRAPPER_SHA_EXPECTED = "690dd6c47fb4f0aa2c695a13c00a017afd62b8b5813dc47e23f37ac2126519ca"
HAIKU = "claude-haiku-5-5"


def sha_b(b): return hashlib.sha256(b).hexdigest()
def sha_f(p):
    with open(p, "rb") as f: return sha_b(f.read())


def read_transcript(sub, aid):
    meta = json.load(open(f"{sub}/agent-{aid}.meta.json", encoding="utf-8"))
    lines = [json.loads(l) for l in open(f"{sub}/agent-{aid}.jsonl", encoding="utf-8")]
    first = lines[0].get("message", {}).get("content") if lines else None
    models, tools, results, times, attach = Counter(), [], {}, [], Counter()
    texts = []
    for d in lines:
        if d.get("timestamp"): times.append(d["timestamp"])
        if d.get("attachment"): attach[d["attachment"].get("type")] += 1
        m = d.get("message") or {}
        if m.get("role") == "assistant" and m.get("model"): models[m["model"]] += 1
        c = m.get("content")
        if isinstance(c, list):
            for x in c:
                if x.get("type") == "tool_use":
                    tools.append({"id": x.get("id"), "name": x.get("name"), "input": x.get("input") or {}})
                elif x.get("type") == "tool_result":
                    results[x.get("tool_use_id")] = {"is_error": bool(x.get("is_error")), "content": x.get("content")}
                elif x.get("type") == "text" and m.get("role") == "assistant":
                    texts.append(x.get("text", ""))
    return {"meta": meta, "first": first, "models": dict(models), "tools": tools, "results": results,
            "start": min(times) if times else None, "end": max(times) if times else None,
            "attach": dict(attach), "texts": texts}


def audit_stage(stage, pkg, run, out, sub, index):
    P = []  # problems
    batches = {}
    for x in sorted(os.listdir(f"{pkg}/batches")):
        b = json.load(open(f"{pkg}/batches/{x}", encoding="utf-8")); batches[b["batch_id"]] = b
    # prompts on disk
    for bid, b in batches.items():
        p = f"{run}/prompts/{bid}.txt"
        if not os.path.exists(p): P.append(f"{bid}: prompt file missing"); continue
        if sha_f(p) != b["prompt_sha256"]: P.append(f"{bid}: prompt file sha != batch prompt_sha256")
        if sha_b(b["prompt"].encode("utf-8")) != b["prompt_sha256"]: P.append(f"{bid}: batch prompt field sha != prompt_sha256")
    log = [json.loads(l) for l in open(f"{run}/dispatch_log.jsonl", encoding="utf-8")]
    keys = Counter((e["batch_id"], int(e["attempt"])) for e in log)
    for k, n in keys.items():
        if n > 1: P.append(f"{k}: logged {n} times")
    by_b = defaultdict(list)
    for e in log: by_b[e["batch_id"]].append(e)
    for bid in batches:
        if bid not in by_b: P.append(f"{bid}: no log entry")
    for bid in by_b:
        if bid not in batches: P.append(f"{bid}: log entry for batch not in package")
    status = Counter(e["status"] for e in log)
    ctx = Counter()
    attach_types = Counter()
    logged_agents = set()
    rows = []
    for e in log:
        bid, att = e["batch_id"], int(e["attempt"]); tag = f"{bid}#{att}"
        aid = e.get("agent_id"); logged_agents.add(aid)
        r = {"batch_id": bid, "attempt": att, "agent_id": aid, "log_status": e["status"]}
        if not aid or not os.path.exists(f"{sub}/agent-{aid}.jsonl"):
            P.append(f"{tag}: transcript for agent {aid} not found"); rows.append(r); continue
        t = read_transcript(sub, aid)
        prompt_path, out_path = f"{run}/prompts/{bid}.txt", f"{run}/outputs/{bid}/attempt_{att}.txt"
        exp = WRAPPER.format(prompt=prompt_path, out=out_path)
        r["desc_ok"] = t["meta"].get("description") == f"Haiku {bid} attempt {att}"
        r["meta_model"] = t["meta"].get("model"); r["picked"] = t["meta"].get("pickedEffortModel")
        r["finished"] = t["meta"].get("finished")
        raw = open(f"{sub}/agent-{aid}.jsonl", encoding="utf-8").read()
        r["effort"] = ",".join(sorted(set(re.findall(r'"effort":"([a-z]+)"', raw)))) or "none"
        seen, out_tok = set(), 0
        for l in raw.splitlines():
            m = (json.loads(l).get("message") or {})
            if m.get("role") == "assistant" and m.get("id") not in seen:
                seen.add(m.get("id")); out_tok += (m.get("usage") or {}).get("output_tokens", 0)
        r["output_tokens"] = out_tok
        r["wrapper_exact"] = t["first"] == exp
        r["models"] = t["models"]
        tools = [(x["name"], x["input"].get("file_path")) for x in t["tools"]]
        r["tools"] = tools
        reads = [f for n, f in tools if n == "Read"]; writes = [f for n, f in tools if n in ("Write", "Edit", "NotebookEdit")]
        other = [n for n, f in tools if n not in ("Read", "Write", "Edit", "NotebookEdit", "SubagentHandback", "ToolSearch")]
        r["only_allowed_tools"] = all(f == prompt_path for f in reads) and all(f == out_path for f in writes) and not other \
            and not any(n == "Edit" for n, _ in tools)
        r["other_tools"] = other
        for k, v in t["attach"].items(): attach_types[k] += 1
        inst = []
        for l in open(f"{sub}/agent-{aid}.jsonl", encoding="utf-8"):
            d = json.loads(l); a = d.get("attachment") or {}
            if a.get("type") == "instructions": inst += [f["path"] for f in a.get("files", [])]
        ctx[";".join(sorted(set(inst)))] += 1
        if not r["desc_ok"]: P.append(f"{tag}: meta description mismatch ({t['meta'].get('description')})")
        if r["meta_model"] != "haiku": P.append(f"{tag}: meta model {r['meta_model']}")
        if not r["finished"]: P.append(f"{tag}: transcript not finished")
        if not r["wrapper_exact"]: P.append(f"{tag}: first message != fixed wrapper")
        if e["status"] == "infra_error":
            if set(t["models"]) - {"<synthetic>"}: P.append(f"{tag}: infra_error but real model replies {t['models']}")
            if tools: P.append(f"{tag}: infra_error but tools used {tools}")
            if os.path.exists(out_path): P.append(f"{tag}: infra_error but output file exists")
            r["infra_text"] = (t["texts"][0][:100] if t["texts"] else "")
            rows.append(r); continue
        if set(t["models"]) != {HAIKU}: P.append(f"{tag}: assistant models {t['models']}")
        if e["status"] == "ok" and not r["only_allowed_tools"]: P.append(f"{tag}: status ok but tools {tools}")
        if e["status"] == "tool_violation" and r["only_allowed_tools"]: P.append(f"{tag}: logged tool_violation but tools look clean")
        # file on disk == what the subagent wrote
        wr = [x for x in t["tools"] if x["name"] == "Write" and x["input"].get("file_path") == out_path]
        if not os.path.exists(out_path): P.append(f"{tag}: output missing"); rows.append(r); continue
        disk = open(out_path, encoding="utf-8", newline="").read()
        r["n_writes"] = len(wr)
        r["disk_equals_last_write"] = bool(wr) and wr[-1]["input"].get("content") == disk
        if not r["disk_equals_last_write"]: P.append(f"{tag}: disk file != subagent's Write content")
        if len(wr) != 1: P.append(f"{tag}: {len(wr)} Write calls to output")
        res = t["results"].get(wr[-1]["id"]) if wr else None
        if res and res["is_error"]: P.append(f"{tag}: Write tool returned error")
        rd = [x for x in t["tools"] if x["name"] == "Read"]
        if rd:
            rr = t["results"].get(rd[0]["id"])
            if rr is None or rr["is_error"]: P.append(f"{tag}: Read failed or no result")
        else: P.append(f"{tag}: never read the prompt")
        if e.get("raw_sha256") and e["raw_sha256"] != sha_f(out_path): P.append(f"{tag}: log raw_sha256 != disk")
        if e.get("prompt_sha256") != batches[bid]["prompt_sha256"]: P.append(f"{tag}: log prompt_sha256 != batch")
        # times
        if t["start"] and e.get("start") and abs(_ts(t["start"]) - _ts(e["start"])) > 2: P.append(f"{tag}: log start {e['start']} vs transcript {t['start']}")
        if t["end"] and e.get("end") and abs(_ts(t["end"]) - _ts(e["end"])) > 2: P.append(f"{tag}: log end {e['end']} vs transcript {t['end']}")
        rows.append(r)
    # every attempt file on disk is logged
    for f in glob.glob(f"{run}/outputs/*/attempt_*.txt"):
        bid = os.path.basename(os.path.dirname(f)); att = int(re.search(r"attempt_(\d+)", f).group(1))
        if (bid, att) not in keys: P.append(f"{bid}#{att}: output file on disk but not logged")
    # every subagent dispatched for this run is logged
    unlogged = []
    for aid, (desc, first) in index.items():
        m = re.fullmatch(r"Haiku (\S+) attempt (\d+)", desc or "")
        if not m or m.group(1) not in batches: continue
        if not isinstance(first, str) or f"{run}/prompts/{m.group(1)}.txt" not in first: continue
        if aid not in logged_agents: unlogged.append((aid, desc))
    for aid, desc in unlogged: P.append(f"unlogged subagent {aid} ({desc})")
    # which attempt the build used
    chosen = {}
    if out and os.path.exists(f"{out}/manifest.csv"):
        for row in csv.DictReader(open(f"{out}/manifest.csv", encoding="utf-8")):
            chosen.setdefault(row["batch_id"], set()).add(int(row["attempt"]))
        for bid, ents in by_b.items():
            ents = sorted(ents, key=lambda e: int(e["attempt"]))
            violated = any(e["status"] == "tool_violation" for e in ents)
            first_ok = next((int(e["attempt"]) for e in ents if e["status"] == "ok"), None)
            used = chosen.get(bid)
            if violated:
                if used: P.append(f"{bid}: discarded batch but manifest uses attempt {used}")
            elif first_ok is None:
                if used: P.append(f"{bid}: no ok attempt but manifest uses {used}")
            elif used is not None and used != {first_ok}: P.append(f"{bid}: manifest uses {used}, first ok is {first_ok}")
    return {"stage": stage, "batches": len(batches), "log_rows": len(log), "status": dict(status),
            "attempts_per_batch": dict(Counter(len(v) for v in by_b.values())),
            "context_instruction_files": dict(ctx), "attachment_types_seen": dict(attach_types),
            "api_models": dict(Counter(",".join(sorted(r.get("models") or {})) for r in rows)),
            "effort_ok_attempts": dict(Counter(r.get("effort") for r in rows if r["log_status"] == "ok")),
            "effort_by_batch": {r["batch_id"]: r.get("effort") for r in rows if r["log_status"] == "ok"},
            "problems": P, "rows": rows}


def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


if __name__ == "__main__":
    sub, dest = sys.argv[1], sys.argv[2]
    assert sha_b(WRAPPER.encode("utf-8")) == WRAPPER_SHA_EXPECTED, "wrapper template sha differs"
    index = {}
    for mp in glob.glob(f"{sub}/agent-*.meta.json"):
        aid = os.path.basename(mp)[len("agent-"):-len(".meta.json")]
        try:
            desc = json.load(open(mp, encoding="utf-8")).get("description")
            first = json.loads(open(f"{sub}/agent-{aid}.jsonl", encoding="utf-8").readline()).get("message", {}).get("content")
        except Exception:
            desc, first = None, None
        index[aid] = (desc, first)
    res = []
    for a in sys.argv[3:]:
        st, v = a.split("=", 1); parts = v.split(":")
        pkg, run = parts[0], parts[1]; out = parts[2] if len(parts) > 2 else None
        r = audit_stage(st, pkg, run, out, sub, index)
        res.append(r)
        print(f"== {st}: batches {r['batches']}, log rows {r['log_rows']}, status {r['status']}, attempts/batch {r['attempts_per_batch']}")
        print(f"   api models {r['api_models']}; effort of ok attempts {r['effort_ok_attempts']}")
        print(f"   instruction files {r['context_instruction_files']}")
        print(f"   problems ({len(r['problems'])}): {r['problems'][:12]}")
    json.dump(res, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
