"""用评测脚本当前的解析规则，离线重新判分（不重新调用模型）。

每次运行的逐条原始返回存在 jsonl 里。解析规则改了以后，用这个脚本从原始返回
重新解析 -1 / 1，导出与评测脚本相同格式的逐条 CSV、错例 CSV 和汇总 JSON。

    python rescore_offline.py 输出目录 ../metrics/raw/*.jsonl.gz

- 读 .jsonl 或 .jsonl.gz；同一 idx 出现多次时取最后一条，请求失败的条目不计（与断点续跑的规则相同）。
- 旁边有原汇总（<名字>_summary.json）时沿用其中的运行信息，只重算统计。
- 没跑完的运行（到了时间上限）也会导出，汇总里 completed 为 false。
"""
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_SCRIPT = HERE / "sst2_instruction_eval.py"
spec = importlib.util.spec_from_file_location("sst2_instruction_eval", EVAL_SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

NAME = re.compile(r"(ollama|deepseek|openai)_(.+?)_(basic|cot|role_cot)_think-(on|off|default)_"
                  r"(all|n\d+)_([0-9a-f]{12})$")
META_KEYS = ("run", "provider", "model", "base_url", "fingerprint", "data_fingerprint", "target_n",
             "completed", "prompt", "think", "data", "max_tokens", "temperature", "finished_at")


def read_records(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    done = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:
                    done[rec["idx"]] = rec
    return [done[i] for i in sorted(done)]


def stem_of(path: Path) -> str:
    name = path.name
    for suffix in (".gz", ".jsonl"):
        name = name.removesuffix(suffix)
    return name


def rescore(path: Path, out_dir: Path, parser_sha: str) -> dict:
    stem = stem_of(path)
    m = NAME.match(stem)
    if not m:
        raise ValueError(f"文件名不是评测脚本的命名格式：{path.name}")
    provider, model, prompt, think, size, fp = m.groups()
    if provider == "ollama":  # 文件名里把 qwen3.5:4b 的冒号换成了 -
        model = model[::-1].replace("-", ":", 1)[::-1]
    recs = read_records(path)
    changes = Counter()
    for r in recs:
        old = r["pred"]
        r["pred"], r["parse"] = ev.parse_label(r["content"])
        r["correct"] = r["pred"] == r["label"]
        changes[(old, r["pred"])] += old != r["pred"]
    target = 872 if size == "all" else int(size[1:])
    meta = {"run": stem, "provider": provider, "model": model, "prompt": prompt, "think": think,
            "target_n": target, "completed": len(recs) == target,
            "max_tokens": ev.auto_max_tokens(prompt, think), "temperature": ev.auto_temperature(think)}
    original = path.with_name(stem + "_summary.json")
    if original.exists():
        keep = json.loads(original.read_text(encoding="utf-8"))
        meta = {k: keep[k] for k in META_KEYS if k in keep}
    meta["rescored_offline"] = True
    meta["parser_script_sha256"] = parser_sha
    summary = ev.summarize(recs, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.write_csv(out_dir / f"{stem}.csv", recs)
    ev.write_csv(out_dir / f"{stem}_errors.csv", [r for r in recs if not r["correct"]])
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    newly = sum(v for (old, new), v in changes.items() if old is None and new is not None)
    altered = sum(v for (old, new), v in changes.items() if old is not None)
    print(f"{stem}: {len(recs)}/{target} 条，准确率 {summary['accuracy']:.4f}，无效 {summary['invalid']}，"
          f"截断 {summary['truncated']}；新解析出 {newly} 条，原有预测改变 {altered} 条")
    return {"run": stem, "done": len(recs), "target": target, "accuracy": summary["accuracy"],
            "invalid": summary["invalid"], "truncated": summary["truncated"],
            "newly_parsed": newly, "changed_existing": altered}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1])
    parser_sha = hashlib.sha256(EVAL_SCRIPT.read_bytes()).hexdigest()
    rows = [rescore(Path(p), out_dir, parser_sha) for p in sys.argv[2:]]
    with open(out_dir / "rescore_overview.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"解析规则来自 {EVAL_SCRIPT.name}（SHA-256 {parser_sha[:12]}…）")


if __name__ == "__main__":
    main()
