"""组装 A0 的 Kaggle 数据集（代码 + 输入，不含权重）。

  python3 -I build_bundle.py <官方仓库目录> <输出目录>

内容：
  teacher_a0.py、test_a0.py、README.md，q4b/（qwen4b_augb 的 upstream_qwen.py、common.py、worker.py、expected_data.json，逐字节复制）
  official/final_data/（四个赛道 train / dev 的 participant_labeled 和 reference）
  archive/Q0/seed{s}/fold{k}/：Q0 阶段 1 归档的 trained.json、history.json、COMPLETE.json、actual_split.json、
    val_probs/epoch{best}.json；seed 42 另带 cache.jsonl 和 pred_fold0_sym.csv（逐折复现用）
  data/b2_inputs.json：B2 每折实际送训的生成行（haiku_scale/merged 的 fold_selection_scale.csv 里 tier 1、2），
    两个方向去重后的有序输入，和来源文件的哈希
  SHA256SUMS（不含自身）
Q0 的 15 份 adapter 不在这里：Pan-Yan-0 把 Qwen 4B +1 阶段 1 下载的权重包作为另一个 Kaggle 数据集加进来。
"""
import csv, hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SE = HERE.parent.parent
OFF, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
Q4B = SE / "code" / "qwen4b_augb"
S1 = SE / "metrics" / "qwen4b_augb_s1" / "formal" / "Q0"
MERGED = SE / "metrics" / "augment_b" / "haiku_scale" / "merged"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists(): shutil.rmtree(OUT)
(OUT / "q4b").mkdir(parents=True)
for n in ("teacher_a0.py", "test_a0.py", "README.md"):
    shutil.copy2(HERE / n, OUT / n)
for n in ("upstream_qwen.py", "common.py", "worker.py", "expected_data.json"):
    shutil.copy2(Q4B / n, OUT / "q4b" / n)
for t in (1, 2, 3, 4):
    for s in ("train", "dev"):
        for k in ("participant_labeled", "reference"):
            src = OFF / "final_data" / s / f"dico_nli_{s}_track{t}_{k}.csv"
            dst = OUT / "official" / "final_data" / s / src.name
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
for seed in (42, 43, 44):
    for fold in range(5):
        src = S1 / f"seed{seed}" / f"fold{fold}"; dst = OUT / "archive" / "Q0" / f"seed{seed}" / f"fold{fold}"
        best = json.loads((src / "trained.json").read_text())["best_epoch"]
        names = ["trained.json", "history.json", "COMPLETE.json", "actual_split.json", f"val_probs/epoch{best}.json"]
        if seed == 42: names += ["cache.jsonl", "pred_fold0_sym.csv"]
        for n in names:
            (dst / n).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src / n, dst / n)
            files = json.loads((src / "COMPLETE.json").read_text())["files"]
            if n in files: assert sha(src / n) == files[n], (src, n)  # 归档文件本身和完成回执一致
man = {r["sample_id"]: r for r in csv.DictReader(open(MERGED / "manifest.csv", encoding="utf-8"))}
sel = list(csv.DictReader(open(MERGED / "fold_selection_scale.csv", encoding="utf-8")))
inputs, rows = {}, {}
for k in range(5):
    rs = [man[s["sample_id"]] for s in sel if int(s["fold"]) == k and int(s["tier"]) in (1, 2)]
    seen, inp = set(), []
    for r in rs:
        for x in ((r["text1"], r["text2"]), (r["text2"], r["text1"])):
            if x not in seen: seen.add(x); inp.append(list(x))
    inputs[str(k)], rows[str(k)] = inp, len(rs)
(OUT / "data").mkdir()
json.dump({"inputs": inputs, "rows_per_fold": rows, "rule": "fold_selection_scale.csv 的 tier 1、2（即 B2），两个方向，首次出现的顺序去重",
           "sources": {"manifest.csv": sha(MERGED / "manifest.csv"), "fold_selection_scale.csv": sha(MERGED / "fold_selection_scale.csv"),
                       "gate.json": sha(MERGED / "gate.json")}},
          open(OUT / "data" / "b2_inputs.json", "w", encoding="utf-8"), ensure_ascii=False)
files = sorted(p for p in OUT.rglob("*") if p.is_file())
(OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
print("生成行", rows, "| 有序输入", {k: len(v) for k, v in inputs.items()}, "合计", sum(len(v) for v in inputs.values()), "| 文件", len(files))
