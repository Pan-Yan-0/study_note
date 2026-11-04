"""实验 B 生成侧：处理 GPT 交回的生成输出 → 确定性过滤、清单、停止条件统计、审核抽样；以及审核计分。

  python process_outputs.py build <批次包目录> <交回目录> <官方仓库目录> <输出目录> [--data-version 名称]
  python process_outputs.py score <输出目录>/review_key.json P1 <标注A.csv> <标注B.csv>
  python process_outputs.py adjudicate <输出目录>/review_key.json <Claude标注.csv> <裁决表.csv>   # P2：生成待裁决清单
  python process_outputs.py score <输出目录>/review_key.json P2 <Claude标注.csv> <裁决表.csv>

交回目录：outputs/<batch_id>/attempt_<n>.txt（原样全文）和 dispatch_log.jsonl（每次尝试一行）。
过滤（按顺序，命中第一条就去掉；只做确定性检查，不做语义筛选）：
  same_text     两侧规范化后相同
  length        任一侧词数不在 1–20
  dup_parent    和本批某个父样本是同一短语对（任一方向）
  dup_official  和官方训练集或 dev 的某个短语对相同（任一方向）
  dup_generated 和前面已保留的生成短语对相同（任一方向；按批次号、批内顺序）
规范化：转小写、合并空白。方向展开：E → E/E；FORWARD → F（原方向）/ B（反向）；O 只保留生成的方向。
"""
import csv, hashlib, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_format import check  # noqa: E402

FILTER_VERSION = "augb-filter-v1"
REVIEW_SEED, REVIEW_PER_GROUP = 20261104, 20
STOP = {"batch_fail_rate": 0.05, "dup_rate": 0.10, "review_mismatch": 0.30, "probe_gap": 0.15, "length_dev": 0.50}
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SHORT = {"E": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "B": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER"}


def norm(s): return " ".join(s.lower().split())
def words(s): return len(s.split())
def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def read_csv(p): return list(csv.DictReader(open(p, encoding="utf-8")))


def write_csv(p, rows, fields):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def canonical_original(rows):
    """官方训练集原方向，BACKWARD 交换两侧成 FORWARD，得到和生成数据同一形式的 (text1, text2, label)。"""
    out = []
    for r in rows:
        if not r["instance_id"].endswith("__original"): continue
        t1, t2, l = r["text1"], r["text2"], r["label"]
        if l == "BACKWARD_ENTAILMENT": t1, t2, l = t2, t1, "FORWARD_ENTAILMENT"
        out.append((t1, t2, l))
    return out


def features(t1, t2):
    a, b = norm(t1).split(), norm(t2).split()
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    return [len(a), len(b), len(a) - len(b), len(a) / max(len(b), 1), inter / max(len(sa | sb), 1),
            float(sb <= sa), float(sa <= sb), len(t1) - len(t2), float(a[:1] == b[:1]),
            float(any(c.isdigit() for c in t1 + t2)), float(norm(t1).startswith(("a ", "an ", "the ")))]


def probe(pairs, seed=REVIEW_SEED):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return None
    X = [features(t1, t2) for t1, t2, _ in pairs]; y = [l for _, _, l in pairs]
    k = min(5, min(Counter(y).values()))
    if k < 2: return None
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y,
                             cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    return balanced_accuracy_score(y, pred)


def build(pkg, ret, off, out, data_version):
    os.makedirs(out, exist_ok=True)
    pkg_manifest = json.load(open(f"{pkg}/MANIFEST.json", encoding="utf-8"))
    for rel, h in pkg_manifest["files"].items():
        assert sha_file(f"{pkg}/{rel}") == h, f"批次包文件变了：{rel}"
    batches = {b["batch_id"]: b for b in (json.load(open(f"{pkg}/batches/{x}", encoding="utf-8"))
                                          for x in sorted(os.listdir(f"{pkg}/batches")))}
    log = [json.loads(l) for l in open(f"{ret}/dispatch_log.jsonl", encoding="utf-8") if l.strip()]
    problems, attempts_report, chosen = [], [], {}
    by_batch = defaultdict(list)
    for e in log: by_batch[e["batch_id"]].append(e)
    for bid, b in batches.items():
        entries = sorted(by_batch.get(bid, []), key=lambda e: e["attempt"])
        if not entries: problems.append(f"{bid}：没有调度记录"); continue
        if [e["attempt"] for e in entries] != list(range(1, len(entries) + 1)): problems.append(f"{bid}：尝试编号不连续")
        model_tries = 0
        for e in entries:
            if e.get("prompt_sha256") != b["prompt_sha256"]: problems.append(f"{bid} 第 {e['attempt']} 次：提示词哈希不一致")
            path = f"{ret}/outputs/{bid}/attempt_{e['attempt']}.txt"
            if e["status"] == "infra_error":
                attempts_report.append({"batch_id": bid, "attempt": e["attempt"], "status": "infra_error", "reasons": e.get("reason", "")})
                continue
            model_tries += 1
            if not os.path.exists(path): problems.append(f"{bid}：缺 attempt_{e['attempt']}.txt"); continue
            r = check(b, open(path, encoding="utf-8").read())
            if r["status"] != e["status"]: problems.append(f"{bid} 第 {e['attempt']} 次：日志状态 {e['status']}，脚本判定 {r['status']}")
            attempts_report.append({"batch_id": bid, "attempt": e["attempt"], "status": r["status"], "reasons": "；".join(r["reasons"]),
                                    "raw_sha256": r["raw_sha256"], "visible_model": e.get("visible_model", "")})
            if r["status"] == "ok" and bid not in chosen:
                chosen[bid] = (e, r)
        model_entries = [e for e in entries if e["status"] != "infra_error"]
        if len(model_entries) > 2: problems.append(f"{bid}：模型尝试超过 2 次")
        if len(model_entries) > 1 and model_entries[0]["status"] == "ok": problems.append(f"{bid}：第一次已合格却又重试")
    failed = sorted(set(batches) - set(chosen))

    # 过滤和展开
    train = read_csv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    dev = read_csv(f"{off}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    official = {(norm(r["text1"]), norm(r["text2"])) for r in train + dev}
    official |= {(b, a) for a, b in official}
    off_phr = {norm(x) for r in train + dev for x in (r["text1"], r["text2"])}
    kept_keys, kept, decisions, manifest = set(), [], [], []
    reuse = Counter()
    for bid in sorted(chosen):
        e, r = chosen[bid]; b = batches[bid]
        parent = {(norm(p["text1"]), norm(p["text2"])) for p in b["parents"]}
        parent |= {(y, x) for x, y in parent}
        parent_phr = {norm(x) for p in b["parents"] for x in (p["text1"], p["text2"])}
        for i, p in enumerate(r["pairs"]):
            uid = f"{bid}-{i:03d}"
            a, c = norm(p["text1"]), norm(p["text2"])
            reuse["any_parent_phrase"] += a in parent_phr or c in parent_phr
            reuse["any_official_phrase"] += a in off_phr or c in off_phr
            if a == c: why = "same_text"
            elif not (1 <= words(a) <= 20 and 1 <= words(c) <= 20): why = "length"
            elif (a, c) in parent: why = "dup_parent"
            elif (a, c) in official: why = "dup_official"
            elif (a, c) in kept_keys: why = "dup_generated"
            else: why = ""
            decisions.append({"pair_uid": uid, "batch_id": bid, "text1": p["text1"], "text2": p["text2"], "label": p["label"], "removed": why})
            if why: continue
            kept_keys |= {(a, c), (c, a)}
            kept.append((uid, bid, p))
            rows = [("original", p["text1"], p["text2"], p["label"])]
            if p["label"] == "EQUIVALENCE": rows.append(("flipped", p["text2"], p["text1"], "EQUIVALENCE"))
            if p["label"] == "FORWARD_ENTAILMENT": rows.append(("flipped", p["text2"], p["text1"], "BACKWARD_ENTAILMENT"))
            for d, t1, t2, l in rows:
                manifest.append({"sample_id": f"{uid}__{d}", "pair_uid": uid, "direction": d, "text1": t1, "text2": t2, "label": l,
                                 "source": "luna", "generator_model": e.get("visible_model", ""), "batch_id": bid, "attempt": e["attempt"],
                                 "context_parent_families": ";".join(b["context_parent_families"]),
                                 "context_parent_ids": ";".join(b["context_parent_ids"]),
                                 "usable_folds": ";".join(map(str, b["usable_folds"])), "tier_by_fold": "", "data_version": data_version})
    fields = ["sample_id", "pair_uid", "direction", "text1", "text2", "label", "source", "generator_model", "batch_id", "attempt",
              "context_parent_families", "context_parent_ids", "usable_folds", "tier_by_fold", "data_version"]
    write_csv(f"{out}/manifest.csv", manifest, fields)
    write_csv(f"{out}/filter_decisions.csv", decisions, ["pair_uid", "batch_id", "text1", "text2", "label", "removed"])
    write_csv(f"{out}/attempts.csv", attempts_report, ["batch_id", "attempt", "status", "reasons", "raw_sha256", "visible_model"])

    # 停止条件（审核那一条由 score 子命令算）
    removed = Counter(d["removed"] for d in decisions if d["removed"])
    n_pairs = len(decisions)
    orig = canonical_original(train)
    gen = [(p["text1"], p["text2"], p["label"]) for _, _, p in kept]
    mw = lambda ps: sum(words(x) for t1, t2, _ in ps for x in (t1, t2)) / (2 * len(ps)) if ps else float("nan")
    acc_o, acc_g = probe(orig), probe(gen)
    stats = {
        "batches": len(batches), "failed_batches": failed, "batch_fail_rate": len(failed) / len(batches),
        "pairs_returned": n_pairs, "removed": dict(removed),
        "dup_rate": (removed["dup_parent"] + removed["dup_official"] + removed["dup_generated"]) / n_pairs if n_pairs else None,
        "kept_pairs_by_label": dict(Counter(p["label"] for _, _, p in kept)), "manifest_rows": len(manifest),
        "mean_words_generated": mw(gen), "mean_words_original": mw(orig),
        "probe_balanced_acc_generated": acc_g, "probe_balanced_acc_original": acc_o,
        "diagnostic_phrase_reuse": {k: v / n_pairs for k, v in reuse.items()} if n_pairs else {},
    }
    stats["length_dev"] = abs(stats["mean_words_generated"] - stats["mean_words_original"]) / stats["mean_words_original"] if gen else None
    stats["probe_gap"] = acc_g - acc_o if acc_g is not None and acc_o is not None else None
    stats["stop"] = {"batch_fail_rate": stats["batch_fail_rate"] > STOP["batch_fail_rate"],
                     "dup_rate": stats["dup_rate"] is not None and stats["dup_rate"] > STOP["dup_rate"],
                     "probe_gap": stats["probe_gap"] is not None and stats["probe_gap"] > STOP["probe_gap"],
                     "length_dev": stats["length_dev"] is not None and stats["length_dev"] > STOP["length_dev"],
                     "review_mismatch": "待审核计分（score 子命令）"}
    stats["thresholds"] = STOP
    stats["problems"] = problems

    # 审核抽样：每个生成标签抽 20 对，固定种子；随机决定展示方向，隐藏生成标签和父样本
    rng = random.Random(REVIEW_SEED)
    sheet, key = [], {}
    for lab in ("EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"):
        pool = sorted([k for k in kept if k[2]["label"] == lab], key=lambda k: k[0])
        for uid, bid, p in rng.sample(pool, min(REVIEW_PER_GROUP, len(pool))):
            flip = rng.random() < 0.5
            t1, t2 = (p["text2"], p["text1"]) if flip else (p["text1"], p["text2"])
            exp = {"EQUIVALENCE": "EQUIVALENCE", "NEGATIVE_OTHER": "NEGATIVE_OTHER"}.get(lab, "BACKWARD_ENTAILMENT" if flip else "FORWARD_ENTAILMENT")
            sheet.append({"text1": t1, "text2": t2, "_uid": uid, "_exp": exp, "_grp": lab})
    rng.shuffle(sheet)
    rows = []
    for i, s in enumerate(sheet, 1):
        rid = f"R{i:03d}"
        key[rid] = {"pair_uid": s["_uid"], "generated_label_shown_orientation": s["_exp"], "group": s["_grp"]}
        rows.append({"review_id": rid, "text1": s["text1"], "text2": s["text2"], "label": "", "note": ""})
    write_csv(f"{out}/review_sheet.csv", rows, ["review_id", "text1", "text2", "label", "note"])
    json.dump({"seed": REVIEW_SEED, "per_group": REVIEW_PER_GROUP, "items": key}, open(f"{out}/review_key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"filter_version": FILTER_VERSION, "data_version": data_version, "split_manifest_sha256": pkg_manifest["split_manifest_sha256"],
            "package_manifest_sha256": sha_file(f"{pkg}/MANIFEST.json"), "dispatch_log_sha256": sha_file(f"{ret}/dispatch_log.jsonl"),
            "manifest_sha256": sha_file(f"{out}/manifest.csv"), "processor_sha256": sha_file(__file__),
            "check_format_sha256": sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_format.py"))}
    json.dump({"meta": meta, "stats": stats}, open(f"{out}/report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"meta": meta, "stats": stats}, ensure_ascii=False, indent=1))
    return stats


def read_ann(p):
    out = {}
    for r in read_csv(p):
        v = r["label"].strip().upper()
        out[r["review_id"]] = SHORT.get(v, v) if v in SHORT or v in LABELS else "UNSURE"
    return out


def score(key_path, protocol, a_path, b_path):
    key = json.load(open(key_path, encoding="utf-8"))["items"]
    a, b = read_ann(a_path), read_ann(b_path)
    res = defaultdict(lambda: Counter())
    for rid, k in key.items():
        g, exp = k["group"], k["generated_label_shown_orientation"]
        if protocol == "P1":  # 两人各自独立标注；只看两人都给确定标签且一致的项
            x, y = a.get(rid, "UNSURE"), b.get(rid, "UNSURE")
            if "UNSURE" in (x, y): res[g]["unsure"] += 1; continue
            if x != y: res[g]["annotators_disagree"] += 1; continue
            res[g]["agreed"] += 1; res[g]["agreed_ne_generated"] += x != exp
        else:  # P2：Claude 独立标注，Pan-Yan-0 的裁决覆盖它（裁决表只含被裁决的项）
            x = b.get(rid) or a.get(rid, "UNSURE")
            if x == "UNSURE": res[g]["unsure"] += 1; continue
            res[g]["final"] += 1; res[g]["final_ne_generated"] += x != exp
    out = {}
    for g, c in res.items():
        den = c["agreed"] if protocol == "P1" else c["final"]
        num = c["agreed_ne_generated"] if protocol == "P1" else c["final_ne_generated"]
        out[g] = dict(c, mismatch_rate=num / den if den else None)
    stop = any(v["mismatch_rate"] is not None and v["mismatch_rate"] > STOP["review_mismatch"] for v in out.values())
    print(json.dumps({"protocol": protocol, "by_group": out, "stop": stop, "threshold": STOP["review_mismatch"]}, ensure_ascii=False, indent=1))
    return out, stop


def adjudication_list(key_path, claude_path, out_path, n_spot=10):
    """P2：列出要 Pan-Yan-0 裁决的项——Claude 和生成标签不同（或 Claude 填了无法确定）的全部项，再加固定种子抽的 n_spot 个一致项。"""
    key = json.load(open(key_path, encoding="utf-8"))["items"]
    a = read_ann(claude_path)
    diff = sorted(r for r, k in key.items() if a.get(r, "UNSURE") != k["generated_label_shown_orientation"])
    agree = sorted(r for r in key if r not in diff)
    spot = sorted(random.Random(REVIEW_SEED + 1).sample(agree, min(n_spot, len(agree))))
    sheet = {r["review_id"]: r for r in read_csv(os.path.join(os.path.dirname(key_path), "review_sheet.csv"))}
    rows = [{"review_id": r, "text1": sheet[r]["text1"], "text2": sheet[r]["text2"], "claude": a.get(r, "UNSURE"),
             "generated": key[r]["generated_label_shown_orientation"], "why": "不一致" if r in diff else "抽查", "label": ""}
            for r in diff + spot]
    write_csv(out_path, rows, ["review_id", "text1", "text2", "claude", "generated", "why", "label"])
    print(f"需要裁决 {len(diff)} 项、抽查 {len(spot)} 项 → {out_path}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        dv = sys.argv[sys.argv.index("--data-version") + 1] if "--data-version" in sys.argv else "luna-pilot-v1"
        build(*map(os.path.abspath, sys.argv[2:6]), dv)
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "adjudicate":
        adjudication_list(sys.argv[2], sys.argv[3], sys.argv[4])
