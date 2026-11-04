"""实验 B 生成侧：处理 GPT 交回的生成输出 → 确定性过滤、候选清单、停止条件统计、审核抽样；审核计分和放行门槛。

  python process_outputs.py build <批次包目录> <交回目录> <官方仓库目录> <输出目录> [--data-version 名称]
  python process_outputs.py adjudicate <输出目录> <Claude标注.csv>                # P2：生成待裁决清单 adjudication_list.csv
  python process_outputs.py score <输出目录> P1 <Pan-Yan-0标注.csv> <Claude标注.csv>
  python process_outputs.py score <输出目录> P2 <Claude标注.csv> <裁决.csv>

build 导出的 manifest.csv 只是候选；能不能进入训练只看 gate.json 的 ready_for_training，
任何一项检查失败、没有执行、依赖缺失、协议违规或审核没有完成，都是 false。

交回目录：outputs/<batch_id>/attempt_<n>.txt（原样全文）和 dispatch_log.jsonl（每次尝试一行）。
尝试上限：每批最多 1 次基础设施重派 + 1 次格式重试，总共不超过 3 次；出现合格输出后不得再追加任何尝试。
调度记录必填：agent_id、start / end（ISO 时间，结束不早于开始）、requested_model、visible_model、prompt_sha256、status；
模型完成的尝试必须有 raw_sha256 且和原文一致；失败必须写 reason；infra_error 必须写 has_partial_output，有部分输出就保留原文和哈希。
审核结果绑定 manifest / review_sheet / review_key / report 的哈希，任一变化即失效；本试点只认 P1。
过滤（按顺序，命中第一条就去掉；只做确定性检查，不做语义筛选）：
  same_text            两侧规范化后相同
  length               任一侧词数不在 1–20
  dup_parent           和本批某个父样本是同一短语对（任一方向）
  reuse_parent_phrase  任一侧和本批某个父样本的短语相同（违反提示词第 1 条）
  dup_official         和官方训练集或 dev 的某个短语对相同（任一方向）
  dup_generated        和前面已保留的生成短语对相同（任一方向；按批次号、批内顺序）
规范化：转小写、合并空白。方向展开：E → E/E；FORWARD → F（原方向）/ B（反向）；O 只保留生成的方向。
"""
import csv, hashlib, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_format import check  # noqa: E402

FILTER_VERSION = "augb-filter-v2"
REVIEW_SEED, REVIEW_PER_GROUP, MIN_DECIDED, N_SPOT = 20261104, 20, 10, 10
STOP = {"batch_fail_rate": 0.05, "dup_rate": 0.10, "review_mismatch": 0.30, "probe_gap": 0.15, "length_dev": 0.50}
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SHORT = {"E": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "B": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER", "U": "UNSURE", "UNSURE": "UNSURE"}
GROUPS = ("EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER")


def norm(s): return " ".join(s.lower().split())
def words(s): return len(s.split())
def sha_file(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()


def read_csv(p):
    with open(p, encoding="utf-8") as f: return list(csv.DictReader(f))


def write_csv(p, rows, fields):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def dump(p, obj):
    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1)


def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)


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
    """表面特征逻辑回归的五折平衡准确率；依赖缺失或样本太少时返回 None（门槛视为没有执行，不放行）。"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return None
    X = [features(t1, t2) for t1, t2, _ in pairs]; y = [l for _, _, l in pairs]
    if len(set(y)) < 2 or min(Counter(y).values()) < 5: return None
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return balanced_accuracy_score(y, pred)


def parse_time(x):
    from datetime import datetime
    if not isinstance(x, str): raise TypeError
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def verdict(value, threshold):
    if value is None: return "not_run"
    return "fail" if value > threshold else "pass"


DERIVED = ("manifest.csv", "filter_decisions.csv", "attempts.csv", "review_sheet.csv", "review_key.json", "report.json",
           "gate.json", "review_result.json", "adjudication_list.csv")


def build(pkg, ret, off, out, data_version):
    os.makedirs(out, exist_ok=True)
    for name in DERIVED:  # 重建时让旧的审核结果和放行记录全部失效
        if os.path.exists(f"{out}/{name}"): os.remove(f"{out}/{name}")
    problems = []
    pkg_manifest = load(f"{pkg}/MANIFEST.json")
    for rel, h in pkg_manifest["files"].items():
        if sha_file(f"{pkg}/{rel}") != h: problems.append(f"批次包文件变了：{rel}")
    batches = {b["batch_id"]: b for b in (load(f"{pkg}/batches/{x}") for x in sorted(os.listdir(f"{pkg}/batches")))}
    log = []
    with open(f"{ret}/dispatch_log.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip(): log.append(json.loads(l))
    attempts_report, chosen = [], {}
    by_batch = defaultdict(list)
    for e in log:
        if e["batch_id"] not in batches: problems.append(f"调度记录里有包外批次：{e['batch_id']}")
        by_batch[e["batch_id"]].append(e)
    for bid, b in batches.items():
        entries = sorted(by_batch.get(bid, []), key=lambda e: e["attempt"])
        if not entries: problems.append(f"{bid}：没有调度记录"); continue
        if [e["attempt"] for e in entries] != list(range(1, len(entries) + 1)): problems.append(f"{bid}：尝试编号不连续")
        infra = [e for e in entries if e["status"] == "infra_error"]
        model = [e for e in entries if e["status"] != "infra_error"]
        if len(infra) > 1: problems.append(f"{bid}：基础设施重派超过 1 次")
        if len(model) > 2: problems.append(f"{bid}：模型尝试超过 2 次")
        if len(entries) > 3: problems.append(f"{bid}：总尝试超过 3 次")
        first_ok = next((e["attempt"] for e in entries if e["status"] == "ok"), None)
        if first_ok is not None and entries[-1]["attempt"] > first_ok: problems.append(f"{bid}：已有合格输出后又追加了尝试")
        for e in entries:
            tag = f"{bid} 第 {e['attempt']} 次"
            if e.get("prompt_sha256") != b["prompt_sha256"]: problems.append(f"{tag}：提示词哈希不一致")
            for k in ("agent_id", "start", "end", "requested_model", "visible_model"):
                if not e.get(k): problems.append(f"{tag}：调度记录缺 {k}")
            if e["status"] not in ("ok", "format_fail", "infra_error"): problems.append(f"{tag}：状态不合法 {e['status']}")
            if e["status"] != "ok" and not e.get("reason"): problems.append(f"{tag}：失败没有写原因")
            try:
                t0, t1 = parse_time(e.get("start")), parse_time(e.get("end"))
                if t1 < t0: problems.append(f"{tag}：结束时间早于开始时间")
            except (TypeError, ValueError):
                problems.append(f"{tag}：起止时间不是 ISO 格式")
            path = f"{ret}/outputs/{bid}/attempt_{e['attempt']}.txt"
            if e["status"] == "infra_error":
                # 基础设施失败：必须写明有没有部分输出；有就保留原文并核对哈希
                if "has_partial_output" not in e: problems.append(f"{tag}：基础设施失败没写 has_partial_output")
                elif e["has_partial_output"] != os.path.exists(path): problems.append(f"{tag}：has_partial_output 和实际文件不符")
                elif os.path.exists(path) and e.get("raw_sha256") != sha_file(path): problems.append(f"{tag}：部分输出的哈希和日志不同")
                attempts_report.append({"batch_id": bid, "attempt": e["attempt"], "status": "infra_error", "reasons": e.get("reason", "")})
                continue
            if not os.path.exists(path): problems.append(f"{bid}：缺 attempt_{e['attempt']}.txt"); continue
            with open(path, encoding="utf-8") as f: r = check(b, f.read())
            if r["status"] != e["status"]: problems.append(f"{tag}：日志状态 {e['status']}，脚本判定 {r['status']}")
            if e.get("raw_sha256") != r["raw_sha256"]: problems.append(f"{tag}：原文哈希缺失或和日志不同")
            attempts_report.append({"batch_id": bid, "attempt": e["attempt"], "status": r["status"], "reasons": "；".join(r["reasons"]),
                                    "raw_sha256": r["raw_sha256"], "visible_model": e.get("visible_model", "")})
            if r["status"] == "ok" and bid not in chosen: chosen[bid] = (e, r)
    failed = sorted(set(batches) - set(chosen))

    # ---------- 过滤和展开 ----------
    train = read_csv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    dev = read_csv(f"{off}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    official = {(norm(r["text1"]), norm(r["text2"])) for r in train + dev}
    official |= {(b, a) for a, b in official}
    off_phr = {norm(x) for r in train + dev for x in (r["text1"], r["text2"])}
    kept_keys, kept, decisions, manifest = set(), [], [], []
    for bid in sorted(chosen):
        e, r = chosen[bid]; b = batches[bid]
        parent = {(norm(p["text1"]), norm(p["text2"])) for p in b["parents"]}
        parent |= {(y, x) for x, y in parent}
        parent_phr = {norm(x) for p in b["parents"] for x in (p["text1"], p["text2"])}
        for i, p in enumerate(r["pairs"]):
            uid = f"{bid}-{i:03d}"
            a, c = norm(p["text1"]), norm(p["text2"])
            if a == c: why = "same_text"
            elif not (1 <= words(a) <= 20 and 1 <= words(c) <= 20): why = "length"
            elif (a, c) in parent: why = "dup_parent"
            elif a in parent_phr or c in parent_phr: why = "reuse_parent_phrase"
            elif (a, c) in official: why = "dup_official"
            elif (a, c) in kept_keys: why = "dup_generated"
            else: why = ""
            decisions.append({"pair_uid": uid, "batch_id": bid, "text1": p["text1"], "text2": p["text2"], "label": p["label"],
                              "removed": why, "any_official_phrase": int(a in off_phr or c in off_phr)})
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
    write_csv(f"{out}/filter_decisions.csv", decisions, ["pair_uid", "batch_id", "text1", "text2", "label", "removed", "any_official_phrase"])
    write_csv(f"{out}/attempts.csv", attempts_report, ["batch_id", "attempt", "status", "reasons", "raw_sha256", "visible_model"])

    # ---------- 停止条件 ----------
    removed = Counter(d["removed"] for d in decisions if d["removed"])
    n_pairs = len(decisions)
    orig = canonical_original(train)
    gen = [(p["text1"], p["text2"], p["label"]) for _, _, p in kept]
    mw = lambda ps: sum(words(x) for t1, t2, _ in ps for x in (t1, t2)) / (2 * len(ps))
    acc_o, acc_g = probe(orig), probe(gen)
    stats = {"batches": len(batches), "failed_batches": failed, "batch_fail_rate": len(failed) / len(batches),
             "pairs_returned": n_pairs, "removed": dict(removed),
             "dup_rate": (removed["dup_parent"] + removed["dup_official"] + removed["dup_generated"]) / n_pairs if n_pairs else None,
             "reuse_parent_phrase_rate": removed["reuse_parent_phrase"] / n_pairs if n_pairs else None,
             "any_official_phrase_rate": sum(d["any_official_phrase"] for d in decisions) / n_pairs if n_pairs else None,
             "kept_pairs_by_label": dict(Counter(p["label"] for _, _, p in kept)), "manifest_rows": len(manifest),
             "mean_words_generated": mw(gen) if gen else None, "mean_words_original": mw(orig),
             "probe_balanced_acc_generated": acc_g, "probe_balanced_acc_original": acc_o}
    stats["length_dev"] = abs(stats["mean_words_generated"] - stats["mean_words_original"]) / stats["mean_words_original"] if gen else None
    stats["probe_gap"] = acc_g - acc_o if acc_g is not None and acc_o is not None else None
    checks = {"protocol": "fail" if problems else "pass",
              "batch_fail_rate": verdict(stats["batch_fail_rate"], STOP["batch_fail_rate"]),
              "dup_rate": verdict(stats["dup_rate"], STOP["dup_rate"]),
              "probe_gap": verdict(stats["probe_gap"], STOP["probe_gap"]),
              "length_dev": verdict(stats["length_dev"], STOP["length_dev"]),
              "review": "pending"}

    # ---------- 审核抽样：每个生成标签抽 20 对，固定种子；展示方向随机，隐藏生成标签和父样本 ----------
    rng = random.Random(REVIEW_SEED)
    sheet, key, short_of = [], {}, {}
    for lab in GROUPS:
        pool = sorted([k for k in kept if k[2]["label"] == lab], key=lambda k: k[0])
        if len(pool) < REVIEW_PER_GROUP: short_of[lab] = len(pool)
        for uid, bid, p in rng.sample(pool, min(REVIEW_PER_GROUP, len(pool))):
            flip = rng.random() < 0.5
            t1, t2 = (p["text2"], p["text1"]) if flip else (p["text1"], p["text2"])
            exp = lab if lab != "FORWARD_ENTAILMENT" else ("BACKWARD_ENTAILMENT" if flip else "FORWARD_ENTAILMENT")
            sheet.append({"text1": t1, "text2": t2, "_uid": uid, "_exp": exp, "_grp": lab})
    rng.shuffle(sheet)
    rows = []
    for i, s in enumerate(sheet, 1):
        rid = f"R{i:03d}"
        key[rid] = {"pair_uid": s["_uid"], "generated_label_shown_orientation": s["_exp"], "group": s["_grp"]}
        rows.append({"review_id": rid, "text1": s["text1"], "text2": s["text2"], "label": "", "note": ""})
    write_csv(f"{out}/review_sheet.csv", rows, ["review_id", "text1", "text2", "label", "note"])
    dump(f"{out}/review_key.json", {"seed": REVIEW_SEED, "per_group": REVIEW_PER_GROUP, "short_of_quota": short_of, "items": key})
    if short_of: checks["review"] = "fail"; problems.append(f"可审核的对不足每类 {REVIEW_PER_GROUP}：{short_of}")
    meta = {"filter_version": FILTER_VERSION, "data_version": data_version, "status": "candidate",
            "split_manifest_sha256": pkg_manifest["split_manifest_sha256"], "package_version": pkg_manifest.get("version"),
            "package_manifest_sha256": sha_file(f"{pkg}/MANIFEST.json"), "dispatch_log_sha256": sha_file(f"{ret}/dispatch_log.jsonl"),
            "manifest_sha256": sha_file(f"{out}/manifest.csv"), "processor_sha256": sha_file(__file__),
            "check_format_sha256": sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_format.py"))}
    report = {"meta": meta, "stats": stats, "checks": checks, "thresholds": STOP, "problems": problems}
    dump(f"{out}/report.json", report)
    write_gate(out)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def bound_files(out):
    return {n: sha_file(f"{out}/{n}") for n in ("manifest.csv", "review_sheet.csv", "review_key.json", "report.json")}


def write_gate(out):
    report = load(f"{out}/report.json")
    checks = dict(report["checks"])
    current = bound_files(out)
    if os.path.exists(f"{out}/review_result.json"):
        rr = load(f"{out}/review_result.json")
        checks["review"] = rr["status"] if rr.get("bound_to") == current and rr.get("protocol") == "P1" else "stale_or_wrong_protocol"
    gate = {"checks": checks, "ready_for_training": all(v == "pass" for v in checks.values()), "bound_to": current,
            "rule": "所有检查都是 pass 才放行；fail / not_run / pending / invalid / insufficient / stale 都不放行。"
                    "训练入口必须核对 manifest.csv 的 SHA-256 等于 bound_to 里的值"}
    dump(f"{out}/gate.json", gate)
    return gate


def read_ann(p, ids):
    """读标注表并校验：恰好覆盖 ids、没有重复、标签合法（E/F/B/O/U 或全名；U = 无法确定）。返回 (标签, 错误列表)。"""
    errs, out = [], {}
    for r in read_csv(p):
        rid, v = r.get("review_id", ""), (r.get("label") or "").strip().upper()
        if rid in out: errs.append(f"重复编号 {rid}"); continue
        lab = SHORT.get(v) or (v if v in LABELS else None)
        if lab is None: errs.append(f"{rid} 标签缺失或不合法：{v!r}"); continue
        out[rid] = lab
    if set(out) - set(ids): errs.append(f"多出编号：{sorted(set(out) - set(ids))[:5]}")
    if set(ids) - set(out) - {e.split()[0] for e in errs}: errs.append(f"缺少编号：{sorted(set(ids) - set(out))[:5]}")
    return out, errs


def adjudication_list(out, claude_path):
    """P2：Claude 和生成标签不同（含无法确定）的全部项，加固定种子抽的 N_SPOT 个一致项。"""
    key = load(f"{out}/review_key.json")["items"]
    a, errs = read_ann(claude_path, list(key))
    if errs: raise SystemExit("Claude 标注表不合格：" + "；".join(errs))
    diff = sorted(r for r, k in key.items() if a[r] != k["generated_label_shown_orientation"])
    agree = sorted(r for r in key if r not in diff)
    spot = sorted(random.Random(REVIEW_SEED + 1).sample(agree, min(N_SPOT, len(agree))))
    sheet = {r["review_id"]: r for r in read_csv(f"{out}/review_sheet.csv")}
    rows = [{"review_id": r, "text1": sheet[r]["text1"], "text2": sheet[r]["text2"], "claude": a[r],
             "generated": key[r]["generated_label_shown_orientation"], "why": "不一致" if r in diff else "抽查", "label": ""}
            for r in diff + spot]
    write_csv(f"{out}/adjudication_list.csv", rows, ["review_id", "text1", "text2", "claude", "generated", "why", "label"])
    print(f"需要裁决 {len(diff)} 项、抽查 {len(spot)} 项 → {out}/adjudication_list.csv")
    return rows


def score(out, protocol, a_path, b_path):
    """P1：a = Pan-Yan-0，b = Claude，各自独立标注全部项；只看两人都给确定标签且一致的项。
    P2：a = Claude 全部项；b = 裁决表，必须覆盖 adjudication_list.csv 的全部项；最终标签 = 裁决（有）或 Claude 的。
    每类：抽样不足 20、或可判定的项少于 MIN_DECIDED → insufficient；不一致率 > 30% → fail；否则 pass。"""
    keyfile = load(f"{out}/review_key.json"); key = keyfile["items"]
    errs = []
    a, ea = read_ann(a_path, list(key)); errs += [f"标注 A：{e}" for e in ea]
    if protocol == "P1":
        b, eb = read_ann(b_path, list(key)); errs += [f"标注 B：{e}" for e in eb]
    else:
        need = [r["review_id"] for r in read_csv(f"{out}/adjudication_list.csv")]
        b, eb = read_ann(b_path, need); errs += [f"裁决：{e}" for e in eb]
    res = {g: Counter() for g in GROUPS}
    for rid, k in key.items():
        g, exp = k["group"], k["generated_label_shown_orientation"]
        res[g]["sampled"] += 1
        if protocol == "P1":
            x, y = a.get(rid), b.get(rid)
            if x is None or y is None: continue
            if "UNSURE" in (x, y): res[g]["unsure"] += 1; continue
            if x != y: res[g]["annotators_disagree"] += 1; continue
            res[g]["decided"] += 1; res[g]["mismatch"] += x != exp
        else:
            x = b.get(rid) or a.get(rid)
            if x is None: continue
            if x == "UNSURE": res[g]["unsure"] += 1; continue
            res[g]["decided"] += 1; res[g]["mismatch"] += x != exp
    by_group = {}
    for g, c in res.items():
        rate = c["mismatch"] / c["decided"] if c["decided"] else None
        if c["sampled"] < REVIEW_PER_GROUP or c["decided"] < MIN_DECIDED: st = "insufficient"
        else: st = "fail" if rate > STOP["review_mismatch"] else "pass"
        by_group[g] = dict(c, mismatch_rate=rate, status=st)
    if errs: status = "invalid"
    elif any(v["status"] == "fail" for v in by_group.values()): status = "fail"
    elif any(v["status"] == "insufficient" for v in by_group.values()): status = "insufficient"
    else: status = "pass"
    result = {"protocol": protocol, "status": status, "errors": errs, "by_group": by_group,
              "threshold": STOP["review_mismatch"], "min_decided_per_group": MIN_DECIDED,
              "inputs_sha256": {"a": sha_file(a_path), "b": sha_file(b_path)}, "bound_to": bound_files(out)}
    dump(f"{out}/review_result.json", result)
    gate = write_gate(out)
    print(json.dumps({"review": result, "gate": gate}, ensure_ascii=False, indent=1))
    return result, gate


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        dv = sys.argv[sys.argv.index("--data-version") + 1] if "--data-version" in sys.argv else "luna-pilot"
        build(*map(os.path.abspath, sys.argv[2:6]), dv)
    elif cmd == "adjudicate":
        adjudication_list(os.path.abspath(sys.argv[2]), sys.argv[3])
    elif cmd == "score":
        score(os.path.abspath(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
