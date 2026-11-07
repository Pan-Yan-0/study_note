"""全量 +1 的合并、逐折选用、第二次 P1 抽样和总放行（方案 B；规则在看到全量输出之前定下）。

  python merge_full.py <官方仓库目录> <合并输出目录> pilot=<包目录>:<处理结果目录> main=<包目录>:<处理结果目录> [supp1=...] [supp2=...] [--pilot-unreviewed]

--pilot-unreviewed（Haiku 单一来源，FULL_README 第 10 节）：试点阶段只做过自动检查、没有单独做 P1。
  这时试点按其他阶段的方式核对（协议、清单哈希），另外要求它的自动停止条件全部通过；
  第二次 P1 的总体是全部阶段的被选用对（不再只抽新增），这次 P1 就是全部数据唯一的一次人工审核。

各阶段先各自跑 process_outputs.py build；本脚本只做跨阶段的事：
1. 来源核对：试点的 gate.json 必须已放行，manifest.csv 的哈希等于它的 bound_to；其余阶段的 report.json 协议必须 pass，
   manifest.csv 的哈希等于 report 里记的值。所有阶段的划分清单哈希相同。
2. 跨阶段去重：阶段顺序 pilot → main → supp1 → supp2，后面的对和前面任一阶段的对相同（任一方向）就去掉，前面的优先。
3. 直接父样本：按 batch_id 和过滤前的位置（pair_uid 的序号，等于 Rk 的 k−1）回填 direct_parent_family_id / sample_id / ref_index，
   核对生成标签等于父样本标签。
4. 隔离：每一行的 usable_folds 必须是批内所有上下文家族签名的交集，而且包含在直接父家族的签名里。
5. 逐折选用（fold_selection.csv）：每折、每类的需求 = 该折 fit 的行数。候选是 usable_folds 含该折、标签组相同的对；
   按阶段顺序，阶段内用固定种子打乱，依次取整对（同一对的两个方向一起选），直到行数恰好等于需求。
   E 对 2 行 E；单向对 1 行 F + 1 行 B；O 对 1 行 O。供给不够就记 fold_quota 失败。
6. 自动检查：合并后的全部候选按冻结的停止条件重算（失败批次、重复、探针、词数）；新增部分（非试点）另算一份，只描述。
7. 第二次 P1：总体 = 新增的、过滤后、至少被一折选用的对（每个 pair_uid 只算一次）；每个生成标签抽 20 对，种子 REVIEW2_SEED，
   展示方向随机，隐藏生成标签和父样本。计分用 process_outputs.py score，门槛不变。
8. 集中度：按直接父家族报告派发数、保留数、各折选用数；独立家族数、最大值、子对数最多的前 1% 家族
   （ceil(1% × 有子对的家族数) 个）的占比。同一对的两个方向算一对。
gate.json 绑定 manifest.csv、fold_selection.csv、sources.json、review_sheet.csv、review_key.json、report.json 的哈希。
"""
import csv, hashlib, json, math, os, random, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import process_outputs as po  # noqa: E402

STAGES = ("pilot", "main", "supp1", "supp2", "r2", "s2a", "s2b", "r3", "r4", "s4a", "s4b")  # 规模轮（+2 / +4）排在 +1 的阶段之后，+1 的选用不变
SELECT_SEED, REVIEW2_SEED = 20261106, 20261107
GROUP_OF = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
            "NEGATIVE_OTHER": "NEGATIVE_OTHER"}
FIELDS = ["sample_id", "pair_uid", "direction", "text1", "text2", "label", "source", "generator_model", "batch_id", "attempt",
          "context_parent_families", "context_parent_ids", "usable_folds", "tier_by_fold", "data_version",
          "stage", "direct_parent_family_id", "direct_parent_sample_id", "ref_index"]


def sha(p): return po.sha_file(p)


def merge(off, out, specs, pilot_reviewed=True, tiers=(1,), review_stages=None):
    """tiers：要选用的规模（+1 / +2 / +4，嵌套：每折每类按同一个固定顺序取前 k × fit 行）。
    review_stages：第二次 P1 只从这些阶段抽（规模轮只审新增的那部分）；None 时沿用原规则。"""
    os.makedirs(out, exist_ok=True)
    for name in po.DERIVED + ("fold_selection.csv", "sources.json", "concentration.json", "fold_selection_scale.csv", "pool_order.json"):
        if os.path.exists(f"{out}/{name}"): os.remove(f"{out}/{name}")
    problems, sources = [], {}
    stages = [s for s in STAGES if s in specs]
    assert stages[:2] == ["pilot", "main"], "至少要有 pilot 和 main"
    fam = None; split_sha = None; batches = {}; stage_reports = {}
    for st in stages:
        pkg, res = specs[st]
        man = po.load(f"{pkg}/MANIFEST.json")
        for rel, h in man["files"].items():
            if sha(f"{pkg}/{rel}") != h: problems.append(f"{st}：批次包文件变了 {rel}")
        if split_sha is None: split_sha = man["split_manifest_sha256"]
        elif man["split_manifest_sha256"] != split_sha: problems.append(f"{st}：划分清单哈希和试点不同")
        rep = po.load(f"{res}/report.json"); stage_reports[st] = rep
        msha = sha(f"{res}/manifest.csv")
        if st == "pilot" and pilot_reviewed:
            gate = po.load(f"{res}/gate.json")
            if not gate["ready_for_training"] or gate["bound_to"]["manifest.csv"] != msha: problems.append("试点没有放行或清单哈希不符")
        else:
            if rep["checks"]["protocol"] != "pass": problems.append(f"{st}：协议检查没有通过")
            if rep["meta"]["manifest_sha256"] != msha: problems.append(f"{st}：清单哈希和 report 不符")
            if st == "pilot":
                bad = [k for k, v in rep["checks"].items() if k != "review" and v != "pass"]
                if bad: problems.append(f"试点（未单独审核）的自动检查没有全部通过：{bad}")
        for x in sorted(os.listdir(f"{pkg}/batches")):
            b = po.load(f"{pkg}/batches/{x}"); batches[b["batch_id"]] = (st, b)
        sources[st] = {"package_manifest_sha256": sha(f"{pkg}/MANIFEST.json"), "manifest_sha256": msha,
                       "report_sha256": sha(f"{res}/report.json"), "dispatch_log_sha256": rep["meta"]["dispatch_log_sha256"]}
    world_split = po.load(f"{specs['pilot'][0]}/split_manifest.json")["folds"]
    sigs = po.load(f"{specs['pilot'][0]}/family_signatures.json")["signature_of_family"]
    train = po.read_csv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    lab_of = {r["instance_id"]: r["label"] for r in train}

    # ---------- 2–4：读各阶段清单，跨阶段去重，回填直接父样本，核对隔离 ----------
    seen, rows, dropped, pairs = set(), [], Counter(), {}
    for st in stages:
        by_uid = defaultdict(list)
        for r in po.read_csv(f"{specs[st][1]}/manifest.csv"): by_uid[r["pair_uid"]].append(r)
        for uid in sorted(by_uid):
            rs = by_uid[uid]; o = next(r for r in rs if r["direction"] == "original")
            key = (po.norm(o["text1"]), po.norm(o["text2"]))
            if key in seen: dropped[st] += 1; continue
            seen |= {key, key[::-1]}
            bst, b = batches[o["batch_id"]]
            if bst != st: problems.append(f"{uid}：批次阶段不符")
            k = int(uid.rsplit("-", 1)[1]); p = b["parents"][k]
            gen_lab = o["label"]
            if p["label"] != gen_lab: problems.append(f"{uid}：生成标签和直接父样本 R{k + 1} 不同")
            inter = set.intersection(*[set(sigs[f]) for f in b["context_parent_families"]])
            usable = {int(x) for x in o["usable_folds"].split(";") if x != ""}
            if usable != inter or not usable <= set(sigs[p["family_id"]]): problems.append(f"{uid}：可用折和上下文家族签名不符")
            pairs[uid] = {"stage": st, "label": gen_lab, "usable": sorted(usable), "family": p["family_id"],
                          "rows": [r["sample_id"] for r in rs], "text": (o["text1"], o["text2"])}
            for r in rs:
                rows.append({**{f: r.get(f, "") for f in FIELDS[:15]}, "stage": st, "direct_parent_family_id": p["family_id"],
                             "direct_parent_sample_id": p["sample_id"], "ref_index": k + 1})
    po.write_csv(f"{out}/manifest.csv", rows, FIELDS)

    # ---------- 5：逐折选用 ----------
    need_rows = {"EQUIVALENCE": 2, "FORWARD_ENTAILMENT": 2, "NEGATIVE_OTHER": 1}
    selection, quota, short = [], {}, {}
    scale_rows, order_out, quota_scale, short_scale = [], {}, {}, {}
    top = max(tiers)
    for kf in range(5):
        fit = Counter(lab_of[i] for i in world_split[str(kf)]["fit"])
        demand = {"EQUIVALENCE": fit["EQUIVALENCE"], "FORWARD_ENTAILMENT": fit["FORWARD_ENTAILMENT"] + fit["BACKWARD_ENTAILMENT"],
                  "NEGATIVE_OTHER": fit["NEGATIVE_OTHER"]}
        assert fit["FORWARD_ENTAILMENT"] == fit["BACKWARD_ENTAILMENT"] and demand["EQUIVALENCE"] % 2 == 0
        for g, d in demand.items():
            cand = [u for u, x in pairs.items() if x["label"] == g and kf in x["usable"]]
            rng = random.Random(f"{SELECT_SEED}-{kf}-{g}")
            ordered = []
            for st in STAGES:
                part = sorted(u for u in cand if pairs[u]["stage"] == st); rng.shuffle(part); ordered += part
            order_out[f"{kf}-{g}"] = ordered
            prev = []
            for t in sorted(tiers):
                tk = ordered[:t * d // need_rows[g]]; gt = len(tk) * need_rows[g]
                quota_scale[f"{t}-{kf}-{g}"] = {"demand_rows": t * d, "selected_rows": gt, "candidates_pairs": len(cand)}
                if gt != t * d and t == top: short_scale[f"{kf}-{g}"] = t * d - gt
                for u in tk[len(prev):]:
                    scale_rows += [{"fold": kf, "sample_id": s, "pair_uid": u, "tier": t} for s in pairs[u]["rows"]]
                prev = tk
            take = ordered[:d // need_rows[g]]
            got = len(take) * need_rows[g]
            quota[f"{kf}-{g}"] = {"demand_rows": d, "selected_rows": got, "candidates_pairs": len(cand)}
            if got != d: short[f"{kf}-{g}"] = d - got
            for u in take: selection += [{"fold": kf, "sample_id": s, "pair_uid": u} for s in pairs[u]["rows"]]
    po.write_csv(f"{out}/fold_selection.csv", selection, ["fold", "sample_id", "pair_uid"])
    selected = {r["pair_uid"] for r in selection}
    if tuple(tiers) != (1,):
        po.write_csv(f"{out}/fold_selection_scale.csv", scale_rows, ["fold", "sample_id", "pair_uid", "tier"])
        po.dump(f"{out}/pool_order.json", {"rule": "每折每组的候选对按阶段顺序、阶段内固定种子打乱；+k 取前 k × fit 行；类别构成的 G 组取各类前缀",
                                           "tiers": sorted(tiers), "order": order_out})
        selected = {r["pair_uid"] for r in scale_rows}
        short = short_scale

    # ---------- 6：自动检查（合并后全部候选；新增部分另算） ----------
    orig = po.canonical_original(train)
    mw = lambda ps: sum(po.words(x) for t1, t2, _ in ps for x in (t1, t2)) / (2 * len(ps))
    def surface(us):
        gen = [(*pairs[u]["text"], pairs[u]["label"]) for u in us]
        acc_g, acc_o = po.probe(gen), po.probe(orig)
        return {"pairs": len(gen), "probe_generated": acc_g, "probe_original": acc_o,
                "probe_gap": acc_g - acc_o if acc_g is not None and acc_o is not None else None,
                "mean_words_generated": mw(gen), "length_dev": abs(mw(gen) - mw(orig)) / mw(orig)}
    allp = sorted(pairs); newp = sorted(u for u in pairs if pairs[u]["stage"] != "pilot" or not pilot_reviewed)
    if review_stages is not None: newp = sorted(u for u in pairs if pairs[u]["stage"] in review_stages)
    n_batches = sum(r["stats"]["batches"] for r in stage_reports.values())
    n_failed = sum(len(r["stats"]["failed_batches"]) for r in stage_reports.values())
    n_ret = sum(r["stats"]["pairs_returned"] for r in stage_reports.values())
    n_dup = sum(sum(r["stats"]["removed"].get(k, 0) for k in ("dup_parent", "dup_official", "dup_generated"))
                for r in stage_reports.values()) + sum(dropped.values())
    s_all, s_new = surface(allp), surface(newp)
    stats = {"stages": stages, "batches": n_batches, "failed_batches": n_failed, "batch_fail_rate": n_failed / n_batches,
             "pairs_returned": n_ret, "dup_rate": n_dup / n_ret, "dropped_cross_stage": dict(dropped),
             "kept_pairs": len(pairs), "kept_pairs_by_stage": dict(Counter(x["stage"] for x in pairs.values())),
             "selected_pairs": len(selected), "fold_quota": quota, "fold_quota_short": short,
             "tiers": sorted(tiers), "fold_quota_scale": quota_scale,
             "surface_all": s_all, "surface_new_only": s_new,
             "probe_gap": s_all["probe_gap"], "length_dev": s_all["length_dev"]}
    checks = {"protocol": "fail" if problems else "pass",
              "batch_fail_rate": po.verdict(stats["batch_fail_rate"], po.STOP["batch_fail_rate"]),
              "dup_rate": po.verdict(stats["dup_rate"], po.STOP["dup_rate"]),
              "probe_gap": po.verdict(stats["probe_gap"], po.STOP["probe_gap"]),
              "length_dev": po.verdict(stats["length_dev"], po.STOP["length_dev"]),
              "fold_quota": "fail" if short else "pass", "review": "pending"}

    # ---------- 7：第二次 P1 抽样 ----------
    rng = random.Random(REVIEW2_SEED)
    sheet, key, short_of = [], {}, {}
    for lab in po.GROUPS:
        pool = sorted(u for u in newp if u in selected and pairs[u]["label"] == lab)
        if len(pool) < po.REVIEW_PER_GROUP: short_of[lab] = len(pool)
        for u in rng.sample(pool, min(po.REVIEW_PER_GROUP, len(pool))):
            flip = rng.random() < 0.5
            t1, t2 = pairs[u]["text"][::-1] if flip else pairs[u]["text"]
            exp = lab if lab != "FORWARD_ENTAILMENT" else ("BACKWARD_ENTAILMENT" if flip else "FORWARD_ENTAILMENT")
            sheet.append({"text1": t1, "text2": t2, "_uid": u, "_exp": exp, "_grp": lab})
    rng.shuffle(sheet)
    rws = []
    for i, s in enumerate(sheet, 1):
        rid = f"R{i:03d}"
        key[rid] = {"pair_uid": s["_uid"], "generated_label_shown_orientation": s["_exp"], "group": s["_grp"],
                    "stage": pairs[s["_uid"]]["stage"], "signature": "".join(map(str, pairs[s["_uid"]]["usable"]))}
        rws.append({"review_id": rid, "text1": s["text1"], "text2": s["text2"], "label": "", "note": ""})
    po.write_csv(f"{out}/review_sheet.csv", rws, ["review_id", "text1", "text2", "label", "note"])
    po.dump(f"{out}/review_key.json", {"seed": REVIEW2_SEED, "per_group": po.REVIEW_PER_GROUP, "population": f"阶段 {sorted(review_stages)} 的、过滤后、至少一折选用" if review_stages is not None else "新增、过滤后、至少一折选用" if pilot_reviewed else "全部阶段、过滤后、至少一折选用（试点未单独审核）",
                                       "short_of_quota": short_of, "items": key,
                                       "coverage": {"stages": dict(Counter(v["stage"] for v in key.values())),
                                                    "signatures": dict(Counter(v["signature"] for v in key.values()))}})
    if short_of: checks["review"] = "fail"; problems.append(f"第二次 P1 可抽的对不足：{short_of}")

    # ---------- 8：集中度 ----------
    disp = Counter(p["family_id"] for st, b in batches.values() for p in b["parents"])
    kept_c = Counter(x["family"] for x in pairs.values())
    sel_c = Counter(pairs[u]["family"] for u in selected)
    top_n = math.ceil(0.01 * len(kept_c))
    top = sum(c for _, c in kept_c.most_common(top_n))
    po.dump(f"{out}/concentration.json", {
        "families_dispatched": len(disp), "families_with_kept_pairs": len(kept_c), "families_selected": len(sel_c),
        "dispatch_per_family": dict(Counter(disp.values())), "kept_pairs_per_family": dict(Counter(kept_c.values())),
        "max_kept_pairs_one_family": max(kept_c.values()),
        "top1pct_rule": "ceil(0.01 × 有子对的家族数) 个家族", "top1pct_families": top_n, "top1pct_share_of_kept_pairs": top / len(pairs)})

    po.dump(f"{out}/sources.json", {"split_manifest_sha256": split_sha, "stages": sources, **({"tiers": sorted(tiers)} if tuple(tiers) != (1,) else {}),
                                     "pilot_reviewed": pilot_reviewed, "select_seed": SELECT_SEED, "review2_seed": REVIEW2_SEED, "merger_sha256": sha(__file__)})
    meta = {"status": "candidate", "split_manifest_sha256": split_sha, "manifest_sha256": sha(f"{out}/manifest.csv"),
            "fold_selection_sha256": sha(f"{out}/fold_selection.csv"), "sources_sha256": sha(f"{out}/sources.json"),
            "merger_sha256": sha(__file__), "processor_sha256": sha(po.__file__)}
    po.dump(f"{out}/report.json", {"meta": meta, "stats": stats, "checks": checks, "thresholds": po.STOP, "problems": problems})
    gate = po.write_gate(out)
    print(json.dumps({"checks": checks, "problems": problems[:10], "kept_pairs_by_stage": stats["kept_pairs_by_stage"],
                      "fold_quota_short": short, "ready_for_training": gate["ready_for_training"]}, ensure_ascii=False, indent=1))
    return gate


if __name__ == "__main__":
    off, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    specs = {}
    for a in sys.argv[3:]:
        if a == "--pilot-unreviewed" or a.startswith("--tiers=") or a.startswith("--review-stages="): continue
        st, v = a.split("=", 1); pkg, res = v.split(":", 1)
        specs[st] = (os.path.abspath(pkg), os.path.abspath(res))
    tiers = tuple(int(x) for x in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tiers=")), "1").split(","))
    rs = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--review-stages=")), None)
    merge(off, out, specs, pilot_reviewed="--pilot-unreviewed" not in sys.argv, tiers=tiers, review_stages=rs)
