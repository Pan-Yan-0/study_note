"""实验 B 生成侧：从实际五折划分算家族签名和配额，建试点批次包。
用法：python build_pilot.py <官方仓库目录> <输出目录> [--families families.json] [--split-manifest split_manifest.json]
默认从仓库读 metrics/diagnostics/families.json 和实验 A 归档的每折 split_ids；在批次包里运行时，
用包内 inputs/families.json 和 split_manifest.json，可以逐字节重建同一个包（test_augment_b 会检查）。

- 签名：一个家族在哪些折里属于 fit。一批只放同一签名的父样本；子样本可用折 = 整批上下文来源可用折的交集。
- 配额按"短语对"下达：E / 单向（以 FORWARD 形式，text1 推出 text2）/ O；方向由 process_outputs.py 确定性展开。
- 父样本按原方向展示；原方向标签是 BACKWARD 的，交换两侧后按 FORWARD 展示（只为和生成标签同一形式）。
"""
import argparse, csv, hashlib, json, os, random, shutil
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SE = os.path.dirname(os.path.dirname(HERE))
VERSION = "augb-pilot-v5"
SEED = 20261104
PILOT = {"batches_4fold": 2, "batches_other": 1, "n_4fold_sigs": 5, "n_other_sigs": 5, "parents_per_batch": 13,
         "quota": {"EQUIVALENCE": 3, "FORWARD_ENTAILMENT": 7, "NEGATIVE_OTHER": 4}}
MARGIN = 1.25
GROUP = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "U", "BACKWARD_ENTAILMENT": "U", "NEGATIVE_OTHER": "O"}
SOURCE_TEXT = {"real": "real examples from the dataset", "hand": "hand-written examples in the style of the dataset"}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    with open(p, "rb") as f: return sha_bytes(f.read())
def canon(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def norm(s): return " ".join(s.lower().split())


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1)


ARTICLES = ("a", "an", "the")


def ref_stats(refs):
    """本批参考对的长度和开头冠词统计，只用本批父样本（不看标签），写进提示词的长度要求。"""
    w = sorted(len(norm(p[k]).split()) for p in refs for k in ("text1", "text2"))
    n = len(w)
    diff = sum(abs(len(norm(p["text1"]).split()) - len(norm(p["text2"]).split())) for p in refs) / len(refs)
    art = sum(norm(p[k]).split()[0] in ARTICLES for p in refs for k in ("text1", "text2"))
    return {"n_ref_phrases": n, "ref_mean_words": f"{sum(w) / n:.1f}", "ref_q1": w[int(0.25 * (n - 1))],
            "ref_q3": w[int(0.75 * (n - 1))], "ref_max_words": w[-1], "ref_mean_diff": f"{diff:.1f}", "ref_article": art}


def fill(template, refs, quota, batch_id, source):
    ref = "\n".join(json.dumps({"text1": p["text1"], "text2": p["text2"], "label": p["label"]}, ensure_ascii=False) for p in refs)
    out = template
    for k, v in ref_stats(refs).items(): out = out.replace("{" + k + "}", str(v))
    out = (out.replace("{reference_source}", SOURCE_TEXT[source]).replace("{reference_pairs}", ref)
           .replace("{n_equivalence}", str(quota["EQUIVALENCE"])).replace("{n_forward}", str(quota["FORWARD_ENTAILMENT"]))
           .replace("{n_other}", str(quota["NEGATIVE_OTHER"])).replace("{n_total}", str(sum(quota.values())))
           .replace("{batch_id}", batch_id))
    assert "{reference_" not in out and "{n_" not in out and "{ref_" not in out
    return out


def load_split(args):
    if args.split_manifest:
        return json.load(open(args.split_manifest, encoding="utf-8"))
    split = {}
    for k in range(5):
        s = json.load(open(f"{SE}/metrics/qwen8b_a1/formal/seed42/fold{k}/task.json", encoding="utf-8"))["split_ids"]
        split[str(k)] = {p: sorted(s[p]) for p in ("fit", "val", "test")}
    return {"source": "metrics/qwen8b_a1/formal/seed42/fold*/task.json split_ids（与 mDeBERTa 联合验证逐行相同）", "folds": split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official"); ap.add_argument("out")
    ap.add_argument("--families", default=f"{SE}/metrics/diagnostics/families.json")
    ap.add_argument("--split-manifest")
    args = ap.parse_args()
    off, out = os.path.abspath(args.official), os.path.abspath(args.out)
    rows = list(csv.DictReader(open(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
    dev = list(csv.DictReader(open(f"{off}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv", encoding="utf-8")))
    fam = json.load(open(args.families, encoding="utf-8"))
    split_manifest = load_split(args); split = split_manifest["folds"]
    byid = {r["instance_id"]: r for r in rows}
    family = {r["instance_id"]: str(fam["family_of_pair"][r["pair_id"]]) for r in rows}
    for k, s in split.items():  # 每折三部分互不重叠、覆盖全部行；留出部分和 families.json 的折号一致
        allids = s["fit"] + s["val"] + s["test"]
        assert len(allids) == len(set(allids)) == len(rows), k
        assert {family[i] for i in s["test"]} == {f for f, x in fam["fold_of_family"].items() if x == int(k)} & set(family.values()), k
    part_of = {(k, i): p for k, s in split.items() for p, ids in s.items() for i in ids}
    fam_rows = defaultdict(list)
    for r in rows: fam_rows[family[r["instance_id"]]].append(r["instance_id"])
    signature = {}
    for f, ids in fam_rows.items():
        folds = []
        for k in split:
            parts = {part_of[(k, i)] for i in ids}
            assert len(parts) == 1, (f, k, parts)  # 一个家族在每折里整体属于同一部分
            if parts == {"fit"}: folds.append(int(k))
        signature[f] = tuple(folds)
    split_sha = sha_bytes(canon(split_manifest).encode())

    # ---------- 签名表和 +1 需求表 ----------
    sig_fams = defaultdict(list)
    for f, s in signature.items(): sig_fams[s].append(f)
    total_group = Counter(GROUP[r["label"]] for r in rows)
    share = {g: total_group[g] / len(rows) for g in "EUO"}
    order = sorted(sig_fams, key=lambda s: (-len(sig_fams[s]), s))
    sig_table, plan = [], {}
    for s in order:
        ids = [i for f in sig_fams[s] for i in fam_rows[f]]
        g = Counter(GROUP[byid[i]["label"]] for i in ids)
        target_rows = len(ids) * MARGIN
        pairs = {"E": round(target_rows * share["E"] / 2), "U": round(target_rows * share["U"] / 2), "O": round(target_rows * share["O"])}
        plan["".join(map(str, s))] = pairs
        sig_table.append({"signature": "".join(map(str, s)), "usable_folds": len(s), "families": len(sig_fams[s]), "rows": len(ids),
                          "rows_E": g["E"], "rows_U": g["U"], "rows_O": g["O"],
                          "plan_pairs_E": pairs["E"], "plan_pairs_U": pairs["U"], "plan_pairs_O": pairs["O"]})
    demand = []
    for k in map(int, split):
        need = Counter(GROUP[byid[i]["label"]] for i in split[str(k)]["fit"])
        supply = Counter()
        for s, p in plan.items():
            if str(k) in s: supply["E"] += 2 * p["E"]; supply["U"] += 2 * p["U"]; supply["O"] += p["O"]
        for g in "EUO":
            demand.append({"fold": k, "group": g, "need_rows_plus1": need[g], "planned_rows_before_filter": supply[g],
                           "ratio": round(supply[g] / need[g], 3)})

    # ---------- 试点：抽父样本、分批 ----------
    rng = random.Random(SEED)
    four = [s for s in order if len(s) == 4][:PILOT["n_4fold_sigs"]]
    other = [s for s in order if len(s) < 4][:PILOT["n_other_sigs"]]
    template_path = os.path.join(HERE, "prompt_template.txt")
    template = open(template_path, encoding="utf-8").read()
    q = PILOT["quota"]
    batches, n = [], 0
    for s in four + other:
        fs = sorted(sig_fams[s]); rng.shuffle(fs)
        for b in range(PILOT["batches_4fold"] if s in four else PILOT["batches_other"]):
            chosen = fs[b * PILOT["parents_per_batch"]:(b + 1) * PILOT["parents_per_batch"]]
            assert len(chosen) == PILOT["parents_per_batch"]
            parents = []
            for f in chosen:
                pid = sorted({byid[i]["pair_id"] for i in fam_rows[f]})[0]
                r = byid[f"{pid}__en-en__original"]
                t1, t2, lab = r["text1"], r["text2"], r["label"]
                if lab == "BACKWARD_ENTAILMENT": t1, t2, lab = t2, t1, "FORWARD_ENTAILMENT"
                parents.append({"sample_id": r["instance_id"], "pair_id": pid, "family_id": f, "usable_folds": list(signature[f]),
                                "text1": t1, "text2": t2, "label": lab, "shown_swapped": r["label"] == "BACKWARD_ENTAILMENT"})
            n += 1
            bid = f"pilot_b{n:02d}"
            prompt = fill(template, parents, q, bid, "real")
            batches.append({"batch_id": bid, "package_version": VERSION, "stage": "pilot", "signature": "".join(map(str, s)),
                            "usable_folds": sorted(set.intersection(*[set(p["usable_folds"]) for p in parents])),
                            "context_parent_ids": [p["sample_id"] for p in parents],
                            "context_parent_families": [p["family_id"] for p in parents],
                            "parents": parents, "quota_pairs": q, "allowed_labels": sorted(q),
                            "template_sha256": sha_file(template_path), "split_manifest_sha256": split_sha,
                            "prompt": prompt, "prompt_sha256": sha_bytes(prompt.encode("utf-8"))})

    # 手写材料（格式示例、调试参考对）不能出现在官方训练集或 dev 里（短语级检查）
    phrases = {norm(x) for r in rows + dev for x in (r["text1"], r["text2"])}
    examples = [json.loads(l) for l in template.split("FORMAT EXAMPLES")[1].split("REFERENCE PAIRS")[0].splitlines() if l.startswith("{")]
    hand = json.load(open(os.path.join(HERE, "handwritten_reference_pairs.json"), encoding="utf-8"))
    hits = [x for e in examples + hand for x in (e["text1"], e["text2"]) if norm(x) in phrases]
    assert len(examples) == 3 and len(hand) == PILOT["parents_per_batch"] and not hits, hits
    dprompt = fill(template, hand, q, "debug_b04", "hand")
    debug = {"batch_id": "debug_b04", "package_version": VERSION, "stage": "debug", "signature": "", "usable_folds": [],
             "context_parent_ids": [], "context_parent_families": [], "parents": hand, "quota_pairs": q, "allowed_labels": sorted(q),
             "template_sha256": sha_file(template_path), "split_manifest_sha256": split_sha,
             "prompt": dprompt, "prompt_sha256": sha_bytes(dprompt.encode("utf-8"))}

    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    write_json(f"{out}/split_manifest.json", split_manifest)
    os.makedirs(f"{out}/inputs")
    shutil.copyfile(args.families, f"{out}/inputs/families.json")
    write_json(f"{out}/family_signatures.json", {"split_manifest_sha256": split_sha,
                                                  "signature_of_family": {f: list(s) for f, s in sorted(signature.items())}})
    for name, table in (("signature_table.csv", sig_table), ("demand_plus1.csv", demand)):
        with open(f"{out}/{name}", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    for b in batches: write_json(f"{out}/batches/{b['batch_id']}.json", b)
    write_json(f"{out}/debug/{debug['batch_id']}.json", debug)
    for name in ("prompt_template.txt", "handwritten_reference_pairs.json"):
        shutil.copyfile(os.path.join(HERE, name), f"{out}/{name}")
    files = sorted(os.path.relpath(os.path.join(d, x), out) for d, _, xs in os.walk(out) for x in xs if x != "MANIFEST.json")
    write_json(f"{out}/MANIFEST.json", {"version": VERSION, "builder_sha256": sha_file(__file__), "seed": SEED, "pilot": PILOT,
                                        "margin": MARGIN, "split_manifest_sha256": split_sha,
                                        "files": {p: sha_file(os.path.join(out, p)) for p in files}})
    tot = Counter()
    for b in batches: tot.update(b["quota_pairs"])
    print(f"{VERSION}：签名 {len(sig_fams)} 种；试点 {len(batches)} 批、{sum(len(b['parents']) for b in batches)} 个父样本；配额（对）{dict(tot)}")


if __name__ == "__main__":
    main()
