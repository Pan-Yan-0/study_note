"""实验 B 全量 +1 的批次包（方案 B，Pan-Yan-0 11-06 定的）：主轮每个合格父家族恰好仿写一次。

  python build_full.py main <官方仓库目录> <v6 试点包目录> <输出目录>
  python build_full.py supp <官方仓库目录> <v6 试点包目录> <补充计划.json> <输出目录>

- 家族、签名、父样本的取法和 v6 试点完全相同（build_pilot.py 的 parent_of：家族里字典序最小的 pair_id 的原方向，
  BACKWARD 交换成 FORWARD 展示；只差一个数量的父样本不用）。划分清单、家族表从试点包读，并核对试点包的 MANIFEST。
- 主轮：试点用过的 210 个家族不再用；其余合格家族各用一次。按签名分包，签名内按固定种子打乱后按标签分池，
  先尽量按 E 3 / 单向 7 / O 4 分层装满 14 个一批，分不出分层批以后，把剩下的父样本按打乱顺序 14 个一批装（最后一批可以不满）。
  一批只放同一签名，所以批的可用折 = 签名。
- 补充轮：读 dryrun_full.py / plan_supplement 给出的补充计划（supplement.plan 的输出），每批只有一个组。
- 提示词、过滤、停止条件和 v6 相同（提示词模板从试点包复制，哈希必须一致）。
"""
import csv, hashlib, json, os, random, shutil, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from label_conventions import number_pair  # noqa: E402
from build_pilot import fill, norm, sha_bytes, sha_file, canon, write_json  # noqa: E402

VERSION = "augb-full-v1"
SEED = 20261106
PER_BATCH = 14
STRATA = {"EQUIVALENCE": 3, "FORWARD_ENTAILMENT": 7, "NEGATIVE_OTHER": 4}
GROUP = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "U", "BACKWARD_ENTAILMENT": "U", "NEGATIVE_OTHER": "O"}
LABEL_OF = {"E": "EQUIVALENCE", "U": "FORWARD_ENTAILMENT", "O": "NEGATIVE_OTHER"}
ROWS_PER_PAIR = {"EQUIVALENCE": 2, "FORWARD_ENTAILMENT": 2, "NEGATIVE_OTHER": 1}


def jload(p):
    with open(p, encoding="utf-8") as f: return json.load(f)


def rcsv(p):
    with open(p, encoding="utf-8") as f: return list(csv.DictReader(f))


def load_world(off, pilot_pkg):
    """从试点包读划分和家族表（核对哈希），返回和 build_pilot 相同口径的家族、签名、父样本。"""
    man = jload(f"{pilot_pkg}/MANIFEST.json")
    for rel, h in man["files"].items():
        if sha_file(f"{pilot_pkg}/{rel}") != h: raise SystemExit(f"试点包文件变了：{rel}")
    split_manifest = jload(f"{pilot_pkg}/split_manifest.json")
    split_sha = sha_bytes(canon(split_manifest).encode())
    assert split_sha == man["split_manifest_sha256"]
    split = split_manifest["folds"]
    fam = jload(f"{pilot_pkg}/inputs/families.json")
    rows = rcsv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    dev = rcsv(f"{off}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    byid = {r["instance_id"]: r for r in rows}
    family = {r["instance_id"]: str(fam["family_of_pair"][r["pair_id"]]) for r in rows}
    fam_rows = defaultdict(list)
    for r in rows: fam_rows[family[r["instance_id"]]].append(r["instance_id"])
    part_of = {(k, i): p for k, s in split.items() for p, ids in s.items() for i in ids}
    signature = {}
    for f, ids in fam_rows.items():
        folds = []
        for k in split:
            parts = {part_of[(k, i)] for i in ids}
            assert len(parts) == 1
            if parts == {"fit"}: folds.append(int(k))
        signature[f] = tuple(folds)
    saved = jload(f"{pilot_pkg}/family_signatures.json")["signature_of_family"]
    assert {f: list(s) for f, s in signature.items()} == saved, "签名和试点包不一致"

    def parent_of(f):
        pid = sorted({byid[i]["pair_id"] for i in fam_rows[f]})[0]
        r = byid[f"{pid}__en-en__original"]
        t1, t2, lab = r["text1"], r["text2"], r["label"]
        if lab == "BACKWARD_ENTAILMENT": t1, t2, lab = t2, t1, "FORWARD_ENTAILMENT"
        return {"sample_id": r["instance_id"], "pair_id": pid, "family_id": f, "usable_folds": list(signature[f]),
                "text1": t1, "text2": t2, "label": lab, "shown_swapped": r["label"] == "BACKWARD_ENTAILMENT"}

    pilot_fams = set()
    for x in sorted(os.listdir(f"{pilot_pkg}/batches")):
        pilot_fams |= {p["family_id"] for p in jload(f"{pilot_pkg}/batches/{x}")["parents"]}
    eligible = {}
    for f in fam_rows:
        if not signature[f]: continue
        p = parent_of(f)
        if number_pair(p["text1"], p["text2"]): continue
        eligible[f] = p
    return {"rows": rows, "dev": dev, "byid": byid, "fam_rows": fam_rows, "signature": signature, "split": split,
            "split_manifest": split_manifest, "split_sha": split_sha, "pilot_manifest": man, "eligible": eligible,
            "pilot_families": pilot_fams}


def sig_str(s): return "".join(map(str, s))


def plan_main(world):
    """主轮分包：返回 [(签名, [父样本, ...]), ...]，每个合格且试点没用过的家族恰好一次。"""
    rng = random.Random(SEED)
    by_sig = defaultdict(list)
    for f, p in world["eligible"].items():
        if f not in world["pilot_families"]: by_sig[world["signature"][f]].append(f)
    out = []
    for s in sorted(by_sig, key=lambda s: (-len(by_sig[s]), s)):
        fs = sorted(by_sig[s]); rng.shuffle(fs)
        pool = defaultdict(list)
        for f in fs: pool[world["eligible"][f]["label"]].append(world["eligible"][f])
        while all(len(pool[l]) >= k for l, k in STRATA.items()):
            refs = [pool[l].pop(0) for l, k in STRATA.items() for _ in range(k)]
            rng.shuffle(refs); out.append((s, refs))
        left = {p["family_id"] for l in pool for p in pool[l]}
        rest = [world["eligible"][f] for f in fs if f in left]  # 分层批装不下的，按打乱顺序装
        for i in range(0, len(rest), PER_BATCH):
            refs = rest[i:i + PER_BATCH]; rng.shuffle(refs); out.append((s, refs))
    return out


def make_batch(bid, stage, s, refs, template, template_path, split_sha):
    prompt = fill(template, refs, bid, "real")
    q = dict(Counter(p["label"] for p in refs))
    return {"batch_id": bid, "package_version": VERSION, "stage": stage, "signature": sig_str(s),
            "usable_folds": sorted(set.intersection(*[set(p["usable_folds"]) for p in refs])),
            "context_parent_ids": [p["sample_id"] for p in refs], "context_parent_families": [p["family_id"] for p in refs],
            "parents": refs, "ref_labels": [p["label"] for p in refs], "quota_pairs": q, "allowed_labels": sorted(q),
            "template_sha256": sha_file(template_path), "split_manifest_sha256": split_sha,
            "prompt": prompt, "prompt_sha256": sha_bytes(prompt.encode("utf-8"))}


def write_package(out, batches, world, pilot_pkg, extra):
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    write_json(f"{out}/split_manifest.json", world["split_manifest"])
    os.makedirs(f"{out}/inputs"); shutil.copyfile(f"{pilot_pkg}/inputs/families.json", f"{out}/inputs/families.json")
    shutil.copyfile(f"{pilot_pkg}/family_signatures.json", f"{out}/family_signatures.json")
    for b in batches: write_json(f"{out}/batches/{b['batch_id']}.json", b)
    for name in ("prompt_template.txt", "handwritten_reference_pairs.json"):
        shutil.copyfile(f"{pilot_pkg}/{name}", f"{out}/{name}")
    table = [{"batch_id": b["batch_id"], "signature": b["signature"], "n_parents": len(b["parents"]),
              "E": b["quota_pairs"].get("EQUIVALENCE", 0), "U": b["quota_pairs"].get("FORWARD_ENTAILMENT", 0),
              "O": b["quota_pairs"].get("NEGATIVE_OTHER", 0)} for b in batches]
    with open(f"{out}/batch_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    files = sorted(os.path.relpath(os.path.join(d, x), out) for d, _, xs in os.walk(out) for x in xs if x != "MANIFEST.json")
    write_json(f"{out}/MANIFEST.json", {"version": VERSION, "builder_sha256": sha_file(__file__), "seed": SEED,
                                        "split_manifest_sha256": world["split_sha"],
                                        "pilot_package_manifest_sha256": sha_file(f"{pilot_pkg}/MANIFEST.json"),
                                        "files": {p: sha_file(os.path.join(out, p)) for p in files}, **extra})


def main():
    cmd = sys.argv[1]
    off, pilot_pkg = os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    world = load_world(off, pilot_pkg)
    template_path = f"{pilot_pkg}/prompt_template.txt"
    with open(template_path, encoding="utf-8") as f: template = f.read()
    if cmd == "main":
        out = os.path.abspath(sys.argv[4])
        plan = plan_main(world)
        batches = [make_batch(f"main_b{i:03d}", "main", s, refs, template, template_path, world["split_sha"])
                   for i, (s, refs) in enumerate(plan, 1)]
        used = [p["family_id"] for b in batches for p in b["parents"]]
        assert len(used) == len(set(used)) and not set(used) & world["pilot_families"]
        write_package(out, batches, world, pilot_pkg, {"stage": "main"})
        c = Counter(p["label"] for b in batches for p in b["parents"])
        print(f"{VERSION} 主轮：{len(batches)} 批、{len(used)} 个父家族（每个一次），{dict(c)}；"
              f"分层满批 {sum(1 for b in batches if b['quota_pairs'] == STRATA)} 批")
    elif cmd == "supp":
        plan_path, out = os.path.abspath(sys.argv[4]), os.path.abspath(sys.argv[5])
        sp = jload(plan_path)
        assert sp["status"] == "ok", sp["status"]
        batches = []
        for x in sp["batches"]:
            refs = [world["eligible"][f] for f in x["parent_families"]]
            assert all(GROUP[p["label"]] == x["group"] for p in refs)
            s = tuple(int(c) for c in x["signature"])
            batches.append(make_batch(x["batch_id"], f"supp{sp['round']}", s, refs, template, template_path, world["split_sha"]))
        write_package(out, batches, world, pilot_pkg, {"stage": f"supp{sp['round']}", "supplement_plan_sha256": sha_file(plan_path)})
        print(f"{VERSION} 补充第 {sp['round']} 轮：{len(batches)} 批")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
