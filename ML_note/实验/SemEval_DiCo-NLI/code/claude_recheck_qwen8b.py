"""Claude 对实验 A（Qwen3-8B 英语五折三种子，v5.5 队列）的独立复核：
从每个任务保存的四类概率（cache.jsonl）重建留出和 dev 的预测，核对家族划分、保存的逐折预测、OOF 拼接和官方评分；
再和 JOINT-QWEN（4B）按同种子配对比较，按家族重抽样（和 registry.py 同样的约定：2,000 次、种子 20261021、各种子共用同一批家族）。
用法：python code/claude_recheck_qwen8b.py <官方仓库目录>"""
import csv, json, os, random, subprocess, sys, tempfile
from collections import Counter, defaultdict

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = f"{SE}/metrics/qwen8b_a1"
J = f"{SE}/metrics/joint_validation"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
SEEDS = [42, 43, 44]

rows = list(csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
ref = {r["instance_id"]: r for r in csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv", encoding="utf-8"))}
dev_rows = list(csv.DictReader(open(f"{OFF}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv", encoding="utf-8")))
fam = json.load(open(f"{SE}/metrics/diagnostics/families.json"))
fold_of = {r["instance_id"]: int(fam["fold_of_family"][str(fam["family_of_pair"][r["pair_id"]])]) for r in rows}
family = {r["instance_id"]: fam["family_of_pair"][r["pair_id"]] for r in rows}
gold = {r["instance_id"]: r["label"] for r in rows}
byid = {r["instance_id"]: r for r in rows}
dev_byid = {r["instance_id"]: r for r in dev_rows}
ids_all = [r["instance_id"] for r in rows]
problems = []


def read_pred(p):
    return {r["instance_id"]: r["label"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def predict(prob, ids, table):
    plain, sym = {}, {}
    for i in ids:
        r = table[i]
        p, q = prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])]
        plain[i] = LABELS[max(range(4), key=lambda k: p[k])]
        s = [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]
        sym[i] = LABELS[max(range(4), key=lambda k: s[k])]
    return plain, sym


def official(pred, gold_csv):
    d = tempfile.mkdtemp()
    g = list(csv.DictReader(open(gold_csv, encoding="utf-8")))
    with open(f"{d}/g.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "pair_id", "reverse_pair_id", "label"])
        for r in g: w.writerow([r["instance_id"], r["pair_id"], r["reverse_pair_id"], r["label"]])
    with open(f"{d}/p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"])
        for r in g: w.writerow([r["instance_id"], pred[r["instance_id"]]])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv",
                    "--output-dir", f"{d}/o"], cwd=OFF, check=True, capture_output=True)
    return json.load(open(f"{d}/o/scores.json"))


def three(s):
    flat = {}
    def walk(x, pre=""):
        for k, v in x.items():
            if isinstance(v, dict): walk(v, pre + k + ".")
            else: flat[pre + k] = v
    walk(s)
    pick = lambda name: next(v for k, v in flat.items() if k.split(".")[-1].replace("_", "").lower() == name)
    return {"wf1": pick("weightedf1"), "soft_cons": pick("softcons"), "hard_cons": pick("hardcons")}


def wf1(ids, pred):
    c, tp, fp, fn = Counter(), Counter(), Counter(), Counter()
    for i in ids:
        g, p = gold[i], pred[i]; c[g] += 1
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    tot = sum(c.values()); out = 0.0
    for l in LABELS:
        if c[l] == 0: continue
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        rc = tp[l] / (tp[l] + fn[l])
        out += c[l] / tot * (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return out


def boot(pairs, B=2000, seed=20261021):
    fams = sorted(set(family.values()))
    ids_of = defaultdict(list)
    for i in ids_all: ids_of[family[i]].append(i)
    rng = random.Random(seed); out = []
    for _ in range(B):
        draw = rng.choices(fams, k=len(fams)); ids = [i for f in draw for i in ids_of[f]]
        out.append(sum(wf1(ids, a) - wf1(ids, b) for a, b in pairs) / len(pairs))
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B) - 1]


report = {"seeds": {}, "problems": problems}
oof = {}
for seed in SEEDS:
    plain_all, sym_all, epochs, dev_scores = {}, {}, [], []
    for k in range(5):
        d = f"{A}/formal/seed{seed}/fold{k}"
        task = json.load(open(f"{d}/task.json"))
        sp = task["split_ids"]
        if task["fold"] != k or task["seed"] != seed or task["smoke"]: problems.append(f"seed{seed}/fold{k} 任务身份不符")
        if sorted(sp["test"]) != sorted(i for i in ids_all if fold_of[i] == k): problems.append(f"seed{seed}/fold{k} 留出和家族划分不符")
        tr_fam = {family[i] for i in sp["fit"] + sp["val"]}
        if tr_fam & {family[i] for i in sp["test"]}: problems.append(f"seed{seed}/fold{k} 训练和留出共享家族")
        if set(sp["fit"]) & set(sp["val"]): problems.append(f"seed{seed}/fold{k} fit 和 val 重叠")
        prob = {}
        for line in open(f"{d}/cache.jsonl", encoding="utf-8"):
            x = json.loads(line); prob[(x["a"], x["b"])] = x["p"]
        pl, sy = predict(prob, sp["test"], byid)
        if sorted(read_pred(f"{d}/reference_fold0.csv")) != sorted(sp["test"]): problems.append(f"seed{seed}/fold{k} 留出参考文件和划分不符")  # 上游文件名固定写 fold0，内容是本折留出
        for mode, mine in (("plain", pl), ("sym", sy)):
            saved = read_pred(f"{d}/pred_fold0_{mode}.csv")
            if saved != mine: problems.append(f"seed{seed}/fold{k} {mode} 留出预测和概率重建不一致")
        plain_all.update(pl); sym_all.update(sy)
        dpl, dsy = predict(prob, list(dev_byid), dev_byid)
        for mode, mine in (("plain", dpl), ("sym", dsy)):
            if read_pred(f"{d}/pred_dev_{mode}.csv") != mine: problems.append(f"seed{seed}/fold{k} dev {mode} 预测不一致")
        dev_scores.append({m: three(official(x, f"{OFF}/final_data/dev/dico_nli_dev_track1_reference.csv")) for m, x in (("plain", dpl), ("sym", dsy))})
        epochs.append(json.load(open(f"{d}/trained.json"))["best_epoch"])
    for mode, mine in (("plain", plain_all), ("sym", sym_all)):
        if read_pred(f"{A}/oof/seed{seed}_{mode}.csv") != mine: problems.append(f"seed{seed} OOF {mode} 拼接不一致")
    oof[seed] = {"plain": plain_all, "sym": sym_all}
    sc = {m: three(official(oof[seed][m], f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv")) for m in ("plain", "sym")}
    for m in ("plain", "sym"):
        saved = three(json.load(open(f"{A}/oof/seed{seed}_{m}_scores/scores.json")))
        if any(abs(saved[x] - sc[m][x]) > 1e-12 for x in saved): problems.append(f"seed{seed} OOF {m} 分数和保存的不同")
    report["seeds"][seed] = {"oof": sc, "best_epoch_by_fold": epochs,
                             "dev_fold_mean": {m: {x: sum(s[m][x] for s in dev_scores) / 5 for x in ("wf1", "soft_cons", "hard_cons")} for m in ("plain", "sym")}}

# 和 JOINT-QWEN（4B，同划分、同种子、同配方，只换模型）配对
q4 = {s: {m: read_pred(f"{J}/oof/qwen/seed{s}/pred_{m}.csv") for m in ("plain", "sym")} for s in SEEDS}
cmp = {}
for m in ("plain", "sym"):
    d = [wf1(ids_all, oof[s][m]) - wf1(ids_all, q4[s][m]) for s in SEEDS]
    lo, hi = boot([(oof[s][m], q4[s][m]) for s in SEEDS])
    cmp[m] = {"diff_by_seed": d, "mean": sum(d) / 3, "ci95_family_bootstrap": [lo, hi],
              "q8b_mean": sum(wf1(ids_all, oof[s][m]) for s in SEEDS) / 3, "q4b_mean": sum(wf1(ids_all, q4[s][m]) for s in SEEDS) / 3}
report["cmp_8b_minus_4b_wf1"] = cmp
# 逐类 F1（对称化，三种子平均）
def per_class(pred):
    out = {}
    for l in LABELS:
        tp = sum(1 for i in ids_all if gold[i] == l and pred[i] == l)
        fp = sum(1 for i in ids_all if gold[i] != l and pred[i] == l)
        fn = sum(1 for i in ids_all if gold[i] == l and pred[i] != l)
        out[l] = 2 * tp / (2 * tp + fp + fn)
    return out
report["per_class_f1_sym_mean"] = {name: {l: sum(per_class(src[s]["sym"])[l] for s in SEEDS) / 3 for l in LABELS}
                                   for name, src in (("qwen3_8b", oof), ("qwen3_4b_joint", q4))}
json.dump(report, open(f"{A}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
print("问题：", problems or "无")
