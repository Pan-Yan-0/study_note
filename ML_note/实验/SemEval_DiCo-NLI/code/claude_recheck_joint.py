"""Claude 对英语五折联合验证和第六组的独立复核：从逐折 logits / 概率重建预测和 OOF，核对家族划分、
保存的预测、OOF 拼接和官方评分，再算同种子配对差和按家族重抽样的近似区间（和 registry.py 同样的抽样约定）。
用法：python code/claude_recheck_joint.py <官方仓库目录>"""
import csv, json, math, os, random, subprocess, sys, tempfile
from collections import Counter, defaultdict

OFF = os.path.abspath(sys.argv[1])  # 官方仓库目录
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = f"{SE}/metrics/joint_validation"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]

rows = list(csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
ref = {r["instance_id"]: r for r in csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv", encoding="utf-8"))}
fam = json.load(open(f"{SE}/metrics/diagnostics/families.json"))
fold_of = {r["instance_id"]: int(fam["fold_of_family"][str(fam["family_of_pair"][r["pair_id"]])]) for r in rows}
family = {r["instance_id"]: fam["family_of_pair"][r["pair_id"]] for r in rows}
gold = {r["instance_id"]: r["label"] for r in rows}
byid = {r["instance_id"]: r for r in rows}
ids_all = [r["instance_id"] for r in rows]


def softmax(x):
    m = max(x); e = [math.exp(v - m) for v in x]; s = sum(e); return [v / s for v in e]


def read_pred(p):
    return {r["instance_id"]: r["label"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def preds_from_probs(prob):  # prob[(a,b)] -> 4 概率
    plain, sym = {}, {}
    for i in ids_all:
        r = byid[i]
        if (r["text1"], r["text2"]) not in prob:
            continue
        p = prob[(r["text1"], r["text2"])]
        q = prob[(r["text2"], r["text1"])]
        plain[i] = LABELS[max(range(4), key=lambda k: p[k])]
        s = [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]
        sym[i] = LABELS[max(range(4), key=lambda k: s[k])]
    return plain, sym


def official(pred):
    d = tempfile.mkdtemp()
    with open(f"{d}/g.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "pair_id", "reverse_pair_id", "label"])
        for i in ids_all:
            w.writerow([i, ref[i]["pair_id"], ref[i]["reverse_pair_id"], ref[i]["label"]])
    with open(f"{d}/p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"])
        for i in ids_all:
            w.writerow([i, pred[i]])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv",
                    "--output-dir", f"{d}/o"], cwd=OFF, check=True, capture_output=True)
    s = json.load(open(f"{d}/o/scores.json"))
    return s


def wf1(ids, pred):
    c = Counter(); tp = Counter(); fp = Counter(); fn = Counter()
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


def boot(pairs, B=2000, seed=20261021):  # pairs: [(predA, predB)]，每个种子一对；联合抽家族
    fams = sorted(set(family.values()))
    ids_of = defaultdict(list)
    for i in ids_all: ids_of[family[i]].append(i)
    rng = random.Random(seed); out = []
    for _ in range(B):
        draw = rng.choices(fams, k=len(fams)); ids = [i for f in draw for i in ids_of[f]]
        out.append(sum(wf1(ids, a) - wf1(ids, b) for a, b in pairs) / len(pairs))
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B) - 1]


def get(scores, key):
    for k in (key, key.replace("_", "")):
        if k in scores: return scores[k]
    return {k: v for k, v in scores.items() if not isinstance(v, dict)}


report = {}
problems = []
# ---------- mDeBERTa ----------
md = f"{J}/mdeberta/dico_mdeberta_results/track1"
OOF = {}
for method in sorted(os.listdir(md)):
    for seed in sorted(os.listdir(f"{md}/{method}")):
        plain_all, sym_all, seen = {}, {}, []
        for k in range(5):
            d = f"{md}/{method}/{seed}/fold{k}"
            lg = json.load(open(f"{d}/logits_test.json"))
            prob = {tuple(x): softmax(l) for x, l in zip(lg["inputs"], lg["logits"])}
            rids = [r["instance_id"] for r in csv.DictReader(open(f"{d}/reference_test.csv", encoding="utf-8"))]
            if any(fold_of[i] != k for i in rids) or sorted(rids) != sorted(i for i in ids_all if fold_of[i] == k):
                problems.append(f"{method}/{seed}/fold{k} 留出行和家族划分不符")
            pl, sy = preds_from_probs(prob)
            pl = {i: pl[i] for i in rids}; sy = {i: sy[i] for i in rids}
            if pl != read_pred(f"{d}/pred_test_plain.csv") or sy != read_pred(f"{d}/pred_test_sym.csv"):
                problems.append(f"{method}/{seed}/fold{k} 从 logits 重建的预测和保存的不同")
            plain_all.update(pl); sym_all.update(sy); seen += rids
        assert len(seen) == len(set(seen)) == 3042
        o = f"{J}/oof/mdeberta/{method}/{seed}"
        if read_pred(f"{o}/pred_sym.csv") != sym_all or read_pred(f"{o}/pred_plain.csv") != plain_all:
            problems.append(f"{method}/{seed} OOF 文件和逐折拼接不同")
        OOF[("md", method, seed)] = (plain_all, sym_all)

# ---------- Qwen ----------
for seed in ("seed42", "seed43", "seed44"):
    plain_all, sym_all = {}, {}
    for k in range(5):
        d = f"{J}/formal/qwen/fold{k}/dico_qwen_ce_results/track1/{seed}"
        prob = {}
        for line in open(f"{d}/cache.jsonl", encoding="utf-8"):
            x = json.loads(line); prob[(x["a"], x["b"])] = x["p"]
        nm = "holdout" if os.path.exists(f"{d}/reference_holdout.csv") else "fold0"
        rids = [r["instance_id"] for r in csv.DictReader(open(f"{d}/reference_{nm}.csv", encoding="utf-8"))]
        if sorted(rids) != sorted(i for i in ids_all if fold_of[i] == k):
            problems.append(f"qwen/{seed}/fold{k} 留出行和家族划分不符")
        pl, sy = preds_from_probs(prob)
        pl = {i: pl[i] for i in rids}; sy = {i: sy[i] for i in rids}
        if pl != read_pred(f"{d}/pred_{nm}_plain.csv") or sy != read_pred(f"{d}/pred_{nm}_sym.csv"):
            problems.append(f"qwen/{seed}/fold{k} 从概率重建的预测和保存的不同")
        plain_all.update(pl); sym_all.update(sy)
    o = f"{J}/oof/qwen/{seed}"
    if read_pred(f"{o}/pred_sym.csv") != sym_all or read_pred(f"{o}/pred_plain.csv") != plain_all:
        problems.append(f"qwen/{seed} OOF 文件和逐折拼接不同")
    OOF[("qwen", "ce", seed)] = (plain_all, sym_all)

# ---------- KD ----------
kd = f"{J}/formal/kd/dico_mdeberta_results/track1/kd"
for seed in sorted(os.listdir(kd)):
    plain_all, sym_all = {}, {}
    for k in range(5):
        d = f"{kd}/{seed}/fold{k}"
        lg = json.load(open(f"{d}/logits_test.json"))
        prob = {tuple(x): softmax(l) for x, l in zip(lg["inputs"], lg["logits"])}
        rids = [r["instance_id"] for r in csv.DictReader(open(f"{d}/reference_test.csv", encoding="utf-8"))]
        if sorted(rids) != sorted(i for i in ids_all if fold_of[i] == k):
            problems.append(f"kd/{seed}/fold{k} 留出行和家族划分不符")
        pl, sy = preds_from_probs(prob)
        pl = {i: pl[i] for i in rids}; sy = {i: sy[i] for i in rids}
        if pl != read_pred(f"{d}/pred_test_plain.csv") or sy != read_pred(f"{d}/pred_test_sym.csv"):
            problems.append(f"kd/{seed}/fold{k} 从 logits 重建的预测和保存的不同")
        plain_all.update(pl); sym_all.update(sy)
    o = f"{J}/oof/kd/{seed}"
    if read_pred(f"{o}/pred_sym.csv") != sym_all:
        problems.append(f"kd/{seed} OOF 文件和逐折拼接不同")
    OOF[("kd", "kd", seed)] = (plain_all, sym_all)

# ---------- 官方评分 ----------
table = defaultdict(list)
for key, (pl, sy) in sorted(OOF.items()):
    s_sym, s_pl = official(sy), official(pl)
    mine = wf1(ids_all, sy)
    table[(key[0], key[1])].append((key[2], s_pl, s_sym))
    report[f"{key}"] = {"plain": s_pl, "sym": s_sym, "my_wf1_sym": mine}
print("问题：", problems or "无")


def metric(s, name):
    flat = {}
    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict): walk(v, pre + k + ".")
            else: flat[pre + k] = v
    walk(s)
    return flat


fl = metric(report[str(next(iter(OOF)))]["sym"], "")
keys = [k for k in fl if any(t in k.lower() for t in ("weighted_f1", "soft", "hard"))]
print("指标键：", keys)
KW, KS, KH = [k for k in keys if "weighted" in k.lower()][0], [k for k in keys if "soft" in k.lower()][0], [k for k in keys if "hard" in k.lower()][0]
print("\n方法 | 种子 | plain WF1 | sym WF1 | sym Soft | sym Hard | plain Soft")
for (fam_, m), lst in sorted(table.items()):
    for subset in ([x for x in lst], [x for x in lst if x[0] in ("seed42", "seed43", "seed44")]) if (fam_ == "md" and m == "ce") else ([x for x in lst],):
        n = len(subset)
        mean = lambda which, k: sum(metric(x[which], "")[k] for x in subset) / n
        print(f"{fam_}/{m} | {n} | {100*mean(1,KW):.2f} | {100*mean(2,KW):.2f} | {100*mean(2,KS):.2f} | {100*mean(2,KH):.2f} | {100*mean(1,KS):.2f}")

def pair(a, b, seeds, which=1):
    ps = [(OOF[a + (s,)][which], OOF[b + (s,)][which]) for s in seeds]
    d = [100 * (wf1(ids_all, x) - wf1(ids_all, y)) for x, y in ps]
    lo, hi = boot(ps)
    return [round(x, 3) for x in d], round(sum(d) / len(d), 3), (round(100 * lo, 2), round(100 * hi, 2))

S5 = ["seed42", "seed43", "seed44", "seed45", "seed46"]; S3 = S5[:3]
for name, a, b, seeds in [("R-Drop − CE", ("md", "rdrop"), ("md", "ce"), S5), ("方向R-Drop − CE", ("md", "dir_rdrop"), ("md", "ce"), S5),
                          ("SCL − CE", ("md", "scl"), ("md", "ce"), S5), ("MoCo − CE", ("md", "moco_m0.999"), ("md", "ce"), S5),
                          ("KD − CE", ("kd", "kd"), ("md", "ce"), S3), ("Qwen − CE", ("qwen", "ce"), ("md", "ce"), S3)]:
    print(name, "sym:", pair(a, b, seeds), " plain:", pair(a, b, seeds, 0)[:2])


# ---------- 第六组：英语 fold 0 ----------
print()
QM = f"{SE}/metrics"
fold0 = [i for i in ids_all if fold_of[i] == 0]
P = {}
def load(dirp):
    prob = {}
    for line in open(f"{dirp}/cache.jsonl", encoding="utf-8"):
        x = json.loads(line); prob[(x["a"], x["b"])] = x["p"]
    pl, sy = preds_from_probs(prob)
    saved = read_pred(f"{dirp}/pred_fold0_sym.csv")
    assert all(sy[i] == saved[i] for i in saved) and sorted(saved) == sorted(fold0), dirp
    return {i: sy[i] for i in fold0}, {i: pl[i] for i in fold0}
for m in ("ce", "ce_drop", "rdrop", "scl", "moco"):
    for s in (42, 43, 44):
        P[(m, s)] = load(f"{QM}/qwen_methods_v2/{m}_results/track1/seed{s}")
for s in (42, 43, 44):
    P[("R3", s)] = load(f"{QM}/qwen_sft_a10_t1/track1/seed{s}") if os.path.exists(f"{QM}/qwen_sft_a10_t1/track1/seed{s}/cache.jsonl") else None
fams = sorted({family[i] for i in fold0}); ids_of = defaultdict(list)
for i in fold0: ids_of[family[i]].append(i)
def cmp(a, b, w=0):
    if P.get((a, 42)) is None or P[(b, 42)] is None: return None
    d = [100 * (wf1(fold0, P[(a, s)][w]) - wf1(fold0, P[(b, s)][w])) for s in (42, 43, 44)]
    rng = random.Random(20261021); bs = []
    for _ in range(2000):
        dr = rng.choices(fams, k=len(fams)); ids = [i for f in dr for i in ids_of[f]]
        bs.append(sum(100 * (wf1(ids, P[(a, s)][w]) - wf1(ids, P[(b, s)][w])) for s in (42, 43, 44)) / 3)
    bs.sort()
    return [round(x, 2) for x in d], round(sum(d) / 3, 2), (round(bs[50], 2), round(bs[1949], 2))
for m in ("ce", "ce_drop", "rdrop", "scl", "moco"):
    print(m, "fold0 sym", round(100 * sum(wf1(fold0, P[(m, s)][0]) for s in (42, 43, 44)) / 3, 2), " plain", round(100 * sum(wf1(fold0, P[(m, s)][1]) for s in (42, 43, 44)) / 3, 2))
for a, b in [("ce_drop", "ce"), ("rdrop", "ce_drop"), ("scl", "ce"), ("moco", "ce"), ("rdrop", "ce"), ("ce", "R3")]:
    print(f"{a} − {b}: sym", cmp(a, b), " plain", cmp(a, b, 1))
