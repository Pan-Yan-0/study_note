"""Claude 复核 Kaggle 上 mDeBERTa CE 在西语 / 巴语 / 跨语言的五折三种子结果：从逐折 logits 重建预测和折外拼接，
核对留出行和家族划分，官方评分复算，统计折间文本重合，按语言组合拆分，并和第五组 fold 0 比较。
用法：python3 code/claude_recheck_multilang.py metrics/mdeberta_multilang_oof <官方仓库目录>"""
import csv, glob, importlib.util, json, math, os, random, subprocess, sys, tempfile, types
from collections import Counter, defaultdict

R, OFF = sys.argv[1], sys.argv[2]
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fake = types.ModuleType("torch"); fake.cuda = types.SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False)
fake.__version__ = "fake"; fake.version = types.SimpleNamespace(cuda=None)
sys.modules["torch"] = fake
os.environ.update(DICO_DIR=OFF, DICO_STAGE="stage1", HF_ENDPOINT="x", DICO_WORK=tempfile.mkdtemp(), DICO_ALLOW_CPU="1")
spec = importlib.util.spec_from_file_location("mdb", f"{SE}/code/joint_validation/dico_mdeberta_methods.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
LABELS, SWAP = m.LABELS, m.SWAP
problems = []


def softmax(x):
    mx = max(x); e = [math.exp(v - mx) for v in x]; s = sum(e); return [v / s for v in e]


def read_pred(p):
    return {r["instance_id"]: r["label"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def official(rows, pred):
    d = tempfile.mkdtemp()
    with open(f"{d}/g.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(m.REFERENCE_FIELDS if hasattr(m, "REFERENCE_FIELDS") else ["instance_id", "pair_id", "reverse_pair_id", "label"])
        for r in rows: w.writerow([r["instance_id"], r["pair_id"], r["reverse_pair_id"], r["label"]])
    with open(f"{d}/p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"])
        for r in rows: w.writerow([r["instance_id"], pred[r["instance_id"]]])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv", "--output-dir", f"{d}/o"],
                   cwd=OFF, check=True, capture_output=True)
    s = json.load(open(f"{d}/o/scores.json"))
    return {"weighted_f1": s["weighted_f1"], "soft_cons": s["soft_cons"], "hard_cons": s["hard_cons"],
            "per_label_f1": {k: v.get("f1") for k, v in s["classification"]["per_label"].items()}}


def wf1(rows, pred):
    c, tp, fp, fn = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        g, p = r["label"], pred[r["instance_id"]]; c[g] += 1
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    n = sum(c.values()); out = 0.0
    for l in LABELS:
        if not c[l]: continue
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0; rc = tp[l] / c[l]
        out += c[l] / n * (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return out


fam_of, fold_of = m.split_base()
summary = json.load(open(f"{R}/COMPLETE.json"))["oof"]
OOF = {}
for track in (2, 3, 4):
    train = m.read_rows(m.data_file(track, "train", "reference"))
    for k in range(5):  # 同一折里留出文本和训练文本的重合（无序短语对）
        p = m.parts_for(track, k, False, smoke=False)
        fit_txt = {frozenset((r["text1"], r["text2"])) for r in p["fit"] + p["val"]}
        ov = sum(frozenset((r["text1"], r["text2"])) in fit_txt for r in p["test"])
        if ov: print(f"  track{track} fold{k}：留出 {len(p['test'])} 行里 {ov} 行的短语对文本也出现在训练 / 内层验证里")
    for seed in (42, 43, 44):
        plain, sym = {}, {}
        for k in range(5):
            d = glob.glob(f"{R}/formal/t{track}_s{seed}_f{k}/track{track}/ce/seed{seed}/fold{k}")[0]
            lg = json.load(open(f"{d}/logits_test.json"))
            prob = {tuple(x): softmax(l) for x, l in zip(lg["inputs"], lg["logits"])}
            if any(not all(math.isfinite(v) for v in l) for l in lg["logits"]): problems.append(f"{d} 有非有限 logits")
            test = m.parts_for(track, k, False, smoke=False)["test"]
            ref = [r["instance_id"] for r in csv.DictReader(open(f"{d}/reference_test.csv", encoding="utf-8"))]
            if ref != [r["instance_id"] for r in test]: problems.append(f"t{track} s{seed} f{k} 留出行和家族划分不同")
            pl, sy = {}, {}
            for r in test:
                a, b = prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])]
                pl[r["instance_id"]] = LABELS[max(range(4), key=lambda i: a[i])]
                s = [(a[i] + b[SWAP[i]]) / 2 for i in range(4)]
                sy[r["instance_id"]] = LABELS[max(range(4), key=lambda i: s[i])]
            if pl != read_pred(f"{d}/pred_test_plain.csv") or sy != read_pred(f"{d}/pred_test_sym.csv"):
                problems.append(f"t{track} s{seed} f{k} 从 logits 重建的预测和保存的不同")
            plain.update(pl); sym.update(sy)
        a = f"{R}/aggregate/track{track}/seed{seed}"
        if read_pred(f"{a}/pred_oof_sym.csv") != sym or read_pred(f"{a}/pred_oof_plain.csv") != plain or len(sym) != len(train):
            problems.append(f"t{track} s{seed} OOF 和逐折拼接不同")
        sp, ss = official(train, plain), official(train, sym)
        for which, s in (("plain", sp), ("sym", ss)):
            for key in ("weighted_f1", "soft_cons", "hard_cons"):
                if abs(s[key] - summary[f"track{track}/seed{seed}"]["scores"][which][key]) > 1e-9:
                    problems.append(f"t{track} s{seed} {which} {key} 复算 {s[key]} ≠ 记录")
        OOF[(track, seed)] = (train, plain, sym, sp, ss)

print("\n赛道 | plain WF1 | sym WF1 | sym Soft | sym Hard | plain Soft | sym 逐类 F1（E/F/B/O）")
for track in (2, 3, 4):
    xs = [OOF[(track, s)] for s in (42, 43, 44)]
    mean = lambda f: sum(f(x) for x in xs) / 3
    pl = {l: mean(lambda x: x[4]["per_label_f1"][l]) for l in LABELS}
    print(f"track{track} | {100*mean(lambda x: x[3]['weighted_f1']):.2f} | {100*mean(lambda x: x[4]['weighted_f1']):.2f} | "
          f"{100*mean(lambda x: x[4]['soft_cons']):.2f} | {100*mean(lambda x: x[4]['hard_cons']):.2f} | {100*mean(lambda x: x[3]['soft_cons']):.2f} | "
          + " / ".join(f"{100*pl[l]:.1f}" for l in LABELS) + "  逐种子 sym " + " / ".join(f"{100*x[4]['weighted_f1']:.2f}" for x in xs))

# 跨语言：按语言组合拆开
print("\n跨语言 sym WF1，按语言组合（三种子均值）")
by = defaultdict(list)
for r in OOF[(4, 42)][0]:
    by[r["instance_id"].split("__")[1]].append(r)
for combo, rows in sorted(by.items()):
    print(f"  {combo}: {len(rows)} 行  " + f"{100*sum(wf1(rows, OOF[(4, s)][2]) for s in (42, 43, 44))/3:.2f}")

# fold 0：和第五组第一阶段（A10 bf16）同划分、同种子比较
print("\nfold 0 sym WF1：Kaggle T4 fp16 本次 对 第五组 A10 bf16（同代码同划分同种子）")
for track in (2, 3, 4):
    out = []
    for seed in (42, 43, 44):
        old = glob.glob(f"{SE}/metrics/mdeberta_methods_stage1/**/track{track}/ce/seed{seed}/fold0/pred_test_sym.csv", recursive=True)
        new = read_pred(glob.glob(f"{R}/formal/t{track}_s{seed}_f0/track{track}/ce/seed{seed}/fold0/pred_test_sym.csv")[0])
        test = m.parts_for(track, 0, False, smoke=False)["test"]
        if old:
            o = read_pred(old[0]); agree = sum(o[i] == new[i] for i in new) / len(new)
            out.append(f"seed{seed}: {100*wf1(test, new):.2f} 对 {100*wf1(test, o):.2f}（逐条一致 {agree:.1%}）")
        else:
            out.append(f"seed{seed}: {100*wf1(test, new):.2f} 对 （没找到第五组输出）")
    print(f"  track{track}: " + "；".join(out))
print("\n问题：", problems or "无")
