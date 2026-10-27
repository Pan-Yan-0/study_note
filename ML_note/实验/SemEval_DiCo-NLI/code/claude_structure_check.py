"""Claude 的检查：只按测试文件里看得到的配对结构（同一个 instance 前缀有没有反向那一条）把单条的定为 NEGATIVE_OTHER、
成对的只在 E / F / B 里选，dev 上的三个分数会变成多少。用已保存的 fold 0 模型在 dev 上的概率，不重新训练。
用法：python3 code/claude_structure_check.py（官方仓库和 metrics 的路径写在开头）"""
import csv, json, math, subprocess, sys, tempfile, glob
import os
OFF = os.environ.get("DICO_DIR", "SemEval-2027-Task-2-DiCo-NLI"); M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics")
L = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]; SW = [0, 2, 1, 3]
def ref(t): return list(csv.DictReader(open(f"{OFF}/final_data/dev/dico_nli_dev_track{t}_reference.csv", encoding="utf-8")))
def part(t): return {r["instance_id"]: r for r in csv.DictReader(open(f"{OFF}/final_data/dev/dico_nli_dev_track{t}_participant_labeled.csv", encoding="utf-8"))}
def score(t, pred):
    d = tempfile.mkdtemp()
    rs = ref(t)
    with open(f"{d}/g.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance_id", "pair_id", "reverse_pair_id", "label"]); [w.writerow([r["instance_id"], r["pair_id"], r["reverse_pair_id"], r["label"]]) for r in rs]
    with open(f"{d}/p.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"]); [w.writerow([r["instance_id"], pred[r["instance_id"]]]) for r in rs]
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv", "--output-dir", f"{d}/o"], cwd=OFF, check=True, capture_output=True)
    s = json.load(open(f"{d}/o/scores.json")); return round(s["weighted_f1"], 4), round(s["soft_cons"], 4), round(s["hard_cons"], 4)
def preds(t, prob, structure):
    P = part(t); groups = {}
    for i in P: groups.setdefault(i.rsplit("__", 1)[0], []).append(i)  # 只用提交模板里也看得到的 instance_id 前缀分组
    out = {}
    for i, r in P.items():
        p = prob[(r["text1"], r["text2"])]; q = prob[(r["text2"], r["text1"])]
        s = [(p[k] + q[SW[k]]) / 2 for k in range(4)]
        if structure:
            if len(groups[i.rsplit("__", 1)[0]]) == 1: out[i] = "NEGATIVE_OTHER"; continue
            out[i] = L[max(range(3), key=lambda k: s[k])]
        else:
            out[i] = L[max(range(4), key=lambda k: s[k])]
    return out
def sm(x): m = max(x); e = [math.exp(v - m) for v in x]; z = sum(e); return [v / z for v in e]
def load_cache(path):
    d = {}
    for l in open(path):
        x = json.loads(l); d[(x["a"], x["b"])] = x["p"]
    return d
def load_logits(path):
    lg = json.load(open(path)); return {tuple(x): sm(l) for x, l in zip(lg["inputs"], lg["logits"])}
cfg = {"Qwen SFT": {1: [f"{M}/qwen_sft_a10_t1/track1/seed{s}/cache.jsonl" for s in (42,43,44)],
                    2: [f"{M}/qwen_sft/track2/seed{s}/cache.jsonl" for s in (42,43,44)],
                    3: [f"{M}/qwen_sft/track3/seed{s}/cache.jsonl" for s in (42,43,44)],
                    4: [f"{M}/qwen_sft_a10/track4/seed{s}/cache.jsonl" for s in (42,43,44)]},
       "mDeBERTa CE": {1: [f"{M}/joint_validation/mdeberta/dico_mdeberta_results/track1/ce/seed{s}/fold0/logits_dev.json" for s in (42,43,44)],
                       **{t: [glob.glob(f"{M}/mdeberta_multilang_oof/formal/t{t}_s{s}_f0/track{t}/ce/seed{s}/fold0/logits_dev.json")[0] for s in (42,43,44)] for t in (2,3,4)}}}
for name, tracks in cfg.items():
    for t, paths in tracks.items():
        res = {False: [], True: []}
        for p in paths:
            prob = load_cache(p) if p.endswith(".jsonl") else load_logits(p)
            for st in (False, True): res[st].append(score(t, preds(t, prob, st)))
        avg = lambda xs: tuple(round(sum(x[k] for x in xs) / len(xs), 4) for k in range(3))
        print(f"{name} track{t}: 只用文本 {avg(res[False])}  用 pair 结构定 O {avg(res[True])}")
