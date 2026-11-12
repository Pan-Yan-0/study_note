"""方向二（检索对齐标注口径）的 CPU 预检，看过结果后才做，只作描述：字符 n-gram TF-IDF（text1、text2 分别编码后拼接，保留方向），
在同折 fit 里取 k 个近邻按金标加权投票，看它在全部行、174 条顽固错误、Qwen 4B 多数票判错的行上的准确率。
用法：python3 -I code/knn_signal_check.py <SemEval_DiCo-NLI 目录> <官方仓库目录>"""
import csv, json, sys
from collections import Counter
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
SE, OFF = sys.argv[1], sys.argv[2]
M = SE + "/metrics"
rows = list(csv.DictReader(open(OFF + "/final_data/train/dico_nli_train_track1_participant_labeled.csv")))
by = {r["instance_id"]: r for r in rows}
gold = {r["instance_id"]: r["label"] for r in rows}
split = json.load(open(M + "/augment_b/train_c5/input_data/split_manifest.json"))["folds"]
L = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
def rp(p): return {r["instance_id"]: r["label"] for r in csv.DictReader(open(p))}
oofs = [rp(f"{M}/augment_b/train_c2/oof/B2-Haiku_seed{s}_sym.csv") for s in (42, 43, 44)] + \
       [rp(f"{M}/joint_validation/oof/qwen/seed{s}/pred_sym.csv") for s in (42, 43, 44)] + \
       [rp(f"{M}/qwen8b_a1/oof/seed{s}_sym.csv") for s in (42, 43, 44)]
stub = {i for i in gold if all(o[i] != gold[i] for o in oofs)}
qwen = [rp(f"{M}/qwen4b_augb_s1/oof/Q0/seed{s}_sym.csv") for s in (42, 43, 44)]
qmaj = {i: Counter(q[i] for q in qwen).most_common(1)[0][0] for i in gold}
print("stubborn", len(stub))
res = {}
for K in (1, 4, 8):
    vote = {}
    for k in map(str, range(5)):
        fit, test = split[k]["fit"], split[k]["test"]
        v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit([by[i]["text1"] for i in fit] + [by[i]["text2"] for i in fit])
        X = lambda ids: hstack([v1.transform([by[i]["text1"] for i in ids]), v1.transform([by[i]["text2"] for i in ids])]).tocsr()
        Xf, Xt = X(fit), X(test)
        S = (Xt @ Xf.T).toarray() / 2
        for r, i in enumerate(test):
            nn = np.argsort(-S[r])[:K]
            c = Counter()
            for j in nn: c[gold[fit[j]]] += S[r, j]
            vote[i] = c.most_common(1)[0][0]
    acc = lambda ids: sum(vote[i] == gold[i] for i in ids) / len(ids)
    qerr = [i for i in gold if qmaj[i] != gold[i]]
    res[K] = {"all": acc(list(gold)), "stubborn": acc(sorted(stub)), "qwen_majority_wrong": acc(qerr), "n_qwen_wrong": len(qerr),
              "qwen_wrong_knn_right_and_agrees_with_nothing": None}
    print(K, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[K].items()})
print("majority-class baseline", Counter(gold.values()).most_common(1)[0][1] / len(gold))
print("stubborn gold dist", Counter(gold[i] for i in stub))
