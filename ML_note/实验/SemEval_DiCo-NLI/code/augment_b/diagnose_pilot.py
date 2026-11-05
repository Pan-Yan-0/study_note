"""试点的描述性诊断（只报告，不过滤、不改停止条件）。读 process_outputs.py build 的输出目录。

  python diagnose_pilot.py <build 输出目录> <官方仓库目录> <输出 json>

报告：
  1. 每个生成标签的短语长度、以 a / an / the 开头的比例，和原数据同口径对比；
  2. 探针的抽样噪声：从原数据按生成数据的类别计数抽同样多的对，重复 100 次的平衡准确率分布；
  3. 探针按特征分组（长度 / 词重叠 / 开头词和数字）分别跑，看差距来自哪一组；
  4. 事先定的两条口径：只差数量的对（label_conventions.number_pair，按生成标签），
     单向对里 text2 的词是 text1 子集的比例及其中只删一个词的比例。
"""
import json, os, random, statistics as st, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from process_outputs import canonical_original, features, norm, probe, read_csv, REVIEW_SEED  # noqa: E402
from label_conventions import number_pair  # noqa: E402

GROUPS = {"length": [0, 1, 2, 3, 7], "overlap": [4, 5, 6], "first_word_digit_article": [8, 9, 10]}
ARTICLES = ("a", "an", "the")


def W(s): return len(norm(s).split())


def probe_subset(pairs, idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = [[features(a, b)[i] for i in idx] for a, b, _ in pairs]; y = [l for *_, l in pairs]
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=REVIEW_SEED))
    return balanced_accuracy_score(y, pred)


def shape(pairs):
    out = {}
    for l in sorted({x[2] for x in pairs}):
        P = [(a, b) for a, b, ll in pairs if ll == l]
        out[l] = {"pairs": len(P), "mean_words_text1": round(st.mean(W(a) for a, _ in P), 2),
                  "mean_words_text2": round(st.mean(W(b) for _, b in P), 2),
                  "mean_len_diff": round(st.mean(W(a) - W(b) for a, b in P), 2),
                  "article_start_share": round(st.mean(norm(t).split()[0] in ARTICLES for p in P for t in p), 3)}
    return out


def main(out_dir, off, dest):
    gen = [(r["text1"], r["text2"], r["label"]) for r in read_csv(f"{out_dir}/filter_decisions.csv") if not r["removed"]]
    orig = canonical_original(read_csv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv"))
    cnt = Counter(l for *_, l in gen); rng = random.Random(1)
    by = {l: [p for p in orig if p[2] == l] for l in cnt}
    noise = sorted(probe([q for l, n in cnt.items() for q in rng.sample(by[l], n)], seed=k) for k in range(100))
    F = [(a, b) for a, b, l in gen if l == "FORWARD_ENTAILMENT"]
    sub = [(a, b) for a, b in F if set(norm(b).split()) <= set(norm(a).split())]
    one = [(a, b) for a, b in sub if W(a) - W(b) == 1]
    Fo = [(a, b) for a, b, l in orig if l == "FORWARD_ENTAILMENT"]
    res = {
        "kept_pairs": len(gen),
        "shape_generated": shape(gen), "shape_original": shape(orig),
        "probe_generated": round(probe(gen), 4), "probe_original_full": round(probe(orig), 4),
        "probe_original_subsample_n_equal": {"draws": 100, "mean": round(st.mean(noise), 4),
                                             "p95": round(noise[94], 4), "max": round(noise[-1], 4)},
        "probe_by_feature_group": {g: {"generated": round(probe_subset(gen, i), 4), "original": round(probe_subset(orig, i), 4)}
                                   for g, i in GROUPS.items()},
        "number_pair_by_label": dict(Counter(l for a, b, l in gen if number_pair(a, b))),
        "forward_text2_subset_of_text1": {"subset": len(sub), "of": len(F), "one_word_deleted": len(one),
                                          "original_subset_share": round(sum(set(norm(b).split()) <= set(norm(a).split()) for a, b in Fo) / len(Fo), 3)},
    }
    with open(dest, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*map(os.path.abspath, sys.argv[1:4]))
