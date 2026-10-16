"""复算 SST2/README.md 里的统计数字：准确率区间、召回、配对检验、DeBERTa 均值与标准差。

    python sst2_report_stats.py

只读 ../metrics/ 下已经导出的逐条结果，不调用模型。用到的只有 Python 标准库。
"""
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

M = Path(__file__).resolve().parent.parent / "metrics"
ALL, FIRST200 = range(872), range(200)


def load_eval_csv(path: Path) -> dict[int, dict]:
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[int(r["idx"])] = {"label": int(r["label"]), "pred": int(r["pred"]) if r["pred"] else None,
                                   "correct": r["correct"] == "1", "finish": r["finish"],
                                   "tokens": int(r["completion_tokens"] or 0), "seconds": float(r["seconds"])}
    return rows


def load_deberta(name: str) -> dict[int, dict]:
    rows = {}
    with open(M / "deberta_sst2/runs" / name / "predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label, pred = (1 if r["label"] == "1" else -1), (1 if r["pred"] == "1" else -1)
            rows[int(r["idx"])] = {"label": label, "pred": pred, "correct": label == pred,
                                   "prob": float(r["prob_positive"])}
    return rows


def one(pattern: str, folder: Path = M) -> dict[int, dict]:
    found = [p for p in folder.glob(pattern + ".csv") if not p.name.endswith("_errors.csv")]
    assert len(found) == 1, (pattern, found)
    return load_eval_csv(found[0])


def acc(rows, idx) -> float:
    idx = list(idx)
    return sum(rows[i]["correct"] for i in idx) / len(idx)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def mcnemar(a, b, idx) -> tuple[int, int, float]:
    """精确 McNemar：只看两边意见不一致的条目，双侧二项检验。"""
    only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in idx)
    only_b = sum(b[i]["correct"] and not a[i]["correct"] for i in idx)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def compare(name, a, b, idx) -> None:
    idx = list(idx)
    only_a, only_b, p = mcnemar(a, b, idx)
    print(f"  {name}（{len(idx)} 条）：{acc(a, idx):.2%} 对 {acc(b, idx):.2%}；只有前者对 {only_a}，"
          f"只有后者对 {only_b}，p = {p:.3g}")


def auc(rows) -> float:
    pos = [r["prob"] for r in rows.values() if r["label"] == 1]
    neg = [r["prob"] for r in rows.values() if r["label"] == -1]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def describe(name, rows) -> None:
    n = len(rows)
    k = sum(r["correct"] for r in rows.values())
    lo, hi = wilson(k, n)
    labels = Counter(r["label"] for r in rows.values())
    rec_neg = sum(r["pred"] == -1 for r in rows.values() if r["label"] == -1) / labels[-1]
    rec_pos = sum(r["pred"] == 1 for r in rows.values() if r["label"] == 1) / labels[1]
    invalid = sum(r["pred"] is None for r in rows.values())
    cut = sum(r["finish"] == "length" for r in rows.values())
    head = f"{acc(rows, FIRST200):.2%}" if n >= 200 else "—"
    print(f"  {name:<26} {k:>3}/{n:<3} {k / n:.2%} [{lo:.2%}, {hi:.2%}]  负面召回 {rec_neg:.2%}  "
          f"正面召回 {rec_pos:.2%}  无效 {invalid:>2}  截断 {cut:>2}  前200条 {head}  "
          f"输出token中位数 {statistics.median(r['tokens'] for r in rows.values()):.0f}  "
          f"总用时 {sum(r['seconds'] for r in rows.values()) / 60:.1f} 分")


def main() -> None:
    q = {}
    for size in ("0.8b", "2b", "4b", "9b"):
        for prompt in ("basic", "cot", "role_cot"):
            for think in ("off", "on"):
                found = [p for p in (M / "qwen_matrix").glob(f"ollama_qwen3.5-{size}_{prompt}_think-{think}_*.csv")
                         if not p.name.endswith("_errors.csv")]
                if found:
                    q[(size, prompt, think)] = load_eval_csv(found[0])
    ds = {key: one(f"deepseek_deepseek-flash_{key[0]}_think-{key[1]}_all_{key[2]}*")
          for key in [("basic", "off", "a4d7"), ("basic", "on", "7944"), ("cot", "off", "8b10"),
                      ("cot", "on", "cb39"), ("cot", "on", "b774"), ("role_cot", "off", "a8fa"),
                      ("role_cot", "on", "45d3")]}
    for rows in q.values():  # 没跑完的组：断点按顺序写入，完成的是开头连续的若干条
        assert sorted(rows) == list(range(len(rows)))

    print("Qwen3.5（Ollama，重新判分后）")
    for key, rows in q.items():
        describe("/".join(key), rows)
    print("DeepSeek（重新判分后）")
    for key, rows in ds.items():
        describe("/".join(key), rows)

    print("模型尺寸（basic，关思考，872 条）")
    compare("0.8b 对 2b", q[("0.8b", "basic", "off")], q[("2b", "basic", "off")], ALL)
    compare("2b 对 4b", q[("2b", "basic", "off")], q[("4b", "basic", "off")], ALL)
    compare("4b 对 9b", q[("4b", "basic", "off")], q[("9b", "basic", "off")], ALL)
    sizes = ("0.8b", "2b", "4b", "9b")
    print("各尺寸的模板（都关思考，前 200 条）")
    for s in sizes:
        b, c, r = q[(s, "basic", "off")], q[(s, "cot", "off")], q[(s, "role_cot", "off")]
        compare(f"{s} basic 对 cot", b, c, FIRST200)
        compare(f"{s} basic 对 role_cot", b, r, FIRST200)
        compare(f"{s} cot 对 role_cot", c, r, FIRST200)
    print("cot / role_cot 关思考时直接只答一个数字（输出不超过 5 个 token）的条数")
    for s in sizes:
        print(f"  {s}：" + "，".join(
            f"{p} {sum(q[(s, p, 'off')][i]['tokens'] <= 5 for i in FIRST200)}/200" for p in ("cot", "role_cot")))
    print("思考开关：同一模板关思考对开思考。先比全部条目（截断算错），再只比开思考时没被截断的条目")
    for s in sizes:
        for p in ("basic", "cot", "role_cot"):
            off, on = q[(s, p, "off")], q[(s, p, "on")]
            done = [i for i in FIRST200 if i in on]  # 只比开思考跑到的条目（现在两边都是 200 条）
            kept = [i for i in done if on[i]["finish"] != "length"]
            cut = [i for i in done if on[i]["finish"] == "length"]
            compare(f"{s} {p} 全部", off, on, done)
            compare(f"{s} {p} 没截断的", off, on, kept)
            print(f"    截断 {len(cut)}/{len(done)} 条（{len(cut) / len(done):.0%}）；这些条目关思考时的准确率 "
                  f"{acc(off, cut) if cut else float('nan'):.1%}；开思考输出 token 中位数 "
                  f"{statistics.median(on[i]['tokens'] for i in done):.0f}")
    print("DeepSeek 两两比较（872 条）")
    dsb = ds[("basic", "off", "a4d7")]
    compare("basic 关 对 basic 开", dsb, ds[("basic", "on", "7944")], ALL)
    compare("cot 关 对 cot 开（8192）", ds[("cot", "off", "8b10")], ds[("cot", "on", "cb39")], ALL)
    compare("role_cot 关 对 role_cot 开", ds[("role_cot", "off", "a8fa")], ds[("role_cot", "on", "45d3")], ALL)
    compare("basic 关 对 cot 关", dsb, ds[("cot", "off", "8b10")], ALL)
    compare("basic 关 对 role_cot 关", dsb, ds[("role_cot", "off", "a8fa")], ALL)
    compare("role_cot 关 对 cot 关", ds[("role_cot", "off", "a8fa")], ds[("cot", "off", "8b10")], ALL)
    compare("cot 开两次运行（8192 对 4096）", ds[("cot", "on", "cb39")], ds[("cot", "on", "b774")], ALL)
    compare("basic 关 对 cot 开（4096）", dsb, ds[("cot", "on", "b774")], ALL)
    keys = list(ds)
    p_small = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))
               if mcnemar(ds[keys[i]], ds[keys[j]], ALL)[2] < 0.05]
    print(f"  {len(keys) * (len(keys) - 1) // 2} 对两两比较中 p < 0.05 的：{p_small}")
    six = [ds[k] for k in keys if k[2] != "b774"]
    wrong = [i for i in ALL if all(not s[i]["correct"] for s in six)]
    print(f"  6 种设置全错 {len(wrong)} 条，标签 {dict(Counter(dsb[i]['label'] for i in wrong))}")
    print("DeepSeek 对 Qwen")
    compare("Qwen 4b 对 DeepSeek（basic 关思考）", q[("4b", "basic", "off")], dsb, ALL)
    compare("Qwen 9b 对 DeepSeek（basic 关思考）", q[("9b", "basic", "off")], dsb, ALL)
    compare("Qwen 4b 对 DeepSeek（cot 关思考）", q[("4b", "cot", "off")], ds[("cot", "off", "8b10")], FIRST200)

    print("DeBERTa-v3-base（872 条，3 个种子）")
    deb = {p.name: load_deberta(p.name) for p in (M / "deberta_sst2/runs").iterdir()}
    assert all(deb["nfull_seed13"][i]["label"] == dsb[i]["label"] for i in ALL)
    for size in ("16", "64", "256", "1024", "full"):
        accs = [acc(deb[f"n{size}_seed{s}"], ALL) for s in (13, 21, 42)]
        aucs = [auc(deb[f"n{size}_seed{s}"]) for s in (13, 21, 42)]
        preds_pos = [sum(r["pred"] == 1 for r in deb[f"n{size}_seed{s}"].values()) for s in (13, 21, 42)]
        print(f"  {size:>5} 条：{' / '.join(f'{a:.2%}' for a in accs)}，均值 {statistics.mean(accs):.2%} ± "
              f"{statistics.stdev(accs):.2%}（样本标准差）；AUC 均值 {statistics.mean(aucs):.4f}；"
              f"预测为正面的条数 {preds_pos}")
    for s in (13, 21, 42):
        compare(f"DeBERTa 全量 种子{s} 对 DeepSeek basic 关", deb[f"nfull_seed{s}"], dsb, ALL)
        compare(f"DeBERTa 全量 种子{s} 对 Qwen 4b basic 关", deb[f"nfull_seed{s}"], q[("4b", "basic", "off")], ALL)
        compare(f"DeBERTa 1024 条 种子{s} 对 Qwen 4b basic 关", deb[f"n1024_seed{s}"], q[("4b", "basic", "off")], ALL)
    four = [dsb, q[("4b", "basic", "off")], q[("9b", "basic", "off")], deb["nfull_seed13"]]
    wrong = [i for i in ALL if all(not s[i]["correct"] for s in four)]
    print(f"  DeepSeek、Qwen 4b、Qwen 9b（basic 关思考）与 DeBERTa 全量种子13 都错：{len(wrong)} 条，"
          f"标签 {dict(Counter(dsb[i]['label'] for i in wrong))}，idx {wrong}")
    first = one("ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd")
    same = sum(first[i]["pred"] == q[("4b", "basic", "off")][i]["pred"] for i in ALL)
    print(f"Qwen 4b basic 关思考两次运行（结果一那次与本次 Kaggle）：{same}/872 条预测相同")

    print("另外四个系列（Ollama，关思考，重新判分后）")
    fam = {}
    for name in ("mistral-7b", "llama3.1-8b", "gemma3-12b", "phi4-14b"):
        for prompt, size in (("basic", "all"), ("cot", "n200")):
            fam[(name, prompt)] = one(f"ollama_{name}_{prompt}_think-off_{size}_*", M / "more_models")
            describe(f"{name}/{prompt}", fam[(name, prompt)])
    names = ("mistral-7b", "llama3.1-8b", "gemma3-12b", "phi4-14b")
    print("和 Qwen、DeepSeek 比（basic 关思考，872 条）")
    for n in names:
        b = fam[(n, "basic")]
        compare(f"{n} 对 Qwen 4b", b, q[("4b", "basic", "off")], ALL)
        compare(f"{n} 对 Qwen 9b", b, q[("9b", "basic", "off")], ALL)
        compare(f"{n} 对 DeepSeek", b, dsb, ALL)
    print("四个系列两两比较（basic，872 条）")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            compare(f"{a} 对 {b}", fam[(a, "basic")], fam[(b, "basic")], ALL)
    print("basic 对 cot（关思考，前 200 条）；cot 直接只答一个数字（输出不超过 5 个 token）的条数")
    for n in names:
        compare(f"{n} basic 对 cot", fam[(n, "basic")], fam[(n, "cot")], FIRST200)
        print(f"    cot 直接只答数字 {sum(fam[(n, 'cot')][i]['tokens'] <= 5 for i in FIRST200)}/200，"
              f"cot 每条平均 {statistics.mean(fam[(n, 'cot')][i]['seconds'] for i in FIRST200):.1f} 秒")
    eight = [fam[(n, "basic")] for n in names] + [q[("4b", "basic", "off")], q[("9b", "basic", "off")], dsb]
    wrong = [i for i in ALL if all(not s[i]["correct"] for s in eight)]
    print(f"  四个系列、Qwen 4b、Qwen 9b、DeepSeek（basic 关思考）都错：{len(wrong)} 条，"
          f"标签 {dict(Counter(dsb[i]['label'] for i in wrong))}，idx {wrong}")


if __name__ == "__main__":
    main()
