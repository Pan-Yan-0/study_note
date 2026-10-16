"""复算 README 第 4 节 Qwen3-4B 指令微调的数字。

用法：python qwen3_sft_report.py [metrics/qwen3_4b_sft 目录]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "metrics" / "qwen3_4b_sft"
SUMMARY = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
zero = pd.read_csv(ROOT / "validation_zero_shot.csv.gz", dtype={"id": str})
sft = pd.read_csv(ROOT / "validation_sft.csv.gz", dtype={"id": str})
assert (zero["id"].values == sft["id"].values).all()
y = sft["label"].values


def ece(y_true, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y_true[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def report(name, frame):
    p = frame["p_positive"].values
    pred = (p >= 0.5).astype(int)
    top1 = frame["top1_is_answer"].astype(str).str.lower().eq("true").values
    wrong = pred != y
    print(f"[{name}]")
    print(f"  accuracy {accuracy_score(y, pred):.4f}  AUC {roc_auc_score(y, p):.6f}  "
          f"answer_loss {frame['answer_loss'].mean():.4f}  answer_mass {frame['answer_mass'].mean():.4f}")
    print(f"  FP {int((wrong & (y == 0)).sum())}  FN {int((wrong & (y == 1)).sum())}  "
          f"Brier {brier_score_loss(y, p):.4f}  ECE {ece(y, p):.4f}  "
          f"错且置信度≥0.99 {int((wrong & (np.maximum(p, 1 - p) >= 0.99)).sum())}")
    if (~top1).any():
        print(f"  第一名是答案词 {top1.mean():.4f}：准确率 {accuracy_score(y[top1], pred[top1]):.4f}；"
              f"不是时 {accuracy_score(y[~top1], pred[~top1]):.4f}（{int((~top1).sum())} 条）")
    return pred, p


pz, qz = report("zero-shot", zero)
ps, qs = report("SFT", sft)

fixed = int(((pz != y) & (ps == y)).sum())
broken = int(((pz == y) & (ps != y)).sum())
print(f"配对：SFT 改对 {fixed}，改错 {broken}，McNemar 精确 p = {binomtest(min(fixed, broken), fixed + broken, 0.5).pvalue:.2g}")

# 按条重抽样 2000 次，求 AUC 及其差值的 95% 区间
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    boot.append((roc_auc_score(y[i], qz[i]), roc_auc_score(y[i], qs[i])))
boot = np.array(boot)
for label, values in (("zero-shot AUC", boot[:, 0]), ("SFT AUC", boot[:, 1]), ("差值", boot[:, 1] - boot[:, 0])):
    print(f"{label} 95% 区间 [{np.percentile(values, 2.5):.4f}, {np.percentile(values, 97.5):.4f}]")

log = pd.read_csv(ROOT / "training_log.csv")
steps = log[log["loss"].notna()]
for a, b in ((1, 50), (51, 200), (201, 600), (601, 1000), (1001, 1250)):
    seg = steps[steps["step"].between(a, b)]
    print(f"第 {a}–{b} 步：平均 loss {seg['loss'].mean():.4f}，平均梯度范数 {seg['grad_norm'].mean():.3f}")

# 吞吐：平均 token 数取自 Notebook 的分词输出（训练 315、验证 314、测试 310）；
# 每个 token 的运算量按非嵌入参数约 36 亿估算，训练取 6 倍（前向、重算、反向），推理取 2 倍
N = 3.6e9
stages = (("zero-shot 验证", SUMMARY["zero_shot_validation"]["seconds"], 5000 * 314, 2),
          ("训练", SUMMARY["train"]["seconds_last_session"], 20000 * 315, 6),
          ("SFT 后验证", SUMMARY["sft_validation"]["seconds"], 5000 * 314, 2),
          ("测试", SUMMARY["test"]["seconds_last_session"], 25000 * 310, 2))
for name, seconds, tokens, factor in stages:
    print(f"{name}：{seconds / 60:.1f} 分钟，{tokens / seconds:.0f} token/秒，约 {tokens * factor * N / seconds / 1e12:.1f} TFLOPS")

sub = pd.read_csv(ROOT / "submission.csv.gz", dtype={"id": str})
print(f"提交文件：{len(sub)} 行，ID 不重复 {sub['id'].is_unique}，判为正面的比例 {(sub['sentiment'] >= 0.5).mean():.4f}")
