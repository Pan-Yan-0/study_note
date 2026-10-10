# -*- coding: utf-8 -*-
"""v3 论文新增的两张图：缺失规律（取自原文表13）与专项推理概览（取自原文附录B、C）。
所有数字都照抄原文表格，不做插值。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent
fm.fontManager.addfont(str((HERE / "../SimSun.ttf").resolve()))
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 10,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
    "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300,
})
BLUE, ORANGE, AQUA, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"

# ---------------- 缺失规律（原文表13，PM-B2 seed 17） ----------------
groups = {
    "(a) 被遮蔽模态": (["T", "A", "V", "TA", "TV", "AV", "TAV"],
                  [0.5563, 0.5975, 0.5942, 0.5583, 0.5573, 0.5952, 0.5582],
                  [0.6487, 0.6125, 0.6126, 0.6494, 0.6488, 0.6131, 0.6495],
                  [0.5894, 0.6144, 0.6156, 0.5887, 0.5900, 0.6162, 0.5908],
                  [0.6979, 0.6500, 0.6494, 0.6985, 0.6988, 0.6507, 0.6994]),
    "(b) 名义缺失比例": (["10%", "30%", "50%"],
                   [0.5905, 0.5716, 0.5595], [0.6180, 0.6300, 0.6526],
                   [0.6141, 0.6001, 0.5879], [0.6601, 0.6756, 0.6978]),
    "(c) 缺失位置": (["头", "中", "尾"],
                 [0.5761, 0.5734, 0.5720], [0.6397, 0.6283, 0.6326],
                 [0.6002, 0.6072, 0.5948], [0.6778, 0.6678, 0.6879]),
}
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [2.3, 1, 1]})
for j, (title, (labs, vf, vm, tf, tm)) in enumerate(groups.items()):
    x = np.arange(len(labs))
    for row, (va, te, ylab) in enumerate([(vf, tf, "macro-F1"), (vm, tm, "MAE")]):
        ax = axes[row, j]
        ls = "none" if j == 0 else "-"   # 模态组合是无序类别，不连线
        ax.plot(x, va, color=BLUE, lw=1.6, ls=ls, marker="o", ms=6, mec="white", mew=1.0, label="验证集")
        ax.plot(x, te, color=ORANGE, lw=1.6, ls=ls, marker="s", ms=5.4, mec="white", mew=1.0, label="测试集")
        ax.set_xticks(x, labs)
        ax.set_xlim(-0.4, len(labs) - 0.6)
        if j == 0:
            ax.set_ylabel(ylab)
        if row == 0:
            ax.set_title(title, fontsize=10, color="#2b2b29")
            ax.set_ylim(0.54, 0.63)
        else:
            ax.set_ylim(0.60, 0.71)
        if j > 0:
            ax.set_yticklabels([])
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(HERE / "fig_q2_missing_law.pdf")
plt.close(fig)

# ---------------- 专项推理概览（原文附录 B、C） ----------------
def load(name):
    with open(HERE / "../data" / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))

a3, a4 = load("att3.csv"), load("att4.csv")
colors = {"负": ORANGE, "中": GRAY, "正": BLUE}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
for k, name in [("负", "负向"), ("中", "中性"), ("正", "正向")]:
    pts = [(float(r["y"]), float(r["pmax"])) for r in a3 if r["pol"] == k]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=colors[k],
               edgecolor="white", linewidth=1.0, label=f"{name}（{len(pts)}条）", zorder=3)
ax.axvline(0, color=GRAY, lw=0.8, zorder=1)
ax.set_xlabel("预测强度")
ax.set_ylabel("最大类别概率")
ax.set_title("(a) 附件3（30条）", fontsize=10, color="#2b2b29")
ax.set_ylim(0.3, 1.02)
ax.legend(loc="upper left", fontsize=8.5)

ax = axes[1]
tt = [r for r in a4 if r["dom_cls"] == "T" and r["dom_reg"] == "T"]
nt = [r for r in a4 if not (r["dom_cls"] == "T" and r["dom_reg"] == "T")]
ax.scatter([float(r["y"]) for r in tt], [float(r["p"]) for r in tt], s=34, color=BLUE,
           edgecolor="white", linewidth=1.0, label=f"两头均以文本为主导（{len(tt)}条）", zorder=3)
ax.scatter([float(r["y"]) for r in nt], [float(r["p"]) for r in nt], s=40, color=ORANGE, marker="D",
           edgecolor="white", linewidth=1.0, label=f"至少一头非文本主导（{len(nt)}条）", zorder=3)
offs = {"01": (-14, -2), "02": (6, -3), "05": (7, -5), "13": (6, -4), "14": (6, 1)}
for r in nt:
    ax.annotate(r["id"], (float(r["y"]), float(r["p"])), textcoords="offset points",
                xytext=offs.get(r["id"], (5, -3)), fontsize=8, color=INK2)
ax.axvline(0, color=GRAY, lw=0.8, zorder=1)
ax.set_xlabel("预测强度")
ax.set_title("(b) 附件4（20条）", fontsize=10, color="#2b2b29")
ax.set_ylim(0.3, 1.02)
ax.set_yticklabels([])
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.30), fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "fig_special_overview.pdf")
plt.close(fig)
print("done")
