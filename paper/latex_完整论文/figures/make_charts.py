# -*- coding: utf-8 -*-
"""根据建模文档中的实测数值重绘论文用图（中文标注）。
所有数字均取自全量运行结果表，不做任何插值或补点。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ["../SimSun.ttf"]:
    p = (HERE / f).resolve()
    if p.exists():
        fm.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.edgecolor": "#8a8984",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#2b2b29",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "axes.axisbelow": True,
})
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK2 = "#52514e"

# ---------------- 图：缺失率鲁棒性（基线 LDSD，整模态缺失） ----------------
r = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
acc3 = [68.78, 65.47, 62.04, 59.56, 56.40, 54.33, 50.76]
f1 = [60.48, 56.04, 51.43, 47.03, 42.13, 35.88, 28.71]
mae = [0.6628, 0.6927, 0.7527, 0.7693, 0.7750, 0.8178, 0.8619]
corr = [0.6722, 0.6164, 0.5240, 0.4883, 0.4467, 0.3440, 0.2283]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
ax.plot(r, acc3, color=BLUE, lw=1.6, marker="o", ms=4.5, mec="white", mew=1.0, label="Acc-3")
ax.plot(r, f1, color=ORANGE, lw=1.6, marker="s", ms=4.2, mec="white", mew=1.0, label="F1-macro")
ax.axhline(100 / 3, color=GRAY, lw=0.9)
ax.text(-0.02, 100 / 3 + 1.2, "三分类随机水平 33.3%", ha="left", va="bottom", color=INK2, fontsize=9)
ax.set_xlabel("缺失率 r")
ax.set_ylabel("百分比 / %")
ax.set_ylim(20, 75)
ax.set_title("(a) 分类指标", fontsize=10.5, color="#2b2b29")
ax.legend(loc="upper right", fontsize=9)
for x, y in [(0.0, 68.78), (0.9, 50.76)]:
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=INK2)
ax.annotate("28.71", (0.9, 28.71), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8.5, color=INK2)

ax = axes[1]
ax.plot(r, mae, color=BLUE, lw=1.6, marker="o", ms=4.5, mec="white", mew=1.0, label="MAE（越低越好）")
ax.plot(r, corr, color=ORANGE, lw=1.6, marker="s", ms=4.2, mec="white", mew=1.0, label="Pearson 相关（越高越好）")
ax.set_xlabel("缺失率 r")
ax.set_ylabel("数值")
ax.set_ylim(0.1, 1.0)
ax.set_title("(b) 回归指标", fontsize=10.5, color="#2b2b29")
ax.legend(loc="lower left", fontsize=9)
ax.annotate("0.862", (0.9, 0.8619), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=INK2)
ax.annotate("0.228", (0.9, 0.2283), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(HERE / "fig_robustness.png")
plt.close(fig)

# ---------------- 图：局部连续缺失，基线 vs 改进模型 ----------------
cond_f1 = ["完整模态", "段缺失\n（整体）", "长段\n>15 帧"]
f1_base = [60.5, 47.3, 46.7]
f1_ours = [60.9, 61.7, 62.7]
cond_mae = ["完整模态", "段缺失\n（整体）", "短段\n≤5 帧", "中段\n6–15 帧", "长段\n>15 帧"]
mae_base = [0.663, 0.711, 0.678, 0.703, 0.720]
mae_ours = [0.656, 0.664, 0.617, 0.685, 0.655]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [3, 5]})
import numpy as np
w = 0.36
ax = axes[0]
x = np.arange(len(cond_f1))
b1 = ax.bar(x - w / 2 - 0.01, f1_base, w, color=ORANGE, label="基线 LDSD")
b2 = ax.bar(x + w / 2 + 0.01, f1_ours, w, color=BLUE, label="LDSD-Seg")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, cond_f1, fontsize=9)
ax.set_ylabel("F1-macro / %")
ax.set_ylim(0, 75)
ax.set_title("(a) F1-macro（越高越好）", fontsize=10.5, color="#2b2b29")
ax.grid(axis="x", visible=False)

ax = axes[1]
x = np.arange(len(cond_mae))
for i in range(len(x)):
    ax.plot([i, i], [mae_base[i], mae_ours[i]], color="#c9c8c3", lw=2.0, zorder=1)
ax.scatter(x, mae_base, s=46, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3, label="基线 LDSD（整模态缺失增强）", marker="s")
ax.scatter(x, mae_ours, s=50, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3, label="LDSD-Seg（M1+M4）")
for i in range(len(x)):
    ax.text(i + 0.12, mae_base[i], f"{mae_base[i]:.3f}", va="center", fontsize=8, color=INK2)
    ax.text(i + 0.12, mae_ours[i], f"{mae_ours[i]:.3f}", va="center", fontsize=8, color=INK2)
ax.set_xticks(x, cond_mae, fontsize=9)
ax.set_xlim(-0.5, len(x) - 0.3)
ax.set_ylabel("MAE")
ax.set_ylim(0.60, 0.73)
ax.set_title("(b) MAE（越低越好）", fontsize=10.5, color="#2b2b29")
ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
l = ["基线 LDSD（整模态缺失增强）", "LDSD-Seg（M1+M4）"]
fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "fig_segment_compare.png")
plt.close(fig)

# ---------------- 图：候选模块逐级筛选 ----------------
cfg = ["基线\n（逐帧散点训练）", "M1\n（保留）", "M1+M2\n（淘汰）", "M1+M3\n（淘汰）", "M1+M4\n（保留）", "M1+M4+M5\n（淘汰）"]
seg_f1 = [47.63, 57.03, 57.25, 56.37, 60.21, 58.02]
full_f1 = [56.65, 57.65, 57.92, 56.92, 60.86, 56.81]
keep = ["ref", "keep", "drop", "drop", "keep", "drop"]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = np.arange(len(cfg))
b1 = ax.bar(x - w / 2 - 0.01, seg_f1, w, color=BLUE, label="段缺失 F1-macro")
b2 = ax.bar(x + w / 2 + 0.01, full_f1, w, color=AQUA, label="完整模态 F1-macro")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, cfg, fontsize=9)
ax.set_ylim(0, 70)
ax.set_ylabel("F1-macro / %")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(HERE / "fig_module_screen.png")
plt.close(fig)
print("done")

# ---------------- 图：问题一三种对齐方案的非零位置数 ----------------
methods = ["方法一\n（口径H）", "方法二\n（全程50等分）", "方法三\n（说话者有效时段）"]
nz_a = [22.5, 50.0, 48.0]
nz_v = [21.1, 46.1, 47.5]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
x = np.arange(len(methods))
w = 0.34
b1 = ax.bar(x - w / 2 - 0.01, nz_a, w, color=BLUE, label="语音")
b2 = ax.bar(x + w / 2 + 0.01, nz_v, w, color=ORANGE, label="视觉")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x, methods, fontsize=9)
ax.set_ylabel("平均非零位置数（共50）")
ax.set_ylim(0, 58)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(HERE / "fig_q1_nonzero.png")
plt.close(fig)
print("q1 chart done")
