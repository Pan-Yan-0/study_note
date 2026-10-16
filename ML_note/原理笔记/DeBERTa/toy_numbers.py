"""《DeBERTa 与 IMDB 三条路学习笔记》里所有玩具数字的来源。

向量都是 2 维或 4 维（真实模型是 768 维），权重是手工设定的，不是训练出来的，只为了能手算。
特征缩放那一节是固定随机种子生成的模拟数据。

运行：python toy_numbers.py   （只依赖 numpy）
"""
import math
from itertools import product

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def title(text):
    print(f"\n=== {text} ===")


def softmax(x):
    e = np.exp(np.asarray(x, dtype=float))
    return e / e.sum(-1, keepdims=True)


def layer_norm(v):
    """一行（一个词）自己做：减均值、除以标准差；gamma = 1, beta = 0。"""
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / np.sqrt(v.var() + 1e-12)


# ---------- 1. AUC：交概率与交 0/1 标签 ----------
title("1. AUC")
POS = [0.97, 0.91, 0.84, 0.62, 0.55, 0.38]   # 6 条正面影评的得分
NEG = [0.71, 0.45, 0.30, 0.22, 0.10, 0.05]   # 6 条负面影评的得分


def pair_auc(pos, neg):
    """36 个“正面 × 负面”配对：正面得分高记 1，相等记 0.5，低记 0。"""
    total = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(pos, neg))
    return total, total / (len(pos) * len(neg))


wins = [sum(p > n for n in NEG) for p in POS]
print("每条正面影评赢了几条负面：", wins, "合计", sum(wins), "→ AUC =", f"{pair_auc(POS, NEG)[1]:.3f}")

for t in (0.25, 0.5):
    lp = [int(x >= t) for x in POS]
    ln = [int(x >= t) for x in NEG]
    tpr, tnr = sum(lp) / 6, (6 - sum(ln)) / 6
    total, auc = pair_auc(lp, ln)
    print(f"阈值 {t}：判成正面 {sum(lp) + sum(ln)} 条，TPR = {sum(lp)}/6 = {tpr:.3f}，"
          f"TNR = {6 - sum(ln)}/6 = {tnr:.3f}；交标签时配对得分 {total}/36 = {auc:.3f}，"
          f"(TPR + TNR)/2 = {(tpr + tnr) / 2:.3f}")

lp = [int(x >= 0.5) for x in POS]
ln = [int(x >= 0.5) for x in NEG]
cats = {}
for p, n in product(lp, ln):
    cats[(p, n)] = cats.get((p, n), 0) + 1
print("阈值 0.5 交标签时的配对（正面标签, 负面标签）: 对数", cats)
print("正面 0.38 对负面 0.30：交概率得", 1.0 if 0.38 > 0.30 else 0.0, "分；交标签都是 0，得 0.5 分")


# ---------- 2. 嵌入相加与 LayerNorm ----------
title("2. 嵌入相加与 LayerNorm")
word, pos, seg = [0.2, 0.5, -0.1, 0.3], [0.1, 0.0, 0.2, -0.1], [0.0, 0.1, 0.0, 0.0]
x = np.add(np.add(word, pos), seg)
print("词向量 + 位置 + 段落 =", x, "→ LayerNorm =", layer_norm(x).round(2))
print("[1, 2, 3, 4] →", layer_norm([1, 2, 3, 4]).round(2),
      "；[10, 20, 30, 40] →", layer_norm([10, 20, 30, 40]).round(2), "（尺度不变）")
print("三个 768 维向量若拼接：", 768 * 3, "维；逐位相加：768 维")


# ---------- 3. 为什么除以 √d ----------
title("3. softmax 与 √d")
print("不除：softmax([16, 0, -8]) =", softmax([16, 0, -8]).round(4))
print("除以 8：softmax([2, 0, -1]) =", softmax([2, 0, -1]).round(2))
print("√64 =", math.sqrt(64), "；8 个头时每头", 768 // 8, "维，除以 √96 =", round(math.sqrt(96), 2))


# ---------- 4. 一个头的注意力（BERT，2 维玩具） ----------
title("4. BERT 一个头的注意力")
H_BAD, H_NOT = np.array([1.0, 0.5]), np.array([0.8, -0.6])
WQ = np.array([[1, 0], [0, 2]])
WK = np.array([[0, 1], [1, 0]])
q_bad, k_bad, k_not, q_not = H_BAD @ WQ, H_BAD @ WK, H_NOT @ WK, H_NOT @ WQ
scores = np.array([q_bad @ k_not, q_bad @ k_bad])
w = softmax(scores / math.sqrt(2))
print("q_bad =", q_bad, " k_bad =", k_bad, " k_not =", k_not)
print("bad 看 [not, bad] 的分数 =", scores, "→ 除以 √2 再 softmax =", w.round(3))
print("W_v 取单位矩阵时 Δx_bad = 权重 × V =", (w[0] * H_NOT + w[1] * H_BAD).round(3))
print("q_not =", q_not, "；not 看 bad 的分数 q_not·k_bad =", round(float(q_not @ k_bad), 2))


# ---------- 5. 参数账 ----------
title("5. 参数账（BERT-base 一层）")
attn = 4 * (768 * 768 + 768)
ffn1, ffn2 = 768 * 3072 + 3072, 3072 * 768 + 768
print(f"注意力 W_q/W_k/W_v/W_o 共 {attn:,}；前馈 W_1 {ffn1:,}，W_2 {ffn2:,}；"
      f"一层前馈 {ffn1 + ffn2:,}，12 层前馈 {12 * (ffn1 + ffn2):,}")
print(f"词向量表 30,522 × 768 = {30522 * 768:,}")


# ---------- 6. DeBERTa 解耦注意力（d = 2，k = 4，投影都取单位矩阵） ----------
title("6. DeBERTa 解耦注意力")
K = 4
P = np.array([[-0.2, 0.1], [-0.1, 0.2], [0.1, 0.3], [0.6, -0.2],
              [0.0, 0.0], [0.9, 0.4], [0.2, 0.1], [-0.1, 0.0]])   # 2k = 8 行，第 δ 行代表一种距离


def delta(i, j, k=K):
    """距离 r = i - j 换成 P 表的行号；超出 ±k 的距离并到两端。"""
    r = i - j
    if r <= -k:
        return 0
    if r >= k:
        return 2 * k - 1
    return r + k


print("r → δ（k = 4）：", {r: delta(r, 0) for r in range(-5, 6)})
print("真实 v1（k = 512）：第 3 个看第 700 个 δ =", delta(3, 700, 512),
      "；第 700 个看第 3 个 δ =", delta(700, 3, 512))

print("bad（query）看 not（key），r = i − j：")
for r in (1, 2, 4):
    c2c = H_BAD @ H_NOT
    c2p = H_BAD @ P[delta(r, 0)]
    p2c = H_NOT @ P[delta(0, r)]
    s = c2c + c2p + p2c
    print(f"  r = +{r}：δ(i,j) = {delta(r, 0)}，δ(j,i) = {delta(0, r)}，内容→内容 {c2c:.2f}，"
          f"内容→位置 {c2p:.2f}，位置→内容 {p2c:.2f}，合计 {s:.2f}，÷√6 = {s / math.sqrt(6):.3f}")

H_THE = np.array([-0.5, 1.0])     # 假设的另一个 query 词
print("同一距离，不同的词加分不同：bad·P[5] =", round(float(H_BAD @ P[5]), 2), " bad·P[7] =", round(float(H_BAD @ P[7]), 2),
      "；the·P[5] =", round(float(H_THE @ P[5]), 2), " the·P[7] =", round(float(H_THE @ P[7]), 2))

# 两个词的完整例子："not bad"，not 在位置 0，bad 在位置 1
H = np.array([H_NOT, H_BAD])
n = len(H)
c2c = H @ H.T
c2p_all = H @ P.T          # n × 2k：每个词对每种距离的在意程度（Q_c × K_rᵀ）
p2c_all = H @ P.T          # n × 2k（K_c × Q_rᵀ；投影都取单位矩阵，所以数值相同）
c2p = np.array([[c2p_all[i, delta(i, j)] for j in range(n)] for i in range(n)])
p2c = np.array([[p2c_all[j, delta(j, i)] for j in range(n)] for i in range(n)])
S = c2c + c2p + p2c
W = softmax(S / math.sqrt(3 * 2))
dx = W @ H
print("\n'not bad' 完整例子（行 = query：not, bad；列 = key：not, bad）")
print("c2p_all（每行 8 个数）=\n", c2p_all.round(2))
print("内容→内容 =\n", c2c.round(2))
print("内容→位置（挑数后）=\n", c2p.round(2))
print("位置→内容（挑数后）=\n", p2c.round(2))
print("三项相加 =\n", S.round(2), "\n÷√6 =\n", (S / math.sqrt(6)).round(3))
print("softmax 权重 =\n", W.round(3))
print("Δx = 权重 × V（V = H）=\n", dx.round(3), "\nx + Δx =\n", (H + dx).round(3))
print("只用内容→内容（÷√2）时的权重 =\n", softmax(c2c / math.sqrt(2)).round(3))

print("\n'not very bad'（not = 0，very = 1，bad = 2）的 δ(i,j) 表：")
print(np.array([[delta(i, j) for j in range(3)] for i in range(3)]))


# ---------- 7. 特征缩放：同一个学习率管不了尺度差很多的参数 ----------
title("7. 特征缩放模拟（房价 = 0.1 × 面积 + 20 × 卧室数）")
rng = np.random.default_rng(0)
N = 200
area = rng.uniform(300, 2000, N)
bed = rng.integers(1, 6, N).astype(float)
y = 0.1 * area + 20 * bed


def gradient_descent(X, lr, steps):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        err = X @ w + b - y
        w -= lr * X.T @ err / N
        b -= lr * err.mean()
    return w


X = np.c_[area, bed]
print("从 w = 0 出发的第一步梯度（面积, 卧室数）=", (X.T @ (-y) / N).round(1))
with np.errstate(over="ignore", invalid="ignore"):
    print("不缩放，学习率 2e-6，50 步：w =", gradient_descent(X, 2e-6, 50), "（发散）")
print("不缩放，学习率 1e-6，1000 步：w =", gradient_descent(X, 1e-6, 1000).round(3))
mu, sd = X.mean(0), X.std(0)
w_std = gradient_descent((X - mu) / sd, 0.1, 100)
print("标准化后，学习率 0.1，100 步：换回原单位 w =", (w_std / sd).round(3))


# ---------- 8. 从 logprob 到 p(正面) ----------
title("8. logprob → p(正面)")
p1, pm = math.exp(-0.05), math.exp(-3.2)
print(f"P('1') = e^-0.05 = {p1:.4f}，P('-') = e^-3.2 = {pm:.4f}，其他 = {1 - p1 - pm:.4f}，"
      f"p(正面) = {p1 / (p1 + pm):.4f}")


# ---------- 9. SST-2 的输出 token 数：-1 是两个 token ----------
title("9. SST-2 输出 token 数")
ds_neg, ds_pos = 421 + 35, 7 + 409       # DeepSeek 预测 -1 / 1 的条数（混淆矩阵按列相加）
qw_neg, qw_pos = 408 + 34, 20 + 410      # Qwen 预测 -1 / 1 的条数
print(f"DeepSeek：{ds_neg} × 2 + {ds_pos} × 1 = {ds_neg * 2 + ds_pos}")
print(f"Qwen（每条多一个结束符）：{qw_neg} × 3 + {qw_pos} × 2 = {qw_neg * 3 + qw_pos * 2}")


# ---------- 10. batch 与梯度累积（训练集 20,000 条，跑一轮） ----------
title("10. batch 与梯度累积")
train = 20000
for micro, accum in ((16, 1), (8, 2), (4, 4), (1, 16), (1, 1)):
    passes, updates = math.ceil(train / micro), math.ceil(train / (micro * accum))
    print(f"{micro} × {accum}：有效 batch {micro * accum}，GPU 前向 + 反向 {passes:,} 次，更新参数 {updates:,} 次")
