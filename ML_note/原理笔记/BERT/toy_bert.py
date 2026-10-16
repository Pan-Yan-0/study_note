"""BERT 原理笔记里所有玩具数字的来源。

一句话 [CLS] i love this movie [SEP]，隐藏维度 4（真实 BERT-base 是 768），
一层编码器：2 个注意力头（每头 2 维）、前馈 4 → 16 → 4。
所有权重都是手工设定或固定随机种子生成的，不是训练出来的，只为了能手算。

运行：python toy_bert.py   （只依赖 numpy）
"""
import math

import numpy as np

np.set_printoptions(precision=3, suppress=True)
NAMES = ["[CLS]", "i", "love", "this", "movie", "[SEP]"]
LOVE = 2

# ---------- 1. 输入嵌入：三张表相加，再 LayerNorm ----------
TOK = np.array([[0.1, 0.4, -0.3, 0.2], [0.3, -0.5, 0.2, 0.1], [0.8, -0.2, 0.1, 0.5],
                [-0.2, 0.3, 0.6, -0.1], [0.4, 0.2, -0.4, 0.7], [-0.1, 0.1, 0.1, -0.3]])
POS = np.array([[0.0, 0.1, 0.2, -0.1], [0.1, 0.0, 0.1, 0.1], [0.1, 0.3, -0.1, 0.0],
                [0.2, -0.1, 0.0, 0.2], [-0.1, 0.2, 0.1, 0.0], [0.0, 0.0, -0.2, 0.1]])
SEG_A = np.array([0.0, 0.1, 0.0, -0.1])          # 单句任务，6 个位置都用句子 A


def layer_norm(m):
    """每一列（每个词）自己做：减均值、除以标准差；玩具里 gamma=1, beta=0。"""
    mean = m.mean(-1, keepdims=True)
    var = ((m - mean) ** 2).mean(-1, keepdims=True)
    return (m - mean) / np.sqrt(var + 1e-12)


X = layer_norm(TOK + POS + SEG_A)                # (6, 4)：送进第 1 层的 6 列

# ---------- 2. 自注意力：两个头，各自 q·k/√2 → softmax → 加权求和 v ----------
WQ1 = np.array([[-1, -1], [1, -1], [-1, -1], [1, 1]])   # 每一列是一个神经元的 4 个权重
WK1 = np.array([[1, 1], [1, 1], [0, 0], [-1, 1]])
WV1 = np.array([[1, 0], [0, 1], [-1, 1], [1, -1]])
WQ2 = np.array([[0, 0], [-1, -1], [0, 0], [0, 1]])
WK2 = np.array([[-1, -1], [-1, 1], [1, 0], [-1, 1]])
WV2 = np.array([[1, 0], [-1, 1], [0, 1], [1, 0]])
WO = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]])


def softmax_rows(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def head(x, wq, wk, wv):
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T / math.sqrt(q.shape[1])      # 6×6 打分表，除以 √(每头维度)
    a = softmax_rows(scores)                      # 每一行加起来等于 1
    return q, k, v, scores, a, a @ v


def gelu(z):
    return 0.5 * z * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


rng = np.random.default_rng(42)
W1 = np.round(rng.normal(0, 0.6, size=(4, 16)), 1)      # 前馈第一层 4 → 16
W2 = np.round(rng.normal(0, 0.3, size=(16, 4)), 1)      # 前馈第二层 16 → 4


def ffn(m):
    return gelu(m @ W1) @ W2                      # 每一行（每个词）各自乘同一套权重


def encoder_layer(x):
    h1, h2 = head(x, WQ1, WK1, WV1), head(x, WQ2, WK2, WV2)
    attn = np.concatenate([h1[5], h2[5]], axis=1) @ WO   # 两个头拼起来再乘 W_O
    ln1 = layer_norm(x + attn)                    # 残差 + LayerNorm
    out = layer_norm(ln1 + ffn(ln1))              # 前馈 + 残差 + LayerNorm
    return h1, h2, attn, ln1, out


if __name__ == "__main__":
    print("1. love 的三项嵌入相加:", (TOK + POS + SEG_A)[LOVE], " LayerNorm 后:", X[LOVE])

    q, k, v, scores, a, out1 = head(X, WQ1, WK1, WV1)
    print("\n2. 头 1：q_love =", q[LOVE])
    print("   各词的 k:\n", k, "\n   各词的 v:\n", v)
    print("   love 的 q·k:", q[LOVE] @ k.T, "\n   ÷√2:", scores[LOVE])
    print("   e^得分:", np.exp(scores[LOVE]), "总和", np.exp(scores[LOVE]).sum())
    print("   权重 a:", a[LOVE], " 输出 Σ a·v =", out1[LOVE])
    print("   完整 6×6 权重表（每行一个提问的词）:\n", a)

    h1, h2, attn, ln1, out = encoder_layer(X)
    print("\n3. 头 2 中 love 的权重:", h2[4][LOVE])
    print("   一层里 love 这一列的变化：")
    print("   输入 x                ", X[LOVE])
    print("   注意力（两头 + W_O）  ", attn[LOVE])
    print("   相加 x + 注意力       ", (X + attn)[LOVE])
    print("   LayerNorm             ", ln1[LOVE])
    print("   前馈 4→16→4           ", ffn(ln1)[LOVE])
    print("   相加                  ", (ln1 + ffn(ln1))[LOVE])
    print("   LayerNorm = 本层输出  ", out[LOVE])
    print("   前馈中间 16 个神经元（GELU 后）:", gelu(ln1 @ W1)[LOVE])

    changed = ln1.copy()
    changed[4] += 5.0                              # 只改 movie 那一列
    print("\n4. 只改 movie 列，love 的前馈输出不变:", np.allclose(ffn(changed)[LOVE], ffn(ln1)[LOVE]))
    x_changed = X.copy()
    x_changed[4] += 5.0
    print("   只改 movie 列，love 的注意力输出会变:", not np.allclose(encoder_layer(x_changed)[2][LOVE], attn[LOVE]))

    big = scores[LOVE] * 8                         # 不除以 √d 时分数会大很多
    print("\n5. 分数放大 8 倍后的 softmax:", softmax_rows(big))
    sim = np.random.default_rng(0)
    for d in (2, 64):
        dots = (sim.normal(size=(100000, d)) * sim.normal(size=(100000, d))).sum(1)
        print(f"   d={d}: 点积标准差 {dots.std():.2f}，除以 √d 后 {(dots / math.sqrt(d)).std():.2f}")
