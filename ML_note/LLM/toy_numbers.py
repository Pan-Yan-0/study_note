"""复算《解码器大模型与路线 C 学习笔记》里的玩具数字。只用 Python 标准库。"""
import math


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


# 1. 双向注意力与因果掩码：[CLS] not bad 三个词的分数
scores = [[1.0, 2.0, 0.5], [0.5, 1.0, 1.5], [0.2, 2.0, 1.0]]
print("双向：", [[round(w, 3) for w in softmax(row)] for row in scores])
causal = [[s if j <= i else -1e6 for j, s in enumerate(row)] for i, row in enumerate(scores)]
print("因果：", [[round(w, 3) for w in softmax(row)] for row in causal])

# 2. LayerNorm 与 RMSNorm（γ = 1、β = 0）
x = [1, 2, 3, 4]
mean = sum(x) / len(x)
std = math.sqrt(sum((v - mean) ** 2 for v in x) / len(x))
print("LayerNorm：", [round((v - mean) / std, 2) for v in x])
rms = math.sqrt(sum(v * v for v in x) / len(x))
print("RMSNorm：", round(rms, 3), [round(v / rms, 3) for v in x])
print("RMSNorm 缩放不变：", [round(2 * v / math.sqrt(sum((2 * u) ** 2 for u in x) / 4), 3) for v in x])

# 3. LM 头的参数量（Qwen3-0.6B：隐藏维度 1024，词表 151,936，全模型约 5.96 亿）
lm_head = 1024 * 151_936
print("LM 头参数：", f"{lm_head:,}", f"占 5.96 亿的 {lm_head / 596e6:.0%}")

# 4. 嵌入 + 逻辑回归的参数量
print("逻辑回归参数：", 1024 + 1)

# 5. 蒸馏：p_T = 0.8 时的交叉熵、熵和 KL
def ce(pt, ps):
    return -(pt * math.log(ps) + (1 - pt) * math.log(1 - ps))


def kl(pt, ps):
    return pt * math.log(pt / ps) + (1 - pt) * math.log((1 - pt) / (1 - ps))


pt = 0.8
for ps in (0.6, 0.8):
    print(f"p_S = {ps}：交叉熵 {ce(pt, ps):.3f} = 熵 {ce(pt, pt):.3f} + KL {kl(pt, ps):.3f}")
best = min((ce(pt, i / 1000), i / 1000) for i in range(1, 1000))
print("交叉熵最小处：", best[1], round(best[0], 3))


# 6. 软标签与硬标签学生的 AUC（手设分数）
def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


soft = ([0.95, 0.48], [0.05, 0.40, 0.52])
hard = ([0.99, 0.03], [0.02, 0.04, 0.97])
for name, (pos, neg) in (("软标签", soft), ("硬标签", hard)):
    acc = (sum(p > 0.5 for p in pos) + sum(n < 0.5 for n in neg)) / 5
    print(f"{name}学生：AUC {auc(pos, neg):.3f}，准确率 {acc:.1f}")

# 7. 成本：IMDB 25,000 条，basic 不思考（输入 1 元/百万，输出 4 元/百万）
inp, out = 25_000 * (300 + 110), 25_000 * 1
cost = inp / 1e6 * 1 + out / 1e6 * 4
print(f"IMDB：输入 {inp:,}，输出 {out:,}，约 {cost:.2f} 元；API 8 路 {25_000 * 1 / 8 / 60:.0f} 分钟；"
      f"Ollama 单路 {25_000 * 0.4 / 3600:.1f} 小时")
print("agent：50 次 × 10 万 token =", f"{50 * 100_000:,}", "输入 token，约", 50 * 100_000 / 1e6, "元")

# 8. SFT：整段都算损失时，答案 token 只占 1/301 的权重
print(f"整段平均：{(300 * 3.0 + 0.7) / 301:.4f}，答案权重 {1 / 301:.2%}；只算答案：0.7")
print(f"p(positive)=0.6、negative=0.3：损失 {-math.log(0.6):.3f}，两词之间归一化 {0.6 / 0.9:.3f}")
for name, loss in (("零样本", 0.9986), ("SFT 后", 0.0930)):
    print(f"{name}：答案平均交叉熵 {loss}，几何平均概率 {math.exp(-loss):.3f}")

# 9. T4 用时：每个 token 训练约 6N（前向、重算、反向各 2N），推理约 2N；N 取 36 亿
N = 3.6e9
train_flops = 20_000 * 315 * 6 * N
print(f"训练约 {train_flops:.2e} 次运算；实测 222 分钟 → {train_flops / (222.2 * 60) / 1e12:.1f} TFLOPS")
