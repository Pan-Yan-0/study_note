# 实验 B Commit 3：Focal / Dice loss 对照

老师提出的对照：在只用原始英语 fit 数据、其余配置和 B0 完全相同的条件下，把 CE 换成 Focal 或 Dice，看能不能缓解类别不平衡（O 类只占约 16%）。mDeBERTa-v3-base（第五组 C0 配方），英语按家族五折 × 种子 42–44，Kaggle T4 x2、fp16，2 组共 30 项。代码是 `train_augb_v4.py` v4.1（`d16e182e…`），由 `make_v4.py` 从 v3 精确替换生成；CE 那一行不变，Focal / Dice 在 fp32、autocast 之外计算。设计由 Codex 提出，Claude 实现，双方同意后写入冻结计划变更记录（11-10）。

| 组 | loss |
|---|---|
| B0（复用 Commit 1） | CE |
| Focal | 四分类 softmax focal，α = 1，γ = 2：`mean(−(1 − p_t)² · log p_t)` |
| Dice | 四类 macro soft Dice，按 minibatch 汇总，缺席的类也参与平均，eps = 1e-6 |

## 结果

**B0 复用通过**：v4 跑的 B0 SMOKE（fold 0、seed 42、两步）和 Commit 1 归档的 B0 SMOKE，留出预测、内层验证预测、额外 logits 三个文件逐字节相同，done.json 只差指纹和耗时；torch / cuda / transformers 都是 2.11.0+cu128 / 12.8 / 4.57.6。所以 B0 直接用 Commit 1 的 15 项，正式队列只跑了 Focal 和 Dice。SMOKE 里 loss 数值检查在 GPU 上通过（Focal γ = 0 和 CE 的 loss、梯度差都是 0；γ = 2 的手算样例一致），三组各实际更新 2 次、没有跳步。

Claude 用 `code/claude_recheck_augb_c3.py` 从每个任务的四类概率重建，30 项核对 0 个问题。核对内容：哈希、选轮、步数；task.json 和 Commit 1 的 B0 只差代码身份、loss、group、group_spec，config 只差 `code_version`、`loss_variants` 两条说明；每折训练行数等于 fit，计划步数等于同种子同折的 B0。

**三种子平均（OOF 3,042 行）**：

| 组 | sym WF1 | plain WF1 | SoftCons | HardCons | O 的 F1 / P / R | O 的 AP |
|---|---|---|---|---|---|---|
| B0（CE） | 79.37% | 79.10% | 90.54% | 82.19% | 58.6 / 55.3 / 62.6 | 56.31% |
| Focal | 80.04% | 78.93% | 90.25% | 82.50% | 59.7 / 55.4 / 64.8 | 56.59% |
| Dice | 78.07% | 77.38% | 91.32% | 81.05% | 58.9 / 56.8 / 61.3 | 55.12% |
| W4（参照，Commit 1） | 80.15% | 79.31% | 90.61% | 82.55% | 60.6 / 56.5 / 65.4 | 53.83% |

SoftCons、HardCons 和 O 的指标是对称化之后的。

**事先登记的主比较**（同种子 sym WF1 差，百分点；按家族配对重抽样 2,000 次，两条比较用 Bonferroni 97.5% 区间；规则：均值 > 0 且下限 > 0）：

| 比较 | 种子 42 / 43 / 44 | 均值 | 97.5% 区间 | 95% 区间 | 判定 |
|---|---|---|---|---|---|
| Focal − B0 | +0.70 / +0.58 / +0.73 | +0.67 | [−0.19, +1.50] | [−0.08, +1.39] | 不算提升 |
| Dice − B0 | +0.50 / −3.22 / −1.16 | −1.29 | [−2.38, −0.25] | [−2.23, −0.37] | 显著更差 |

**描述**：

| 比较 | 均值 [95% 区间] |
|---|---|
| Focal − B0（plain） | −0.17 [−0.86, +0.49] |
| Dice − B0（plain） | −1.73 [−2.58, −0.84] |
| Focal − W4 | −0.11 [−0.99, +0.74] |
| Focal − W2 | +0.49 [−0.31, +1.28] |
| Dice − Focal | −1.96 [−2.92, −1.05] |

- **Focal**：三个种子都比 B0 高约 0.6–0.7 个百分点，方向一致，但区间含 0，按规则不算提升。只在对称化后才有这个差，plain 方向是 −0.17。O 的召回升了约 2 个百分点、精确率不变，AP 几乎不变（56.59% 对 56.31%），所以主要是 O 的判定稍微放宽，排序能力没变。和 W4 差不多（−0.11），而且没有 W4 那样把 O 的 AP 拉低约 2.5 个百分点。
- **Dice**：显著变差，O 的 AP 也降了。种子之间差别很大（+0.50 / −3.22 / −1.16）。种子 43 的 fold 3 选在第 3 轮，留出 sym WF1 只有 0.692（B0 同折 0.782）；内层验证只有 244 行，选轮本身有噪声。这是按既定选轮规则的结果，只作描述，不改判定。Dice 并不是学得慢：第 1 轮内层验证（0.30–0.52）比 B0（0.21–0.48）还高，第 3 轮平均和 B0 相同；差在后期，选中轮的内层验证平均 0.763，B0 0.777，第 12 轮 0.753 对 0.764。
- **后续**：冻结计划规定"至少一个 loss 过线才做增广 + 新 loss 的叠加"。两个都没过线，所以叠加不做。

## 问题

- **区间不包含重新训练的随机性**：重抽样只抽家族。Focal 三个种子同号但幅度小；Dice 种子间波动和效应同量级。
- **registry 把这两条归为系统比较**：设计上只改 loss 一处，但 loss 记成 `method` 和 `method_params` 两个字段，registry 的规则按字段数归类。W2 / W4 也是同样情况。
- **超参数没有调**：Focal 只试了 γ = 2，Dice 只用了 batch 级、四类都参与的写法，学习率、轮数都沿用 CE 的 C0 配方。Dice 的梯度范数比 CE 小一个数量级（SMOKE 里 0.24–0.28 对 1.4–1.9），AdamW 对梯度尺度不敏感，所以这不直接说明有效学习率变小。这次的结论只针对这个配方，不说明 Dice 本身不好。
- **fp16 跳步**：Focal 15 项合计跳过 34 次更新，Dice 36 次，只报告。选中轮平均：Focal 第 8.5 轮，Dice 第 9.1 轮（B0 第 8.8 轮）。
- **多重比较**：主比较两条已经用 Bonferroni 校正；描述的几条比较没有校正。

## 下一步计划

1. 实验 B 当前登记的训练（+1、Commit 1–3）都已完成。生成数据扩充有效、在 +2 饱和；类别加权和这两种 loss 都没有按规则带来提升。
2. 要不要继续试 Dice 的其他写法（调学习率、和 CE 混合）或 Focal 的其他 γ，由 Pan-Yan-0 和老师定；要试的话要事先登记比较和规则，不在看了这次结果之后改判定。
3. 下一阶段做什么（Qwen 上的扩充、其他赛道验证、或论文写作）回到 [实验状态与算力计划](../../../实验状态与算力计划.md) 由 Pan-Yan-0 决定。

## 文件

| 文件 | 内容 |
|---|---|
| `claude_recheck.json` | 全部分数（含逐类 P / R / F1、O 的 AP、O 的预测数）、比较、每个任务的训练记录 |
| `oof/` | 每组每个种子的 OOF 标签（plain / sym） |
| `tasks/<组>/seed*/fold*/` | 原样的 `task.json`、`done.json`、留出和内层验证概率、额外 logits |
| `smoke/` | B0 / Focal / Dice 的 SMOKE，含 `grad_trace` |
| `loss_checks.json`、`b0_reuse.json` | GPU 上的 loss 数值检查、B0 复用判定 |
| `input_data/`、`logs/` | 五折划分、+1 的清单（只在预检用）、两个 worker 的日志 |

## 来源

Pan-Yan-0 在 Kaggle T4 x2 上运行；loss 对照由老师提出、Codex 设计并验收 v4 / v4.1 的代码；`train_augb_v4.py` 由 Claude 在 v3（Claude 在 GPT 的 v2 训练核心上改出）上加 loss 分支；重建和分析脚本由 Claude 编写。Focal loss 见 Lin et al., *Focal Loss for Dense Object Detection*（ICCV 2017）；Dice loss 见 Milletari et al., *V-Net*（3DV 2016）和 Li et al., *Dice Loss for Data-imbalanced NLP Tasks*（ACL 2020）。
