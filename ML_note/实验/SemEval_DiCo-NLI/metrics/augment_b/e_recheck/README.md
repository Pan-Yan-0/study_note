# 实验 B：E 类按原数据校准的复审

比赛数据视为绝对准确：生成的 E 和原数据的 E 混在一起盲审，用同一把尺子量，看生成 E 是否明显差于原数据。规则在看结果之前写定（`code/augment_b/e_recheck.py`，FULL_README 第 12 节）。

## 结果

60 对，隐藏来源，种子 20261109。流程：Claude 初标 → Pan-Yan-0 全部采纳 → GPT 对照参考卡（[E 对照](../e_control/README.md)的 40 对带金标的原数据）复审，改了 6 处。

| 来源 | 组 | 可判定 | Claude 初标不一致 | 最终不一致 | 最终不一致率 |
|---|---|---|---|---|---|
| Haiku | E | 20 | 2 | 3 | 15% |
| 原数据 | E | 20 | 2 | 4 | 20% |
| Haiku | 单向 | 5 | 1 | 1 | 20% |
| 原数据 | 单向 | 5 | 1 | 1 | 20% |
| Haiku | O | 5 | 0 | 0 | 0% |
| 原数据 | O | 5 | 3 | 0 | 0% |

- **放行规则**：Haiku E 不一致率 ≤ 原数据 E + 15 个百分点，两边可判定 ≥ 10。实际 15% 对 20%，**通过**。只看 Claude 初标也是 2/20 对 2/20。
- 合并目录的审核结果记为 `P1C-R`：E 用这次复审，单向和 O 沿用 P1C（22% / 0%）。`gate.json` 全部检查 pass，`ready_for_training: true`，清单 SHA-256 `b380b4a6…`。
- 按比赛的标法审时，原数据 E 的不一致率从 E 对照里的 44% 降到 20%，Haiku E 从 P1 的 47% 降到 15%。

最终和预期不同的 E：

| 编号 | 来源 | 短语对 | 预期 | 最终 |
|---|---|---|---|---|
| Q011 | 原数据 | behind them / in the background | EQ | O |
| Q017 | 原数据 | in the white snow / in the snow | EQ | FE |
| Q023 | 原数据 | are juggling / stand juggling | EQ | BE |
| Q036 | 原数据 | on the water / afloat on the water | EQ | BE |
| Q025 | Haiku | to flood-hit Queensland / flood struck Queensland | EQ | FE |
| Q033 | Haiku | submerged in brine / in the brine | EQ | FE |
| Q045 | Haiku | stacks / is stacked | EQ | O |

## 问题

- 两步依次修改，不是独立标注；Claude 是派发方，这次按 Pan-Yan-0 的决定先标。审核人之间的一致率算不出来。
- GPT 的 6 处修改里，3 处把 Claude 标成 EQ 的原数据 O 改对了（Q019、Q024、Q031），另 3 处让 E 的不一致各多了一个（原数据 Q011、Q017，Haiku Q025）。说明按比赛标法审，"所指相同"和"场景相近"之间的边界仍不稳定。
- 每组只有 20 对，5 个百分点的差别（1 对）分不出高低，只能说 Haiku E 没有明显差于原数据。
- 原数据也有看起来不太对的金标（Q042 NATO troops / NATO 3 troops 金标是 FE），这里按规则视为准确。

## 下一步

Haiku 数据已放行并完成训练，结果见 [+1 训练对照](../train_plus1/README.md)。

## 来源

抽样、计分代码由 Claude 编写；初标由 Claude 完成，Pan-Yan-0 采纳，GPT 对照参考卡复审；原数据来自 SemEval-2027 Task 2 官方英语训练集。
