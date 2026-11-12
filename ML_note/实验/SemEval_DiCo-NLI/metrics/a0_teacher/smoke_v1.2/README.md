# A0 Kaggle SMOKE（代码 v1.2，通过）

## 结果

Kaggle T4，fp16，torch 2.11.0+cu128 / CUDA 12.8，transformers 5.19.0、peft 0.21.2、accelerate 1.15.0、datasets 5.1.0；模型快照 `cdbee75f…`。fold 0 seed 42，选中第 3 轮，adapter `e07c957b…` 和归档 `COMPLETE.json` 一致，载入 504 个张量。

| 检查 | 实测 | v1.2 门槛 |
|---|---|---|
| 留出 sym 标签一致率 | 608 / 608 | ≥ 99% |
| 99% 分位概率分量差 | 0.044 | ≤ 0.10 |
| 单方向 argmax 翻转 | 4 / 1,448（0.28%） | ≤ 1% |
| 内层验证 argmax 一致率 | 243 / 244（99.6%） | ≥ 98% |
| 单条最大差（只报告） | 0.184（Robber in Cannes / Cannes thief，argmax 不变） | — |
| 教师推理 | 128 条，顺序和 B2 输入一致，概率合法，mass < 0.9 的 0 条 | — |

复现统计和 v1.1 的 SMOKE（[smoke_v1.1_fail](../smoke_v1.1_fail/README.md)）逐值相同，同卡同精度下推理是确定的。Claude 用 v1.2 的 `accept()` 离线复核回执，0 个问题；运行身份的代码、数据、包清单、门槛都和 v1.2 一致。

## 问题

最大差和 4 个翻转的成因见 v1.1 的说明：归档是 bf16 量化的，3 个翻转在归档里是精确平局。长尾没有隔离到具体算子。

## 下一步计划

用同一个包和 notebook 跑全量（`RUN_FULL = True`，Commit）：两张 T4 并行，每折先 seed 42 的复现门槛，15 份教师推理逐份验收。

## 来源

Pan-Yan-0 在 Kaggle 上运行；脚本由 Claude 编写；门槛修订经 Codex 同意（[复核](../Codex复核_v1.2.md)）。
