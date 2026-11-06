# 实验 B：v6 试点的处理结果

v6 试点通过了全部自动停止条件，还差 P1 审核。分析见 [生成侧说明](../../../code/augment_b/README.md) 第 9 节。

| 文件 | 内容 |
|---|---|
| `report.json`、`gate.json` | `process_outputs.py build` 的统计、停止条件判定和放行状态（审核 `pending`，`ready_for_training: false`） |
| `attempts.csv`、`raw/dispatch_log.jsonl`、`raw/debug_dispatch_log.jsonl` | 每次尝试的记录和调度日志 |
| `diagnosis.json` | `diagnose_pilot.py` 的描述性诊断（只有统计，没有短语对） |
| `raw/GPT交接说明.md`、`raw/GPT意见.md` | GPT 的交接说明和对方案的意见 |

**还没有提交的**：生成原文、`manifest.csv`、`filter_decisions.csv`、`review_sheet.csv`、`review_key.json`。它们带着生成标签或能对出审核项。P1 审核完成后再一起提交，以免审核人从仓库看到生成标签。

## 来源

生成由 GPT 调度 GPT6Luna 子 agent 执行；批次、提示词、过滤和诊断代码由 Claude 编写，处理和诊断由 Claude 运行。
