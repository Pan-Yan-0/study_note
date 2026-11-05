# 实验 B：v4 试点的生成原文和处理结果

v4 试点没有通过事先定的停止条件（平均词数偏离 65%，探针高 22.9 个百分点），数据不进入任何训练版本，只留档。分析见 [生成侧说明](../../../code/augment_b/README.md) 第 9 节。

| 文件 | 内容 |
|---|---|
| `raw/outputs/<批次>/attempt_N.txt` | GPT6Luna 子 agent 的原始回复（GPT 调度，原样保存），`check_N.json` 是格式检查的输出 |
| `raw/dispatch_log.jsonl`、`raw/GPT交接说明.md` | 16 次尝试的调度记录和 GPT 的交接说明 |
| `report.json`、`gate.json` | `process_outputs.py build` 的统计、停止条件判定和放行状态（`ready_for_training: false`） |
| `filter_decisions.csv`、`manifest.csv`、`attempts.csv` | 每对的过滤结果、候选清单（未放行）、每次尝试的记录 |
| `diagnosis.json` | `diagnose_pilot.py` 的描述性诊断 |

## 来源

生成由 GPT 调度 GPT6Luna 子 agent 执行；批次、提示词、过滤和诊断代码由 Claude 编写，处理和诊断由 Claude 运行。
