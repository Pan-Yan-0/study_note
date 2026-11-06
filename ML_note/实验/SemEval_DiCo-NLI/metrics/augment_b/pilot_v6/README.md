# 实验 B：v6 试点的生成原文、处理结果和 P1 审核

v6 试点通过了全部自动停止条件和 P1 审核，`gate.json` 已放行（`ready_for_training: true`）。分析见 [生成侧说明](../../../code/augment_b/README.md) 第 9 节。

| 文件 | 内容 |
|---|---|
| `raw/outputs/<批次>/attempt_N.txt` | GPT6Luna 子 agent 的原始回复（GPT 调度，原样保存），`check_N.json` 是格式检查的输出 |
| `raw/debug/` | 调试批次 `debug_b05` 的两次尝试和继续判定（不进入任何数据版本） |
| `raw/dispatch_log.jsonl`、`raw/debug_dispatch_log.jsonl`、`raw/GPT交接说明.md`、`raw/GPT意见.md` | 调度日志、GPT 的交接说明和意见 |
| `report.json`、`gate.json` | 统计、停止条件判定和放行状态 |
| `manifest.csv` | 放行的候选清单：204 对展开成 349 行（SHA-256 `448ade30…`，等于 `gate.json` 的 `bound_to`） |
| `filter_decisions.csv`、`attempts.csv` | 每对的过滤结果、每次尝试的记录 |
| `review_sheet.csv`、`review_key.json`、`review_result.json` | P1 审核表、答案键和计分结果 |
| `p1/Pan-Yan-0标注.csv`、`p1/Pan-Yan-0原始回复.txt`、`p1/Claude标注.csv` | 两人的独立标注（Claude 的先封存哈希 `59d4ee88…`） |
| `diagnosis.json` | `diagnose_pilot.py` 的描述性诊断 |

## 来源

生成由 GPT 调度 GPT6Luna 子 agent 执行；批次、提示词、过滤和诊断代码由 Claude 编写，处理和诊断由 Claude 运行。P1 审核由 Pan-Yan-0 和 Claude 各自独立标注。
