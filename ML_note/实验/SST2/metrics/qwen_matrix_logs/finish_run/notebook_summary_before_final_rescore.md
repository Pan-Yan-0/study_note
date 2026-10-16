# SST-2 剩余实验汇总

总用时 11.20 小时。开思考的组每张卡 4 路并行（出过问题的模型改为单路，见“这次 16 组”的并行列）。

## 这次 16 组

| 模型 | 模板 | 思考 | 状态 | 在哪张卡 | 并行 | 跑之前 → 跑之后 | 用时（分钟） |
|---|---|---|---|---|---:|---|---:|
| qwen3.5:0.8b | cot | off | completed | gpu0 | 1 | 0 → 200 / 200 | 8.3 |
| qwen3.5:0.8b | role_cot | off | completed | gpu1 | 1 | 0 → 200 / 200 | 8.4 |
| qwen3.5:2b | cot | off | completed | gpu0 | 1 | 0 → 200 / 200 | 12.4 |
| qwen3.5:2b | role_cot | off | completed | gpu1 | 1 | 0 → 200 / 200 | 11.9 |
| qwen3.5:9b | cot | off | completed | gpu0 | 1 | 0 → 200 / 200 | 14.1 |
| qwen3.5:9b | role_cot | off | completed | gpu1 | 1 | 0 → 200 / 200 | 17.3 |
| qwen3.5:0.8b | basic | on | completed | gpu0 | 1 | 152 → 200 / 200 | 22.6 |
| qwen3.5:0.8b | cot | on | completed | gpu0 | 1 | 109 → 200 / 200 | 54.3 |
| qwen3.5:0.8b | role_cot | on | completed | gpu1 | 1 | 0 → 200 / 200 | 120.2 |
| qwen3.5:2b | cot | on | completed | gpu0 | 1 | 86 → 200 / 200 | 84.8 |
| qwen3.5:2b | role_cot | on | completed | gpu1 | 1 | 0 → 200 / 200 | 152.2 |
| qwen3.5:4b | cot | on | completed | gpu0 | 1 | 86 → 200 / 200 | 78.9 |
| qwen3.5:4b | role_cot | on | completed | gpu0 | 1 | 99 → 200 / 200 | 60.4 |
| qwen3.5:9b | basic | on | completed | gpu0 | 1 | 60 → 200 / 200 | 43.9 |
| qwen3.5:9b | cot | on | stopped_deadline | gpu0 | 1 | 22 → 184 / 200 | 285.2 |
| qwen3.5:9b | role_cot | on | stopped_deadline | gpu1 | 1 | 0 → 168 / 200 | 356.1 |

## 24 组的前 200 条准确率（已用修正后的解析规则重新判分）

| 模型 | basic 思考off | basic 思考on | cot 思考off | cot 思考on | role_cot 思考off | role_cot 思考on |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5:0.8b | 74.50% | 56.50% | 77.50% | 42.50% | 77.50% | 39.50% |
| qwen3.5:2b | 83.50% | 85.50% | 86.50% | 68.00% | 86.50% | 68.00% |
| qwen3.5:4b | 92.00% | 92.50% | 94.50% | 80.50% | 90.00% | 85.00% |
| qwen3.5:9b | 94.00% | 95.00% | 92.50% | 66.85%（184 条） | 93.00% | 70.83%（168 条） |

## 每组明细

| 模型 | 模板 | 思考 | 完成/目标 | 准确率 | 前 200 条 | 无效 | 截断 | 请求失败 | 秒/条 | 输出 token/条 | 结果文件 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5:0.8b | basic | off | 872/872 | 78.78% | 74.50% | 2 | 0 | 0 | 0.207 | 2.6 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_basic_think-off_all_8808a639c2ba |
| qwen3.5:0.8b | basic | on | 200/200 | 56.50% | 56.50% | 73 | 73 | 0 | 24.658 | 2618.5 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_basic_think-on_n200_b64dc64a59d6 |
| qwen3.5:0.8b | cot | off | 200/200 | 77.50% | 77.50% | 3 | 0 | 0 | 2.495 | 219.5 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_cot_think-off_n200_2c1236ba3a00 |
| qwen3.5:0.8b | cot | on | 200/200 | 42.50% | 42.50% | 107 | 107 | 0 | 34.215 | 3541.4 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_cot_think-on_n200_0e246b02592a |
| qwen3.5:0.8b | role_cot | off | 200/200 | 77.50% | 77.50% | 7 | 1 | 0 | 2.511 | 223.4 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_role_cot_think-off_n200_7ab8bd177637 |
| qwen3.5:0.8b | role_cot | on | 200/200 | 39.50% | 39.50% | 113 | 110 | 0 | 36.062 | 3591.9 | qwen3.5-0.8b_de63045f2975/ollama_qwen3.5-0.8b_role_cot_think-on_n200_e5032c6ed50f |
| qwen3.5:2b | basic | off | 872/872 | 83.14% | 83.50% | 12 | 0 | 0 | 0.125 | 2.6 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_basic_think-off_all_7aaee91ff0a4 |
| qwen3.5:2b | basic | on | 200/200 | 85.50% | 85.50% | 18 | 18 | 0 | 17.912 | 1194.0 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_basic_think-on_n200_d9de6d47a35c |
| qwen3.5:2b | cot | off | 200/200 | 86.50% | 86.50% | 6 | 1 | 0 | 3.725 | 235.4 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_cot_think-off_n200_bcc3b9a42eec |
| qwen3.5:2b | cot | on | 200/200 | 68.00% | 68.00% | 55 | 54 | 0 | 43.418 | 2905.0 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_cot_think-on_n200_3b785a2f31a2 |
| qwen3.5:2b | role_cot | off | 200/200 | 86.50% | 86.50% | 5 | 0 | 0 | 3.572 | 226.5 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_role_cot_think-off_n200_8b89dd0dbac1 |
| qwen3.5:2b | role_cot | on | 200/200 | 68.00% | 68.00% | 56 | 56 | 0 | 45.666 | 3030.2 | qwen3.5-2b_0689d44085e0/ollama_qwen3.5-2b_role_cot_think-on_n200_37f640d4434f |
| qwen3.5:4b | basic | off | 872/872 | 93.81% | 92.00% | 0 | 0 | 0 | 0.238 | 2.5 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd |
| qwen3.5:4b | basic | on | 200/200 | 92.50% | 92.50% | 9 | 9 | 0 | 15.44 | 717.3 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_basic_think-on_n200_f99f49af8375 |
| qwen3.5:4b | cot | off | 200/200 | 94.50% | 94.50% | 0 | 0 | 0 | 3.028 | 132.0 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_cot_think-off_n200_3b6d1cb8adef |
| qwen3.5:4b | cot | on | 200/200 | 80.50% | 80.50% | 33 | 33 | 0 | 41.58 | 1900.3 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_cot_think-on_n200_feef13cbaf24 |
| qwen3.5:4b | role_cot | off | 200/200 | 90.00% | 90.00% | 1 | 0 | 0 | 3.2 | 129.3 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_role_cot_think-off_n200_818616711ace |
| qwen3.5:4b | role_cot | on | 200/200 | 85.00% | 85.00% | 25 | 25 | 0 | 36.038 | 1625.8 | qwen3.5-4b_d8b0f5e9760c/ollama_qwen3.5-4b_role_cot_think-on_n200_5be799124791 |
| qwen3.5:9b | basic | off | 872/872 | 93.35% | 94.00% | 0 | 0 | 0 | 0.334 | 2.5 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_basic_think-off_all_592c71504568 |
| qwen3.5:9b | basic | on | 200/200 | 95.00% | 95.00% | 5 | 5 | 0 | 22.312 | 554.0 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_basic_think-on_n200_5194c28620c6 |
| qwen3.5:9b | cot | off | 200/200 | 92.50% | 92.50% | 5 | 1 | 0 | 4.237 | 108.1 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_cot_think-off_n200_028407195a0b |
| qwen3.5:9b | cot | on | 184/200 | 66.85% | 66.85% | 56 | 55 | 0 | 111.401 | 3013.2 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_cot_think-on_n200_7bb4e18c58f6 |
| qwen3.5:9b | role_cot | off | 200/200 | 93.00% | 93.00% | 4 | 0 | 0 | 5.196 | 120.4 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_role_cot_think-off_n200_ac9c856ab4e1 |
| qwen3.5:9b | role_cot | on | 168/200 | 70.83% | 70.83% | 48 | 48 | 0 | 126.444 | 3172.8 | qwen3.5-9b_56671c2ab938/ollama_qwen3.5-9b_role_cot_think-on_n200_1b58c504eaed |

## 运行记录

- [0.68 小时] [gpu0] qwen3.5:0.8b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [0.72 小时] [gpu1] qwen3.5:0.8b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [1.06 小时] [gpu0] qwen3.5:0.8b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [1.97 小时] [gpu0] qwen3.5:4b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [2.73 小时] [gpu1] qwen3.5:2b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [2.98 小时] [gpu0] qwen3.5:9b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [3.71 小时] [gpu0] qwen3.5:4b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [5.03 小时] [gpu0] qwen3.5:2b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [5.26 小时] [gpu1] qwen3.5:9b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [6.45 小时] [gpu0] qwen3.5:9b：Ollama 只开了 1 路并行（设置的是 4 路）。
- [11.20 小时] 实验结束，已用 11.20 小时。
