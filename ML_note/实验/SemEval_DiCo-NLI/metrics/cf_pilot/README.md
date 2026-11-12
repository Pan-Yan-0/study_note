# 反事实最小对试点（研究对齐 v4 的 B 部分）

规则在生成和审核之前写定（`code/cf_pilot.py`，冻结计划变更记录 11-12「研究对齐 v4」B 部分）。父样本是 fold 0 fit 里两段共享至少一个实词的金标 E 家族，排除了 Claude 看过金标的家族，只按金标和文本筛选，不用任何模型预测。

## 结果

**两种编辑都放行**（按 Codex 的最终标签计分，放行线是"编辑项分歧率 ≤ 同现象原数据对照的分歧率 + 15 个百分点"，用整数交叉相乘判定）。

| 编辑 | 编辑项分歧 | 对照分歧 | 判定 |
|---|---|---|---|
| CF-SPEC（插入 1–3 词的限制性修饰语，候选 F / B） | 3 / 20（15%） | 1 / 15（7%），U 5 条 | pass |
| CF-HEAD（换一个中心词，候选 O） | 1 / 20（5%） | 4 / 20（20%） | pass |

- **生成**：6 批，每批一个全新的 Haiku 子 agent，transcript 核对模型都是 `claude-haiku-5-5`。合格率 SPEC 25 / 30（另 5 条 skip）、HEAD 28 / 30（另 2 条 skip），没有因格式被过滤的。每种编辑抽 20 对进审核表，另配同现象的原数据对照各 20 对，四类混排。
- **盲标和最终标签**：Codex 先在只有短语的第一包上盲标，交表后才看第二包（Claude 初标、候选标签和金标、生成原文）。
  - 只用盲标计分（看答案前已封存）：SPEC 编辑 3 / 18、对照 1 / 14；HEAD 编辑 1 / 20、对照 4 / 19。两种编辑都 pass，和最终标签的结论相同。
  - 看完第二包后，Codex 把 4 条 U 改成了确定标签（C006 → B、C013 → F、C017 → B、C070 → O），都写了理由，改后的标签都和答案一致；另有 3 条只改了依据的措辞。
- **两位审阅人的一致**（两人都给了确定标签的行）：SPEC 编辑 17 / 20、SPEC 对照 14 / 15、HEAD 编辑 18 / 20、HEAD 对照 18 / 20。Claude 初标不完全盲（编辑规则是 Claude 写的），只作描述。

## 问题

- **SPEC 的 3 条分歧都是改 text2、Codex 判 O**：Many cars / A group of parked cars、A large jet / A large passenger airplane、within the open field / in the muddy field。三条的父样本本身就是数据集口径下的 E（"许多"对"一群"、jet 对 airplane、限定词不同），按逻辑蕴含并不等价，再加一个修饰语后审阅人判成 O。也就是说，问题出在父样本的 E 比逻辑等价宽，不在插入的修饰语。改 text1 的 10 条分歧为 0。
- **HEAD 的 1 条分歧**：Nobel poet / Nobel laureate，换词后变成上下位，Codex 判 F。规则要求"非同义、非上下位"，生成时没拦住。
- **对照的分歧**是原数据金标和逻辑判断的差异：A young girl / a little girl、down / along the sidewalk 金标 O，审阅人判 E；A lamb / A sheep、of cows / of cattle 金标 O，审阅人判 F；trapped / feared trapped 金标 B，审阅人判 O。这和之前几轮盲审看到的口径差异同类。
- **规模**：每种编辑 20 对，区间很宽，放行只说明没有发现比原数据明显更差的生成质量。

## 下一步计划

按冻结计划：
1. **B1 测试集**：从每折留出集的金标 E 生成编辑对，只按金标和文本条件筛选，审核后锁定标签，再用每折的 Q0 在 T4 fp16 上推理（先复现归档预测，再评编辑对）。报编辑后准确率、联合准确率、条件比例。
2. **B2 训练**：CF-k 对目标标签受控的仿写 IMIT-k，同一批 fit 父样本。

## 文件

| 文件 | 内容 |
|---|---|
| `batches/`、`outputs/` | 6 批的提示词输入和 Haiku 原文 |
| `dispatch_log.jsonl` | 派发记录；容器内的绝对路径换成了 `<run>/`，原文件 SHA-256 `daa70efa…` |
| `valid_edits.json`、`pilot_key.json` | 合格的编辑和父样本 |
| `review_sheet.csv`（`28189c7d…`）、`review_key.json`（`e1d5aea0…`） | 审核表和答案，审核前封存 |
| `review_sheet_Claude.csv`（`70b8f57d…`） | Claude 初标，审核前封存 |
| `review/review_sheet_Codex_stage1.csv`（`fa6aea37…`）、`review/review_result_stage1.json`（`6ca8564e…`） | Codex 盲标和只用盲标的计分 |
| `review/review_sheet_Codex_final.csv`（`d4c7ebee…`）、`review_result.json` | Codex 最终标签和计分（Claude 用 `cf_pilot.py score` 独立重算，和 Codex 交回的结果相同） |
| `review/参考卡.md` | 审核用的参考卡 |

## 来源

生成由 Claude 用 Agent 工具调度 Haiku 子 agent 完成；脚本由 Claude 编写；审核人 Codex（盲标、最终标签），Claude 初标只作描述。
