# 实验 B：E 类审核标准对照

不生成，只抽原训练集，看审核人对**原数据**的 E 有多认同。规则在看结果之前写定（`code/augment_b/e_control.py`，FULL_README 第 11 节）。

## 结果

40 对：原训练集 E 20 对，加 20 个干扰项（单向 12、O 8），隐藏金标。标注流程和 Haiku 的 P1C 相同：GPT 先标并写理由，Pan-Yan-0 审核，几乎全部同意。

| 金标 | 可判定 | 和金标不同 | 不一致率 |
|---|---|---|---|
| E | 18 | 8 | 44% |
| 单向 | 9 | 3 | 33% |
| O | 8 | 2 | 25% |

- 原数据 E 的不一致率 44%，和 Haiku 生成 E 的 47%（9/19）几乎一样。按 30% 的门槛，原数据自己的 E 也通不过。
- 按事先的规则（m_orig ≥ 30%），v7 只收紧修饰语和具体程度、人名变体保留，并且要先把门槛问题交回 Pan-Yan-0。

金标是 E、审核人不认同的 8 对：

| 编号 | 短语对 | 审核 |
|---|---|---|
| C003 | carried out by law-enforcement officers / by police | BE |
| C011 | from gunshot / from shot | FE |
| C016 | keeping hold of / is holding | FE |
| C020 | A room / an indoor space | FE |
| C026 | on a metal band / on a metal bar | O |
| C031 | to Philippines / for the Philippine nation | O |
| C034 | An adult man / an adult male | FE |
| C035 | to Mandela / to Nelson Mandela | BE |

另有 2 对审核人拿不准（C018 nuclear talks / negotiations over nuclear matters，C025 work / is working）。

## 问题

- 数据集的 E 不是严格的双向逻辑蕴含，更接近"在同一张图或同一条新闻里指同一个东西"：上下位但所指相同（police / law-enforcement officers）、姓和全名（Mandela / Nelson Mandela）、时态和体（is holding / keeping hold of）、介词换了说法（to Philippines / for the Philippine nation）都标 E。
- 另一方面，多出一个修饰语就不算 E：lush green field、black train、young men、playing fetch 金标都是单向，审核人也都同意。
- 审核人按逻辑蕴含判断，这和数据集的标法有系统差别，所以 30% 的绝对门槛量的主要是这个差别，不是生成质量。样本只有 18–19 对，两边 3 个百分点的差别分不出高低。

## 下一步

由 Pan-Yan-0 定：改成和原数据对照的相对门槛，在新抽的样本上重新审核（门槛事先写定，不复用已经看过的这批）。

## 来源

抽样和计分代码由 Claude 编写；标注由 GPT 先标、Pan-Yan-0 审核；原数据来自 SemEval-2027 Task 2 官方英语训练集。
