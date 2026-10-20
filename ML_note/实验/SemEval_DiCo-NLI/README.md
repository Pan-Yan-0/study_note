# SemEval-2027 Task 2：DiCo-NLI

方向一致的细粒度自然语言推理。给一对短语 (text1, text2)，判断四个标签之一：

- EQUIVALENCE；
- FORWARD_ENTAILMENT：text1 能推出 text2；
- BACKWARD_ENTAILMENT：text2 能推出 text1；
- NEGATIVE_OTHER。

同一对短语的两个方向都要预测，并且要互相一致：对调后 FORWARD 和 BACKWARD 互换，EQUIVALENCE 不变。

- 官方网站：https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/
- 官方仓库（数据、评分脚本、starter kit）：https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI
- 评测窗口：2027-01-10 至 01-31

## 数据

| | Track 1 英语 | Track 2 西班牙语 | Track 3 巴斯克语 | Track 4 跨语言 |
|---|---:|---:|---:|---:|
| 训练集条数 | 3,042 | 3,042 | 3,042 | 18,252 |
| dev 条数 | 660 | 660 | 660 | 3,960 |

- **训练集的构成**：1,760 个短语对（pair_id）。
  - 1,282 对可以对调，各有正反两条（后缀 `original`、`flipped`）。
  - 478 对是 NEGATIVE_OTHER，只有一条。
- **dev 的构成**：383 对，其中 277 对可以对调。训练集和 dev 没有重叠的短语对。
- **三个单语赛道和 Track 4 是同一批内容**：Track 4 是这批内容的 6 种跨语言组合。
- **训练集标签分布**（单语赛道）：FORWARD 881、BACKWARD 881、EQUIVALENCE 802、NEGATIVE_OTHER 478。
- **短语很短**：两个短语合计平均约 6 个词。

## 结果

dev 集，官方评分脚本。随机基线是 3 个种子的平均。

| 系统 | 赛道 | Weighted F1 | SoftCons | HardCons |
|---|---|---:|---:|---:|
| majority（全部预测 FORWARD） | 1–4 | 0.129 | 0 | 0 |
| random | 1 | 0.254 | 0.196 | 0.065 |
| random | 2 | 0.259 | 0.194 | 0.067 |
| random | 3 | 0.249 | 0.185 | 0.060 |
| random | 4 | 0.251 | 0.190 | 0.063 |

官方试点（LREC 2026）里 DeBERTa-v3-base 的成绩是 WF1 0.75、SoftCons 0.84、HardCons 0.79。它只做英语，只用三个正向标签，和这里的设置不同，只作参考。

## 问题

1. **数据结构会泄露 NEGATIVE_OTHER。**
   - 训练集和 dev 里，NEGATIVE_OTHER 只出现在只有一条记录的短语对上。
   - 官方 trial 里测试格式的文件也一样：只有一条记录的短语对，125 / 125 都是 NEGATIVE_OTHER。
   - 如果正式测试集还是这种结构，只数 pair_id 出现几次就能认出 NEGATIVE_OTHER，这和理解短语之间的关系无关。
   - 本仓库的模型只用两个短语的文本，不用 pair_id 和 instance_id 后缀。要不要告诉主办方，待定。
2. **SoftCons 会把 NEGATIVE_OTHER 算作"不一致"。** 评分脚本只把"一个方向预测为可对调的标签，另一个方向正好是它的反向"算作一致。所以在可对调的短语对上，只要有一个方向预测成 NEGATIVE_OTHER，这一对就不一致。
3. **巴斯克语是低资源语言。** 多语言模型在它上面的表现需要单独看，不能只看平均分。
4. **PhrasIS 原数据能否用于训练**，以官方最终规则为准。确认之前不用。

## 下一步计划

1. 运行 [notebooks/dico_encoders.ipynb](notebooks/dico_encoders.ipynb)：
   - 模型：mBERT、mDeBERTa-v3-base；
   - 赛道：四个；
   - 种子：每组 3 个；
   - 输入：`[CLS] text1 [SEP] text2 [SEP]`；
   - 轮数：只在训练集里按 pair_id 分出 10% 来选，dev 只用来报告；
   - 预测：同时记录"只看原方向"和"两个方向对称化后"两种结果。
2. 大模型：Qwen 零样本、少样本、CoT，然后用 LoRA 做 SFT。
3. 在最好的编码器上逐个试 R-Drop、SCL、MoCo。
4. 用 DeepSeek 扩充数据，看数据量和成绩的关系。
5. 反事实推断（导师讲解后）。

## 文件

- `code/baselines.py`：random / majority 基线，调用官方评分脚本。
- `notebooks/dico_encoders.py` / `.ipynb`：第一组编码器实验，可以断点续跑。
- `metrics/baselines/summary.json`：基线的逐种子分数。

## 来源

- 数据、评分脚本和试点结果来自官方仓库。
- 统计、基线脚本和 notebook 由 Claude 编写，基线由 Claude 运行。
