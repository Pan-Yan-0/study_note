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

dev 集，官方评分脚本。编码器和随机基线都是 3 个种子的平均；括号里是 WF1 的样本标准差。

### 编码器（[notebooks/dico_encoders.ipynb](notebooks/dico_encoders.ipynb)）

- 输入：`[CLS] text1 [SEP] text2 [SEP]`；6 轮，按训练集里分出的 10%（按 pair_id 分组）选轮。
- "对称化"：把 p(text2, text1) 的 FORWARD / BACKWARD 对调，再和 p(text1, text2) 取平均。不需要训练。

| 系统 | 赛道 | 原方向 WF1 | SoftCons | HardCons | 对称化 WF1 | SoftCons | HardCons |
|---|---|---:|---:|---:|---:|---:|---:|
| mDeBERTa-v3-base | 1 英语 | .792 (.018) | .847 | .793 | **.805** | .915 | **.844** |
| mDeBERTa-v3-base | 2 西班牙语 | .746 | .816 | .746 | **.758** | .912 | **.810** |
| mDeBERTa-v3-base | 3 巴斯克语 | .720 | .826 | .735 | **.723** | .905 | **.775** |
| mDeBERTa-v3-base | 4 跨语言 | .743 | .834 | .749 | **.745** | .882 | **.780** |
| mBERT | 1 英语 | .722 | .770 | .691 | .733 | .857 | .741 |
| mBERT | 2 西班牙语 | .718 | .777 | .688 | .723 | .864 | .745 |
| mBERT | 3 巴斯克语 | .692 | .764 | .668 | .698 | .840 | .717 |
| mBERT | 4 跨语言 | .634 | .753 | .624 | .638 | .830 | .668 |

- mDeBERTa 在四个赛道上都比 mBERT 好。差距最大的是跨语言赛道（WF1 .745 对 .638）。
- 对称化在 8 组里都提高了 SoftCons（+5–10 个点）和 HardCons（+3–6 个点），WF1 提高 0–1.3 个点。代价是推理要多做一次前向。
- 英语的 mDeBERTa 已经超过官方试点里 DeBERTa-v3-base 的成绩（.75 / .84 / .79）。但试点只做三个正向标签，两边不能直接比较。
- 选中的轮数（3 个种子）：mDeBERTa 是 4–6，12 组里有 6 组选在最后一轮（第 6 轮）；赛道 1、3 的内部验证 F1 到第 6 轮还在上升。逐轮日志在各次运行的 `train_log.json` 里。

### 浅层探针（[code/probes.py](code/probes.py)）

逻辑回归，字符 2–4-gram TF-IDF。下表是英语赛道，单次运行：

| 探针 | 原方向 WF1 / Soft / Hard | 对称化 WF1 / Soft / Hard |
|---|---|---|
| 表面特征（两侧长度、词重叠、包含关系） | .559 / .791 / .570 | .546 / .791 / .570 |
| 只看 text1 | .453 / .426 / .289 | .543 / .960 / .639 |
| 只看 text2 | .437 / .412 / .292 | .530 / .964 / .635 |
| 两侧词袋 + 表面特征 | .620 / .845 / .646 | .618 / .888 / .668 |
| random 基线 | .254 / .196 / .065 | — |
| majority 基线（全部预测 FORWARD） | .129 / 0 / 0 | — |

- "只看一侧"的探针一旦对称化，就同时用到了两侧短语：每一侧各给一个分布，再合起来。所以对称化后的分数不能再叫"单侧输入的成绩"。
- 西班牙语、巴斯克语的结果在 `metrics/probes/summary.json`，规律相同。

### 错例分析（[code/error_analysis.py](code/error_analysis.py)，英语 dev，mDeBERTa）

下面的准确率按 383 个短语对的原方向统计，和官方 WF1（660 行、两个方向都算）不是同一个口径。

- **FORWARD 和 BACKWARD 直接颠倒的很少**，只占 1–2%。但 EQUIVALENCE 和单向蕴含之间（单向还是双向）仍有错误，这也是方向判断的一部分。
- **错误集中在 NEGATIVE_OTHER 的边界**。原方向上，3 个种子多数票错了 92 / 383 对：
  - 金标 NEG 被判成别的：44；
  - 别的被判成 NEG：26；
  - EQUIVALENCE 和单向蕴含互相混淆：19；
  - FORWARD 和 BACKWARD 互相混淆：3。
- **按编辑形状分**：

  | 形状 | 定义 | 训练集对数 | dev 对数 | mDeBERTa 准确率 |
  |---|---|---:|---:|---:|
  | 插入 | 一侧的词是另一侧的子集 | 458 | 93 | .91 |
  | 单词替换 | 两侧各多出 1 个词 | 294 | 90 | .74 |
  | 无共同词 | | 365 | 70 | .69 |
  | 其他 | | 641 | 129 | .75 |
  | 词相同、顺序不同 | | 2 | 1 | — |

  - 插入类用"词多的一侧推出词少的一侧"这一条规则，就能对 86%。规则不成立的那部分，模型只有 .56 的准确率，35% 的预测跟着规则走。
- **切片 oracle**（[code/slice_oracle.py](code/slice_oracle.py)）：只把某个切片里的预测改成金标，其余不动，再用官方脚本重算。它只说明这个切片最多能影响多少，不是任何方法能达到的效果。对称化后、3 个种子平均的 WF1 增量：

  | 切片 | 英语 | 西班牙语 | 巴斯克语 |
  |---|---:|---:|---:|
  | 插入类里长度规则不成立的部分 | +.017 | +.034 | +.023 |
  | 单词替换 | +.051 | +.036 | +.054 |

- **短语很短**：平均 3 个词，39% 不超过 2 个词。训练集 1,760 对里，含否定词的 5 对，含量词的 33 对。
- **有 31 对在 6 次运行（2 个模型 × 3 个种子）里全部判错。** 其中有些金标和字面上的词汇逻辑不一致，原因还没查清。例如：
  - "over child ‖ over daughter" 的金标是 NEGATIVE_OTHER；
  - 训练集里同一个有序输入 "South Africans → South Africa"，在 dico_phrasis_0000065 的 flipped 行标 FORWARD，在 dico_phrasis_0002985 标 NEGATIVE_OTHER。只看文本的模型不可能同时满足这两条。

## 问题

1. **数据结构会泄露 NEGATIVE_OTHER。**
   - 训练集和 dev 里，NEGATIVE_OTHER 只出现在只有一条记录的短语对上。
   - 官方 trial 里测试格式的文件也一样：只有一条记录的短语对，125 / 125 都是 NEGATIVE_OTHER。
   - 本仓库的模型只用两个短语的文本。要不要告诉主办方，待定。
2. **对称化以后的 SoftCons 是构造出来的。**
   - 对称化后两个方向在构造上就互为对调，剩下的不一致只来自"预测成 NEGATIVE_OTHER"。评分脚本里，可对调的短语对只要有一个方向预测成 NEGATIVE_OTHER，就算不一致。
   - 所以"只看一侧"的探针对称化后 SoftCons 有 .96，比 mDeBERTa 还高。
   - 每个方法都报两行：原方向预测的三个指标，和对称化后的三个指标。比较模型本身的一致性时看原方向的 SoftCons；对称化后的 SoftCons 照官方口径报，但只说明避免了预测 NEGATIVE_OTHER。
3. **主要难点是 NEGATIVE_OTHER 的边界和单词替换。**
   - 被替换的词是同义词、上下位词，还是只是相似或对比，很大程度上决定了标签。
   - dev 的 90 个替换词对里，只有 9 个在训练集出现过。
   - "主要缺的是词汇知识"目前还是假设。错误也可能来自词义选择、搭配，或者标注本身。
4. **一部分金标和字面逻辑不一致，原因未明。**
   - PhrasIS 的作者稿写明，对齐的短语对是脱离原句语境重新标注的。所以不是"丢了原句语境"。这一条由 GPT 查到，Claude 没有取到原文。
   - 剩下的可能：标注指南有自己的口径、标注噪声、DiCo 转换时出的问题。不能因为模型全错就改金标。
5. **NEGATIVE_OTHER 的反向标签没有定义。**
   - 官方评分的映射里没有定义 NEGATIVE_OTHER 的反向。据 GPT 查到的 PhrasIS 作者稿表 1，非蕴含类有相反、相似、相关、无关，看起来都是对称关系，所以"反向仍是 NEGATIVE_OTHER"是有依据的假设。但 PhrasIS 第 3 节的标签指南和 DiCo 的映射还没核对。
   - PhrasIS 原数据能否用于训练，以官方最终规则为准。
6. **分组还不够严。**
   - 内部验证按 pair_id 分组，但验证集 176 对里有 11 对的文本和训练部分相同（pair_id 不同）。
   - 训练集和 dev 也有 7 对文本相同。
   - 以后先把文本相同的 pair_id 合成一个家族，再分组。
7. **这一轮错例分析用的是 dev，已经留下了影响。** 以后的错例分析改用训练集上按家族分组的折外预测，dev 只用来报告。

## 下一步计划

每项都和当前最好的设置（mDeBERTa + 对称化）对照，一次只改一处，3 个种子。

- 比较时看同一种子下两种设置的配对差值，不拿单个模型的种子标准差当门槛。
- 要写进结论的比较，补到 5 个种子。

1. **单词替换的诊断**（只做诊断，不选方法）：
   - 数据：在英语训练集上做 5 折折外预测，每个家族只由没见过它的模型预测，轮数在训练折内部选。
   - 做法：同一检查点，比较"输入完整短语"和"只输入被替换的两个词"，列 2×2 配对表（两种都对、删掉上下文后变错、删掉后变对、两种都错）。
   - 解读：两者接近，只能说明这个模型在这个切片上没有用到短语内部的上下文；完整输入明显更好，说明删掉的内容对模型有用。
   - 再从"变错"和"都错"里抽小样本人工看：是词汇关系、搭配、歧义，还是疑似标注问题。
2. **训练配方：6 轮对 12 轮。**
   - 学习率是按总步数线性衰减的，改轮数会同时改变前 6 轮的学习率。所以这是两个训练配方的比较，不是"只改轮数"。
   - 记录实际选中的轮数，并保存检查点。
3. **把现有模型跑全**（导师的要求）：XLM-R large；Qwen 零样本、少样本、CoT；然后用 LoRA 做 SFT。
4. **多语联合训练**：一个模型用三个单语加跨语言的全部训练数据，和逐赛道单独训练比较。
   - 同轮数、同步数两种口径都报。
   - 记录每种语言组合的曝光次数：跨语言数据行数多，直接拼接会让它主导采样。
5. **两个二元头**：同一个有向打分函数分别给 (text1, text2) 和 (text2, text1) 打分，这样对调输入时两个输出自然互换。
   - 注意：两个独立的概率相乘只能表示一部分四类分布，比如"不是 EQUIVALENCE 就是 NEGATIVE_OTHER"这种不确定性就表示不了。所以它是要做消融的假设，不是天然更合适。
6. **NEGATIVE_OTHER 反向增广**：核对 PhrasIS 标签指南之后再做。
   - 要加一组对照：把原方向的 NEGATIVE_OTHER 重复同样次数，排除只是类别权重变化带来的影响。
7. **训练技巧**：R-Drop、SCL、MoCo，逐个加。
8. **DeepSeek 扩充**：生成新的单词替换短语对和 NEGATIVE_OTHER 边界对，按 +1、+2、+4 倍画数据量曲线。
   - 先抽 50 条，人工按 PhrasIS 的口径核对 DeepSeek 打的标签。
   - 和"随机生成的短语对"比较时，条数、标签分布、长度都要匹配，并排除和 dev 近似重复的样本。
9. **反事实推断**：等导师讲解。

## 文件

- `code/baselines.py`：random / majority 基线。
- `code/probes.py`：浅层探针。
- `code/error_analysis.py`：错例、编辑形状、长度规则的统计。
- `code/slice_oracle.py`：切片 oracle。
- `notebooks/dico_encoders.py` / `.ipynb`：编码器实验，可以断点续跑。
- `metrics/baselines/`：基线逐种子分数。
- `metrics/encoders/`：
  - `summary.json`：编码器的汇总；
  - `internal_val_pair_ids.json`：内部验证集的 pair_id，四个赛道相同；
  - 每次运行的 dev 两个方向的 logits、两种预测、逐轮训练日志和官方评分。
- `metrics/probes/summary.json`：探针分数。
- `metrics/analysis/`：错例统计、切片 oracle。

## 来源

- 数据、评分脚本和试点结果来自官方仓库。
- 编码器实验由 Pan-Yan-0 在 GPU 上运行。notebook、基线、探针、错例分析和切片 oracle 的脚本由 Claude 编写，也由 Claude 运行（编码器除外）。
- GPT 复核了第一版分析，指出了几处问题：PhrasIS 脱离语境重新标注、South Africa 冲突的方向、切片上限的算法、二元头的表达能力、增广时类别权重的混杂。本页已按这些意见修改。
- PhrasIS：López-Gazpio et al., *PhrasIS: Phrase Inference and Similarity benchmark*, Logic Journal of the IGPL, doi:10.1093/jigpal/jzae037。
- Glockner, Shwartz, Goldberg. *Breaking NLI Systems with Sentences that Require Simple Lexical Inferences*. ACL 2018. https://aclanthology.org/P18-2103/
