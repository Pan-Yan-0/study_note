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
- 对称化在 8 组里都提高了 SoftCons（+5–10 个点）和 HardCons（+3–6 个点），WF1 提高 0–1.3 个点。
- 英语的 mDeBERTa 已经超过官方试点里 DeBERTa-v3-base 的成绩（.75 / .84 / .79）。但试点只做三个正向标签，两边不能直接比较。
- 赛道 1、3 上，mDeBERTa 的内部验证 F1 到第 6 轮还在上升。

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

西班牙语、巴斯克语的结果在 `metrics/probes/summary.json`，规律相同。

### 错例分析（[code/error_analysis.py](code/error_analysis.py)，英语 dev，mDeBERTa）

- **方向本身基本不出错**：FORWARD 和 BACKWARD 互相混淆只占 1–2%。
- **错误集中在 NEGATIVE_OTHER 的边界**。原方向上，3 个种子多数票错了 92 / 383 对：
  - 金标 NEG 被判成别的：44；
  - 别的被判成 NEG：26；
  - EQUIVALENCE 和单向蕴含互相混淆：19；
  - FORWARD 和 BACKWARD 互相混淆：3。
- **按编辑形状分**（dev 对数，mDeBERTa 准确率）：

  | 形状 | 定义 | dev 对数 | 准确率 |
  |---|---|---:|---:|
  | 插入 | 一侧的词是另一侧的子集 | 93 | .91 |
  | 单词替换 | 两侧各多出 1 个词 | 90 | .74 |
  | 无共同词 | | 70 | .69 |
  | 其他 | | 129 | .75 |

  - 插入类用"词多的一侧推出词少的一侧"这一条规则，就能对 86%。
  - 规则不成立的那部分，模型只有 .56 的准确率，35% 的预测跟着规则走。这部分只占 dev 行数的 3%。
- **短语很短**：平均 3 个词，39% 不超过 2 个词。训练集 1,760 对里，含否定词的 5 对，含量词的 33 对。
- **有 31 对在 6 次运行（2 个模型 × 3 个种子）里全部判错。** 其中一部分金标依赖语境：
  - "over child ‖ over daughter" 金标 NEGATIVE_OTHER；
  - 训练集里 "South Africa ‖ South Africans" 出现了两次，一次标 BACKWARD，一次标 NEGATIVE_OTHER。

## 问题

1. **数据结构会泄露 NEGATIVE_OTHER。**
   - 训练集和 dev 里，NEGATIVE_OTHER 只出现在只有一条记录的短语对上。
   - 官方 trial 里测试格式的文件也一样：只有一条记录的短语对，125 / 125 都是 NEGATIVE_OTHER。
   - 本仓库的模型只用两个短语的文本。要不要告诉主办方，待定。
2. **对称化以后，SoftCons 不能再用来比较方法。**
   - 评分脚本里，可对调的短语对只要有一个方向预测成 NEGATIVE_OTHER，就算不一致。
   - 对称化后两个方向在构造上就互为对调，剩下的不一致只来自"预测成 NEGATIVE_OTHER"。所以只看一侧的探针对称化后 SoftCons 有 .96，比 mDeBERTa 还高。
   - 比较方法时，SoftCons 用原方向的预测来报；对称化后的结果主要看 WF1 和 HardCons。
3. **主要难点是 NEGATIVE_OTHER 的边界和单词替换。**
   - 被替换的词是同义词、上下位词，还是只是相似或对比，决定了标签。
   - dev 的 90 个替换词对里，只有 9 个在训练集出现过，所以这部分靠的是预训练学到的词汇知识。
4. **一部分标签依赖原句语境。**
   - PhrasIS 先把句对切块再对齐标注，标签是在句子语境里打的，而语境没有提供给我们。
   - "6 次全错"的 31 对里，有一部分可能是模型做不到的。
5. **PhrasIS 的原标签集还没核实。**
   - 官方说 NEGATIVE_OTHER 是"可对调标签以外的全部 PhrasIS 标签"。如果这些标签都是对称关系，NEGATIVE_OTHER 的反向仍是它自己，可以用来增广训练数据；没核实之前不用。
   - PhrasIS 原数据能否用于训练，以官方最终规则为准。
6. **这一轮错例分析用的是 dev。** 以后改用训练集上按 pair_id 分组的交叉验证预测，dev 只用来报告。
7. **训练集和 dev 有 7 对短语文本完全相同**（pair_id 不同），做"未见组合"检查时要排除。

## 下一步计划

每项都和当前最好的设置（mDeBERTa + 对称化）对照，一次只改一处，3 个种子。

1. **轮数**：6 → 12。依据：第 6 轮的内部验证 F1 还在上升。
2. **单词替换的诊断**：用同一组检查点，比较"输入完整短语"和"只输入被替换的两个词"。
   - 两者差不多：说明标签几乎由词对决定，优先走词汇知识路线（更大的预训练模型、大模型、按词对关系扩充数据）。
   - 只输入词对时明显变差：说明上下文起作用，再做针对上下文的最小对照。
3. **把现有模型跑全**（导师的要求）：XLM-R large；Qwen 零样本、少样本、CoT；然后用 LoRA 做 SFT。
4. **多语联合训练**：一个模型用三个单语加跨语言的全部训练数据，和逐赛道单独训练比较。
5. **两个二元头**：分别判断"text1 能推出 text2 吗""text2 能推出 text1 吗"，和四分类头比较。
6. **训练技巧**：R-Drop、SCL、MoCo，逐个加。
7. **DeepSeek 扩充**：生成新的单词替换短语对和 NEGATIVE_OTHER 边界对，按 +1、+2、+4 倍画数据量曲线。先抽 50 条，人工核对 DeepSeek 打的标签和金标的口径是否一致。
8. **反事实推断**：等导师讲解。

## 文件

- `code/baselines.py`：random / majority 基线。
- `code/probes.py`：浅层探针。
- `code/error_analysis.py`：错例、编辑形状、长度规则的统计。
- `notebooks/dico_encoders.py` / `.ipynb`：编码器实验，可以断点续跑。
- `metrics/baselines/`：基线逐种子分数。
- `metrics/encoders/`：编码器的汇总（`summary.json`），以及每次运行的 dev 预测、训练日志和官方评分。
- `metrics/probes/summary.json`：探针分数。
- `metrics/analysis/error_analysis.json`：错例统计。

## 来源

- 数据、评分脚本和试点结果来自官方仓库。
- 编码器实验由 Pan-Yan-0 在 GPU 上运行。notebook、基线、探针和错例分析的脚本由 Claude 编写。基线、探针和错例分析由 Claude 运行。
- PhrasIS：López-Gazpio et al., *PhrasIS: Phrase Inference and Similarity benchmark*, Logic Journal of the IGPL, doi:10.1093/jigpal/jzae037。
