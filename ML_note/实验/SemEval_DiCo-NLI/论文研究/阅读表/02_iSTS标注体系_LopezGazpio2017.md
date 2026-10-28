# Lopez-Gazpio et al. 2017：Interpretable Semantic Textual Similarity，及 SemEval-2015 Task 2 标注指南

Knowledge-Based Systems 2017（arXiv:1612.04868），以及 SemEval-2015 Task 2 的 iSTS 标注指南。PhrasIS 是在这一套数据和标签上重新标注的（DiCo 提案第 2 节）。PhrasIS 论文本身（Logic Journal of the IGPL 32(6)）在两个仓储页上都只有元数据，云端取不到全文，所以用它的前身来理解标注口径。

## 阅读表

| 项目 | 内容 |
|---|---|
| 解决的问题 | 句子相似度只给一个分数，说不清两句哪里一样、哪里不同。iSTS 把两句切成语块、对齐，再给每个对齐标关系类型和 0–5 分。 |
| 核心想法 | 关系类型：EQUI（等价）、OPPO（相反）、SPE1 / SPE2（某一边更具体或更详细）、SIMI（相似：同类但不等价、不更具体、不相反）、REL（相关）、NOALI（没有对应）。按"从强到弱"的顺序对齐和定类型，结合整句语境。 |
| 依赖的假设 | ① 类型能由标注者稳定判断；② "更具体 / 更详细"是一个可以独立于严格逻辑蕴含判断的关系；③ 语块的意义要放在整句里看。 |
| 在我们数据上是否成立 | ① 论文给出的类型一致性只有 .77（新闻标题）/ .81（图片描述），比对齐本身低得多，说明"类型"边界本来就不稳——和我们共同错误多一致。② 成立，而且这正是我们粒度错误的来源：指南例子里 standing ↔ grazing 标成"grazing 更具体"，Red double decker bus ↔ Double decker passenger bus 也算 SPE。我们的单词替换对里，WordNet 下位词被标成 E 的 6 对、复数和数字被标成蕴含的多对，都更符合"更具体 / 更详细"的口径，而不是严格蕴含。③ PhrasIS 是脱离原句重新标注的（AGENTS 记录，GPT 查到），所以这一条到 PhrasIS 已经改了，不能用"丢了语境"解释金标。 |
| 对应哪类错误 | 粒度（E ↔ 单向）、O 边界里的"相似"。 |
| 验证用的小实验和算力 | 不单独跑 GPU 实验。把口径写成生成数据的约束（交回的实验 E3），并在 R2 的人工复核里按"口径 / 分歧 / 疑似错标"三类标注全错行（无 GPU）。 |

## 三遍法笔记

### 第一遍

- 类型：数据集 + 系统论文，同一作者团队（第一作者也是 DiCo 的组织者之一）。
- 结论：类型判断是最难的一步；系统的类型 F1 比对齐 F1 低约 23 个点。

### 第二遍

- **数据**：新闻标题（2013–2014 年的 Europe Media Monitor）和 PASCAL 图片描述，各约 750 对句子；语块平均约 2 个词（标题 1.9、图片 2.25 个词）。这和我们"短语平均 3 个词"一致。
- **标签分布**：EQUI 最多（标题 61%、图片 54%）；SIMI 15–18%；SPE1 / SPE2 各 9–12%；REL 3–5%；OPPO 不到 1%。
- **一致性（表 3，20 对句子）**：对齐 .91 / 1.00，类型 .77 / .81，类型 + 分数 .74 / .77。
- **系统的主要错误**：过度预测 EQUI（类别不平衡）；同义词和命名实体把 EQUI 判成 SPE（"Bloomberg ↔ Michael Bloomberg"）。
- **指南里的具体口径**：
  - 不同数字标 SIMI（"8 ↔ 14"，3 分），不是 OPPO；
  - 南北这种互斥方位标 OPPO（"in southern Iraq ↔ in northern Iraq"）——我们训练集里同一对标 O，一致；
  - "on the porch ↔ on a couch" 是 REL；
  - 有一边多出细节就用 SPE，意思相同就 EQUI，只有都不是时才 SIMI。

### 第三遍

- **复述**：iSTS 的关系类型本质上是"相似度的解释"，SPE 表示"信息量多了"，不要求逻辑上必然推出。PhrasIS 把它改造成推理标签时，保留了这种"更具体"的直觉，并补了附录里的量词、复数口径。
- **和 DiCo 的差别**：数字。iSTS 里 8 ↔ 14 是 SIMI；DiCo 试点表 1 里 "22 dead ↔ 89 dead" 是后向蕴含（大数推出小数）。所以 PhrasIS 在数字上改了口径，我们训练集的计数规则（26 对无反例）应以 PhrasIS / DiCo 为准，不能套 iSTS。
- **对我们的启示**：
  1. 粒度错误不是"模型不懂 WordNet"，而是模型没学到"近义说法、同一实体的不同称呼标 E；有细节差别标单向"这套口径。增加带口径的训练样本，比加外部词典更对路。
  2. 类型一致性 .77–.81 给了一个粗的上限参考：这些边界连专家也有分歧。不应期待 O 边界做到很高。
- **存疑**：PhrasIS 的标注一致性我们没看到原文数字（摘要只说"很高"），不能直接套用 iSTS 的 .77–.81。

## 链接

- 论文：<https://arxiv.org/abs/1612.04868>
- 标注指南：<https://alt.qcri.org/semeval2015/task2/data/uploads/annotation_guidelines_semeval-2015_task2_interpretablests.pdf>
- PhrasIS（只有元数据）：<https://doi.org/10.1093/jigpal/jzae037>，<https://addi.ehu.eus/handle/10810/69010>
