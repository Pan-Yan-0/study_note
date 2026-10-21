# DiCo-NLI：句对输入、方向与一致性 学习笔记

第二章跟着 SemEval-2027 Task 2（DiCo-NLI）的实验学。这一份讲赛题本身、模型怎么分清两段短语，以及两个一致性指标和对称化。实验结果见 [SemEval_DiCo-NLI 实验](../../实验/SemEval_DiCo-NLI/README.md)。

## 1. 赛题在做什么

给一对短语，判断它们之间是哪种"推出"关系。两个方向都要判，两个方向的答案还要对得上。

| 步骤 | 内容 |
|---|---|
| 数据 | 每一行是一个有序的短语对 (text1, text2)，带一个标签 |
| 模型 | 读入 `[CLS] text1 [SEP] text2 [SEP]`，输出 4 个类别的概率 |
| 输出 | 每一行一个标签 |
| 评分 | Weighted F1、SoftCons、HardCons |

**四个标签**（训练集里的例子，最后一行来自 dev）

| text1 | text2 | 标签 | 意思 |
|---|---|---|---|
| A collie | A dog | FORWARD | 由 text1 能推出 text2：牧羊犬一定是狗 |
| A dog | A collie | BACKWARD | 由 text2 能推出 text1 |
| of guys | of men | EQUIVALENCE | 两边能互相推出 |
| A red double decker bus | A white bus | NEGATIVE_OTHER | 两个方向都推不出 |

直觉上，FORWARD 就是 text1 更具体、text2 更笼统。

**两个方向**：除 NEGATIVE_OTHER 外，同一对短语在数据里出现两次，第二次把顺序对调，两行的标签绑在一起。NEGATIVE_OTHER 只出现一次。

| 第一次 | 对调后 |
|---|---|
| FORWARD | BACKWARD |
| BACKWARD | FORWARD |
| EQUIVALENCE | EQUIVALENCE |

**三个指标**

| 指标 | 看什么 |
|---|---|
| Weighted F1 | 每一行判得对不对：按类别算 F1，再按各类的行数加权平均 |
| SoftCons | 一对短语的两行答案能不能互相对上（FE 对 BE、BE 对 FE、EQ 对 EQ），不管对错；有一行答 NEGATIVE_OTHER 就算对不上 |
| HardCons | 一对短语的两行是否都答对 |

**四个赛道**：同一批短语对，分别出英语、西班牙语、巴斯克语版，另有一个混合版（例如 text1 是英语、text2 是西班牙语）。

## 2. 模型怎么分清 A 和 B

BERT 的输入是三个向量逐位相加：

| 向量 | 作用 |
|---|---|
| 词向量 | 这个 token 是什么词 |
| 位置向量 | 它在第几个位置 |
| 段向量（segment / token_type） | 它属于 A（0）还是 B（1） |

- mBERT 有段向量。mDeBERTa 的配置里没有（type_vocab_size 是 0），只能靠 `[SEP]` 和相对位置分清两段。
- 把 A、B 对调后，同一个词的词向量不变，位置向量变了，段向量也变了。在 mBERT 里，段向量是"属于前提还是假设"最直接的信号，位置向量也带着方向信息。
- 不是某一个向量单独"看出"方向：三个向量加起来进入第 1 层，注意力让两段的 token 互相交流，最后由 [CLS] 汇总出判断。
- 模型学得会"对调后标签要变"，是因为训练集里每对可对调的短语都有正反两条，标签正好相反，逼着模型去利用顺序。

## 3. 不看顺序的模型与 SoftCons

假设一个"词袋"模型，只看两段里出现了哪些词，不看顺序，也不管词来自 A 还是 B。(A, B) 和 (B, A) 放进袋子以后完全一样，所以两行的预测一定相同。

- 两行答案相同时，只有 EQUIVALENCE 对调后还是它自己，所以 SoftCons 只有"两行都答 EQ"这一种一致的答法。
- 金标是 FE / BE 的一对：要一致就只能答 EQ，答错，HardCons 拿不到。
- 金标是 EQ 的一对：两行都答 EQ，一致且答对。
- 一个所有可对调的对都答 EQ 的模型，SoftCons 是 100%，但它完全不懂方向。

**和实验的对应**：只看一侧短语的探针加上对称化，SoftCons 有 0.96，比 mDeBERTa 的 0.915 还高。所以比较模型的一致性时，看原方向（不对称化）的 SoftCons，再加上 HardCons。

## 4. 对称化

真正的模型对调输入后看到的东西不同（位置、段都变了），所以两个方向会给出两份不一样的概率。对称化把两份合成一份：

1. 取 p(B, A)，把其中 FE、BE 两个数互换，变成站在 (A, B) 角度的概率。
2. 和 p(A, B) 取平均，作为 (A, B) 这一行的最终概率。

例子（概率顺序是 [EQ, FE, BE, NEG]）：

| 步骤 | 概率 |
|---|---|
| p(A, B) | [.2, .5, .1, .2] |
| p(B, A) | [.3, .1, .4, .2] |
| p(B, A) 互换 FE、BE | [.3, .4, .1, .2] |
| 两者平均 = (A, B) 的最终概率 | [.25, .45, .10, .20] → FE |

(B, A) 这一行用同样的办法，得到 [.25, .10, .45, .20] → BE。两行在构造上互为对调，所以 FE / BE 之间一定一致；但如果最大的是 NEGATIVE_OTHER，这一对仍然算不一致。代价是推理时每行要多算一次前向。

## 5. 检查题与我的作答

✓ 答对；△ 部分对或经提示后改对；✗ 答错后已更正；— 还没作答（参考答案折叠在下面）。

| # | 题目 | 我的作答 | 判定与要点 |
|---|---|---|---|
| 1 | A = "Conservatives"、B = "Conservative opposition" 对调后喂给 mBERT："Conservatives" 的词向量、位置向量、段向量变不变；模型靠哪些看出方向变了 | 词向量不变，位置向量变了，段向量变了；靠位置向量 | △ 前三问对。mBERT 里段向量是最直接的信号，位置向量也带信息；mDeBERTa 没有段向量，靠 [SEP] 和相对位置 |
| 2 | 金标 (collie, dog) FE、(dog, collie) BE；模型甲 FE / BE、乙 BE / FE、丙 FE / EQ：各自的 SoftCons 和 HardCons | 甲、乙 SoftCons 一致，丙不一致；HardCons 甲对，乙全错，丙对一半但仍不算对 | ✓ 乙两行都错却算一致：SoftCons 只看能不能对上 |
| 3 | 词袋模型两行答案相同，哪个标签能让 SoftCons 一致；collie / dog 这一对的 HardCons 对不对 | 只有 EQUIVALENCE；错 | ✓ |
| 4 | 金标 EQ 的一对，词袋模型两行都答 EQ：SoftCons、HardCons；所有可对调的对都答 EQ 的模型，SoftCons 多少，懂不懂方向 | 一致、对、100%；不懂，只是因为 A、B 在同一个词袋里，同一个词袋就直接判定了 | ✓ |

还没作答的题：

<details>
<summary>p(A, B) = [.1, .3, .1, .5]，p(B, A) = [.2, .1, .6, .1]（顺序 [EQ, FE, BE, NEG]）：不对称化时两行各答什么、SoftCons 一致吗；对称化后 (A, B) 的概率和答案；(B, A) 答什么、为什么</summary>

- 不对称化：(A, B) 答 NEGATIVE_OTHER，(B, A) 答 BACKWARD；有一行是 NEGATIVE_OTHER，不一致。
- 对称化：p(B, A) 互换 FE、BE 得 [.2, .6, .1, .1]，和 [.1, .3, .1, .5] 平均得 [.15, .45, .10, .30] → FORWARD。
- (B, A) 答 BACKWARD：它的最终概率是同样两份数换个角度平均，正好是 (A, B) 的 FE、BE 互换。
</details>

## 6. 易混点

- **SoftCons 与 HardCons**：前者只看两行能不能对上，后者要两行都答对。一致不等于对。
- **对称化后的 SoftCons**：FE / BE 之间在构造上一定一致，剩下的不一致只来自答 NEGATIVE_OTHER，所以它不能用来比较模型懂不懂方向。
- **只有 EQUIVALENCE 对调后不变**；FE 和 BE 互换；NEGATIVE_OTHER 只出现一次，没有对调的那一行。
- **段向量**：mBERT 有，mDeBERTa 没有；两者都还有位置信息。

## 7. 下一步

- 对称化那道题。
- R-Drop 与"方向版 R-Drop"：让两次前向或两个方向的分布接近，要用到 KL。
- 有监督对比学习（SCL）。
- 反事实与因果推断，等导师讲。

## 来源

- Pan-Yan-0 与 Claude 的问答；第 5 节"我的作答"是 Pan-Yan-0 回答的意思。
- 赛题、标签和指标来自 [SemEval-2027 Task 2 官方说明](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)。
- 探针和 mDeBERTa 的数字来自本仓库的 [SemEval_DiCo-NLI 实验](../../实验/SemEval_DiCo-NLI/README.md)。
