# IMDB 情感分析实验

Kaggle 比赛 *Bag of Words Meets Bags of Popcorn*（`word2vec-nlp-tutorial`）：用 25,000 条有标签影评训练正负情感二分类，对 25,000 条测试影评提交预测，Kaggle 按测试集 ROC-AUC 评分。这里有两组实验：GloVe 词向量加六种网络，以及三个预训练模型（BERT 全量、BERT LoRA、DeBERTa-v2-xxlarge LoRA）。

## 文件

| 文件 | 内容 |
|---|---|
| [notebooks/imdb-glove-six-models-2026-09-12.ipynb](notebooks/imdb-glove-six-models-2026-09-12.ipynb) | 六模型的代码与完整运行输出 |
| [notebooks/imdb-three-models-official-trainer.ipynb](notebooks/imdb-three-models-official-trainer.ipynb) | 三个预训练模型依次训练（Transformers Trainer + PEFT），含 Kaggle 运行输出 |
| [notebooks/imdb-bert-deberta-lora.ipynb](notebooks/imdb-bert-deberta-lora.ipynb)、[code/imdb_pretrained.py](code/imdb_pretrained.py) | 更早的预训练版本（手动选模型 + 共用训练代码），对应第 3 节的 BERT 日志 |
| [code/requirements-pretrained.txt](code/requirements-pretrained.txt) | 预训练实验的依赖版本 |
| [metrics/model_summary.csv](metrics/model_summary.csv)、[metrics/training_history.csv](metrics/training_history.csv) | 六模型的汇总与 120 条逐轮指标 |
| [metrics/pretrained_three_models_summary.csv](metrics/pretrained_three_models_summary.csv)、[metrics/pretrained_three_models_epochs.csv](metrics/pretrained_three_models_epochs.csv) | 三个预训练模型的汇总与逐轮验证指标 |
| [metrics/bert_full_kaggle_report_2026-10-11.csv](metrics/bert_full_kaggle_report_2026-10-11.csv) | 更早一次 BERT 全量的三轮日志 |
| [metrics/kaggle_submissions.csv](metrics/kaggle_submissions.csv) | 全部 18 条 Kaggle 提交记录与版本链接 |

相关笔记：[BERT 原理笔记](../BERT/BERT原理笔记.md)、[六模型实战与概念问答](../IMDB六模型实战与概念问答.md)、[DeBERTa 与 IMDB 三条路学习笔记](../DeBERTa/DeBERTa与IMDB三条路学习笔记.md)。与 LoRA 论文、GMCM 实验的统一对照见[实验结果汇总](../实验结果汇总.md)。

## 结果速览

| 模型 | 验证 accuracy | 验证 AUC | Kaggle AUC | 提交内容 |
|---|---:|---:|---:|---|
| BiGRU（六模型最好） | 0.9031（batch 平均，第 8 轮） | — | 0.89996 | 0/1 标签 |
| BERT 全量微调 | 0.9206 | 0.974062 | 0.97522 | 正类概率 |
| BERT LoRA | 0.8766 | 0.949744 | 0.95315 | 正类概率 |
| DeBERTa-v2-xxlarge LoRA | 0.9334 | 0.979953 | **0.98294** | 正类概率 |

提交 0/1 标签时，ROC-AUC 等于（真正率 + 真负率）/ 2，接近准确率；提交概率时才是排序意义上的 AUC。所以两组的 Kaggle 分数不能直接相减，两组的验证划分也不同。

## 1. GloVe 词向量 + 六种网络

| 条件 | 设置 |
|---|---|
| 数据 | 有标签影评按 80% / 20% 划分训练与验证（`random_state=0`，不分层） |
| 文本处理 | 去 HTML，只保留英文字母，小写分词；词表 101,399 个词（来自全部有标签文本和测试文本）；截断或补齐到 512 个词 |
| 词向量 | GloVe 840B 300 维，六个模型都冻结；没有匹配的词为零向量 |
| 训练 | batch 64，Adam，学习率 0.001，交叉熵，每个模型 20 轮 |
| 预测 | 第 20 轮模型，`argmax` 导出 0/1 标签 |

| 模型 | 结构 | 最高验证（轮次） | 第 20 轮验证 | Kaggle AUC：Version 2 / 修复 AttLSTM / epoch=20 |
|---|---|---|---:|---|
| CNN | 128 个宽 3 的卷积核，ReLU，全局最大池化 | 0.8977（7、10、11、15、20） | 0.8977 | 0.88880 / — / 0.89088 |
| BiLSTM | 2 层双向 LSTM，隐藏 128，拼接首尾时间步 | 0.8995（10） | 0.8916 | 0.89128 / — / 0.89368 |
| BiGRU | 2 层双向 GRU，隐藏 128，拼接首尾时间步 | 0.9031（8） | 0.8942 | **0.89996** / — / 0.89320 |
| Attention-BiLSTM | 双向 LSTM 后按注意力权重沿时间加权求和 | 0.9021（4、5） | 0.8948 | 0.50004 / 0.89660 / 0.89388 |
| CNN-BiLSTM | 卷积提取局部特征后接双向 LSTM | 0.8983（8） | 0.8756 | 0.88580 / — / 0.87208 |
| TransformerEncoder | 300→128 投影，2 层编码器、8 头，序列平均池化 | 0.8869（4） | 0.8661 | 0.86944 / — / 0.85216 |

- 验证准确率是 Notebook 打印的 batch 平均值（每个 batch 等权，末尾不满 64 条的 batch 也一样），和按样本算的准确率略有差别。代码没有保存最佳检查点，导出的都是第 20 轮模型。
- Version 2 的 Attention-BiLSTM 为 0.50004（随机水平），修复后为 0.89660。提交列表另有两条未标模型的记录：Version 2 的 0.88580、Version 1 的 0.88400。
- 后期训练准确率上升而验证回落（BiGRU 验证从第 8 轮 0.9031 降到第 20 轮 0.8942，训练从 0.9723 升到 0.9947），20 轮版本的 Kaggle 分数也没有普遍更高，符合过拟合。只有单次运行，不同版本之间代码和随机状态也可能有变化。
- 已知问题：循环网络没有按真实长度 pack，注意力和平均池化都包含 padding；TransformerEncoder 没有位置编码和 padding mask，它是随机初始化的学习用编码器，不是 BERT；只固定了划分种子；词表包含了验证和测试文本里的词。

## 2. 三个预训练模型（Transformers Trainer）

一次 Run All 依次训练三个模型，三组共用同一份划分和训练设置：

| 条件 | 设置 |
|---|---|
| 数据 | 有标签影评分层划分（`random_state=42`）：20,000 训练 / 5,000 验证；只去 HTML，保留大小写和标点，交给各模型自己的 tokenizer |
| 训练 | 3 轮；AdamW，学习率 2e-5，weight decay 0.01，线性衰减，前 10% 步数 warmup；FP16 混合精度；有效 batch 16 |
| 选模型 | 每轮验证，按验证 AUC 选最佳轮次，权重在内存中保存，训练结束后恢复 |
| LoRA | r = 8，α = 16，dropout 0.05 |
| 提交 | `submission.csv` 为正类概率；另存阈值 0.5 的 `test_labels.csv` |
| 环境 | Kaggle 单张 Tesla T4；torch 2.11.0、transformers 5.19.0、peft 0.21.2、accelerate 1.15.0、datasets 5.1.0 |

| 模型 | 训练方式 | 长度 / micro batch × 累积 | 可训练参数 | 最佳轮次 | 验证 accuracy / AUC | Kaggle AUC | 显存峰值 | 用时 |
|---|---|---|---:|---:|---|---:|---:|---:|
| BERT 全量 | `bert-base-uncased` 全部参数 | 256 / 8 × 2 | 109,483,778（100%） | 2 | 0.9206 / 0.974062 | 0.97522 | 2.69 GiB | 18 分钟 |
| BERT LoRA | query、value 加 LoRA，训练分类头 | 256 / 8 × 2 | 296,450（0.27%） | 3 | 0.8766 / 0.949744 | 0.95315 | 1.24 GiB | 14 分钟 |
| DeBERTa-v2-xxlarge LoRA | FP16 冻结底座，query_proj、value_proj 加 LoRA，训练 pooler 和分类头，开梯度检查点 | 128 / 16 × 1 | 4,723,202（0.30%） | 3 | 0.9334 / 0.979953 | **0.98294** | 4.44 GiB | 2.3 小时 |

逐轮验证（训练 loss 是 Trainer 每 100 步记录的最近一次值）：

| 模型 | 轮次 | 训练 loss | 验证 loss | 验证 accuracy | 验证 AUC |
|---|---:|---:|---:|---:|---:|
| BERT 全量 | 1 | 0.2601 | 0.2328 | 0.9120 | 0.971985 |
| | 2 | 0.1627 | 0.2525 | 0.9206 | **0.974062** |
| | 3 | 0.0955 | 0.3468 | 0.9212 | 0.973945 |
| BERT LoRA | 1 | 0.3567 | 0.3355 | 0.8594 | 0.935947 |
| | 2 | 0.3157 | 0.3034 | 0.8744 | 0.947399 |
| | 3 | 0.2879 | 0.2989 | 0.8766 | **0.949744** |
| DeBERTa-v2-xxlarge LoRA | 1 | 0.2221 | 0.2213 | 0.9274 | 0.975361 |
| | 2 | 0.1988 | 0.1949 | 0.9338 | 0.979054 |
| | 3 | 0.2141 | 0.1969 | 0.9334 | **0.979953** |

- **BERT 全量**：第 2 轮之后训练 loss 继续下降、验证 loss 上升（0.2525 → 0.3468），开始过拟合，按 AUC 选回了第 2 轮。
- **BERT LoRA 对全量**：同底座、同划分、同长度、同学习率，只差训练方式。LoRA 只训练 0.27% 的参数，显存不到全量的一半，但验证 AUC 低 0.024，Kaggle AUC 低 0.022。它的验证 AUC 到第 3 轮仍在上升；学习率沿用了全量微调的 2e-5，而 LoRA 论文 GLUE 实验给 RoBERTa-base 的 LoRA 用 4e-4 到 5e-4。学习率偏小、训练不足是候选解释，需要加大学习率或轮数来验证。
- **DeBERTa-v2-xxlarge LoRA**：分数最高，三轮内没有明显过拟合。它和 BERT 的底座、分词方式、截断长度（128 对 256）同时不同，差距不能只归因于 LoRA。
- 三个模型的 Kaggle AUC 都比验证 AUC 略高（+0.001 到 +0.003），两者一致。
- 可训练参数与 LoRA 公式完全对上：BERT 为 12 层 × 2 × 8 × (768 + 768) = 294,912，加分类头 1,538；DeBERTa 为 48 层 × 2 × 8 × (1536 + 1536) = 2,359,296，加 pooler 2,360,832 和分类头 3,074。

这一版 Notebook 比上一版改了三处才在 Kaggle 跑通：彻底清理 Kaggle 预装 torchao 的残留元数据（否则 PEFT 版本检查报错）；最佳权重改为在内存中保存和恢复（从磁盘回载 BERT 检查点时，LayerNorm 的 gamma/beta 命名转换会让这部分权重漏载）；DeBERTa 从 batch 1 × 累积 16 改为 batch 16 × 累积 1，有效 batch 不变，Notebook 记录 batch 1 时慢十倍左右。

## 3. 更早一次 BERT 全量（共用代码版本）

用[单模型 Notebook](notebooks/imdb-bert-deberta-lora.ipynb)和[共用代码](code/imdb_pretrained.py)跑的 BERT 全量微调：不分层划分（`random_state=0`），没有学习率调度，长度 256、batch 8 × 累积 2、学习率 2e-5；日志从 Kaggle 输出粘贴，没有提交分数。

| 轮次 | 训练 loss | 训练 accuracy | 验证 loss | 验证 accuracy | 验证 AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.288101 | 0.88305 | 0.228602 | 0.9090 | 0.972725 |
| 2 | 0.171451 | 0.94770 | 0.250490 | 0.9140 | 0.973270 |
| 3 | 0.105459 | 0.97265 | 0.350316 | 0.9128 | **0.973673** |

和第 2 节的 BERT 全量接近（验证 AUC 0.9737 对 0.9741），第 2 轮之后验证 loss 上升的趋势也一样。显存峰值 allocated 2.73 GiB / reserved 2.88 GiB。

## 来源

- 六模型 Notebook：数据处理在第 10—28 个单元，训练与导出函数在第 36、37 个，六个模型在第 39—44 个（从 1 开始计数）。
- [Kaggle 比赛](https://www.kaggle.com/c/word2vec-nlp-tutorial)，[评分说明](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/evaluation)为 ROC 曲线下面积；[GloVe](https://nlp.stanford.edu/projects/glove/)，Notebook 用的是 `glove.840B.300d.txt`。数据、GloVe 文件和模型权重不在仓库里。
- Kaggle 分数来自本人提供的提交列表，每条的版本链接见 `metrics/kaggle_submissions.csv`。
