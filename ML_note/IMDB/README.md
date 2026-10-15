# IMDB 情感分析实验

Kaggle 比赛 *Bag of Words Meets Bags of Popcorn*（`word2vec-nlp-tutorial`）：用 25,000 条有标签影评训练正负情感二分类，对 25,000 条测试影评提交预测，Kaggle 按测试集 ROC-AUC 评分。这里有三组实验：GloVe 词向量加六种网络；三个预训练编码器（BERT 全量、BERT LoRA、DeBERTa-v2-xxlarge LoRA）；Qwen3-4B-Base 指令微调。

## 文件

| 文件 | 内容 |
|---|---|
| [notebooks/imdb-glove-six-models-2026-09-12.ipynb](notebooks/imdb-glove-six-models-2026-09-12.ipynb) | 六模型的代码与完整运行输出 |
| [notebooks/imdb-three-models-official-trainer.ipynb](notebooks/imdb-three-models-official-trainer.ipynb) | 三个预训练模型依次训练（Transformers Trainer + PEFT），含 Kaggle 运行输出 |
| [notebooks/imdb-qwen3-4b-sft.ipynb](notebooks/imdb-qwen3-4b-sft.ipynb) | Qwen3-4B-Base 指令微调（LoRA，只在答案 token 上算损失），含 Colab 运行输出，见第 4 节 |
| [code/qwen3_sft_report.py](code/qwen3_sft_report.py) | 从 `metrics/qwen3_4b_sft/` 复算第 4 节的数字 |
| [notebooks/imdb-bert-deberta-lora.ipynb](notebooks/imdb-bert-deberta-lora.ipynb)、[code/imdb_pretrained.py](code/imdb_pretrained.py) | 更早的预训练版本（手动选模型 + 共用训练代码），对应第 3 节的 BERT 日志 |
| [code/requirements-pretrained.txt](code/requirements-pretrained.txt) | 预训练实验的依赖版本 |
| [metrics/model_summary.csv](metrics/model_summary.csv)、[metrics/training_history.csv](metrics/training_history.csv) | 六模型的汇总与 120 条逐轮指标 |
| [metrics/pretrained_three_models_summary.csv](metrics/pretrained_three_models_summary.csv)、[metrics/pretrained_three_models_epochs.csv](metrics/pretrained_three_models_epochs.csv) | 三个预训练模型的汇总与逐轮验证指标 |
| [metrics/bert_full_kaggle_report_2026-10-11.csv](metrics/bert_full_kaggle_report_2026-10-11.csv) | 更早一次 BERT 全量的三轮日志 |
| [metrics/qwen3_4b_sft/](metrics/qwen3_4b_sft/) | Qwen3-4B 的汇总 `summary.json`、配置、训练日志、验证集逐条概率（训练前与训练后）、提交文件 |
| [metrics/kaggle_submissions.csv](metrics/kaggle_submissions.csv) | 全部 19 条 Kaggle 提交记录与版本链接 |

相关笔记：[BERT 原理笔记](../BERT/BERT原理笔记.md)、[六模型实战与概念问答](../IMDB六模型实战与概念问答.md)、[DeBERTa 与 IMDB 三条路学习笔记](../DeBERTa/DeBERTa与IMDB三条路学习笔记.md)、[解码器大模型与路线 C 学习笔记](../LLM/解码器大模型与路线C学习笔记.md)。与 LoRA 论文、GMCM 实验的统一对照见[实验结果汇总](../实验结果汇总.md)。

## 结果速览

| 模型 | 验证 accuracy | 验证 AUC | Kaggle AUC | 提交内容 |
|---|---:|---:|---:|---|
| BiGRU（六模型最好） | 0.9031（batch 平均，第 8 轮） | — | 0.89996 | 0/1 标签 |
| BERT 全量微调 | 0.9206 | 0.974062 | 0.97522 | 正类概率 |
| BERT LoRA | 0.8766 | 0.949744 | 0.95315 | 正类概率 |
| DeBERTa-v2-xxlarge LoRA | 0.9334 | 0.979953 | 0.98294 | 正类概率 |
| Qwen3-4B-Base 零样本（不训练） | 0.9440 | 0.984729 | — | — |
| Qwen3-4B-Base 指令微调（LoRA） | 0.9686 | **0.994138** | **0.99493** | 正类概率 |

提交 0/1 标签时，ROC-AUC 等于（真正率 + 真负率）/ 2，接近准确率；提交概率时才是排序意义上的 AUC。所以两组的 Kaggle 分数不能直接相减，两组的验证划分也不同。第 2 节和第 4 节用的是同一份验证集，验证指标可以直接对照。

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

## 4. Qwen3-4B 指令微调

把影评套进 Alpaca 指令模版，让解码器模型在 `### Response:` 之后接着写 `positive` 或 `negative`。损失只算答案这一个 token，提示词和影评部分都不计损失。

| 条件 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen3-4B-Base`，fp16 冻结底座 + LoRA（r = 16，α = 32，q/k/v/o/gate/up/down 七个投影层，33,030,144 个参数，占 0.81%） |
| 数据 | 与第 2 节相同的分层划分（`random_state=42`）：20,000 训练 / 5,000 验证 |
| 截断 | 超过 512 个 token 的影评保留开头 128 + 结尾 384 个 token，模版本身不截断；约 12% 的影评被截断，加上模版平均 315 个 token |
| 训练 | 1 轮（1,250 步），有效 batch 16（8 × 累积 2），AdamW，学习率 1e-4，前 5% warmup 后线性衰减，FP16，梯度检查点 |
| 损失 | 答案位置在整个词表上的交叉熵，标签为 `positive` / `negative` 对应的 token |
| 预测 | 读答案位置上两个词的 logit：p(positive) = softmax(logit_pos, logit_neg)，提交概率 |
| 对照 | 训练前先在同一验证集上测一次零样本（LoRA 初始时 B = 0，等于原模型） |
| 环境 | Colab 单张 Tesla T4；torch 2.11.0、transformers 5.19.0、peft 0.21.2；模型的 Hub 版本号没有记录下来 |

### 结果

| | 验证 accuracy | 验证 AUC [95% 区间] | 答案位置的交叉熵 | 两个答案词的概率之和 | 最可能的下一个 token 是答案词 |
|---|---:|---:|---:|---:|---:|
| 零样本（训练前） | 0.9440 | 0.9847 [0.9816, 0.9877] | 0.999 | 0.436 | 88.6% |
| 指令微调后 | 0.9686 | 0.9941 [0.9924, 0.9957] | 0.093 | 0.9998 | 100% |

Kaggle AUC **0.99493**（提交文件 `metrics/qwen3_4b_sft/submission.csv.gz`），比验证 AUC 高 0.0008，与第 2 节三个模型 Kaggle 略高于验证的情况一致；比 DeBERTa-v2-xxlarge LoRA 的 0.98294 高 0.012。区间是对验证集按条重抽样 2,000 次得到的；本节数字可用 `code/qwen3_sft_report.py` 复算。

- **零样本已经很强**：不训练，验证 AUC 就有 0.9847，高于第 2 节最好的 DeBERTa-v2-xxlarge LoRA（0.9800）。但两个答案词平均只占答案位置 43.6% 的概率；11.4% 的影评，模型最想写的下一个 token 不是这两个词，这些影评的准确率只有 71.4%（其余 97.4%）。只比较两个词的概率，排序仍然很准。
- **微调后**：两个答案词几乎占满答案位置的概率。错的从 280 条降到 157 条（改对 162 条、改错 39 条，McNemar 精确检验 p ≈ 5×10⁻¹⁹），AUC 提高 0.0094 [0.0074, 0.0117]。
- **训练曲线**：第 10 步 loss 0.835，第 20 步 0.200（这时只见过 320 条）；第 51–200 步平均 0.117，最后 250 步平均 0.094，梯度范数从 2.36 降到 0.65。答案格式在最初几十步就学会了，之后是慢慢提高判断。验证集答案位置的交叉熵 0.093 与最后阶段的训练 loss 相当，一轮之内没有过拟合的迹象。
- **校准**：Brier 0.043 → 0.025，ECE（10 个分箱）0.018 → 0.008；但微调后有 12 条以 ≥ 0.99 的置信度判错，零样本是 5 条。
- **不能只归因于训练方法的地方**：和 DeBERTa 相比，底座（40 亿参数的解码器对 15 亿参数的编码器）、输入长度（最多 512 个影评 token 对 128 个）和损失形式同时不同。Qwen3 的预训练语料可能包含 IMDB 影评（这个数据集 2011 年就公开了），零样本表现好可能部分来自见过这些文本，无法排除。只有一次运行、一个种子、一轮，超参数没有调；验证集没有参与选模型。
- **用时与显存**：分词 2.6 分钟，零样本验证 14.5 分钟，训练 222 分钟（10.7 秒/步，约 473 token/秒），微调后验证 17.4 分钟，测试 87 分钟，合计约 5.7 小时（不含下载和加载模型）；显存峰值 10.44 GiB。按每个 token 的计算量折算，T4 实际跑到约 10–13 TFLOPS（训练约 10），低于事前估计时假设的 12–20，所以总时长比估计的 3.5–5.5 小时略长。

### 与导师脚本（Unsloth + TRL `SFTTrainer`）的区别

| 项目 | 导师脚本 | 本 Notebook |
|---|---|---|
| 损失范围 | 整段文本（含影评与模版）都算损失 | 只算答案 token |
| 底座精度 | Unsloth 4-bit | fp16（T4 显存够，沿用第 2 节已跑通的 transformers 5.19 + PEFT 环境，不额外装 Unsloth / bitsandbytes） |
| 预测 | `generate` 3 个 token 后做字符串匹配，解析失败记为 0 | 直接读两个答案词的概率，不生成 |
| 截断 | 整段 2048 token，从右侧截断，超长时会切掉 `### Response:` | 只截影评，模版完整保留 |
| 划分 | `random_state=3407`，不分层 | 与第 2 节相同 |
| 轮数 / 学习率 | 3 轮 / 2e-5 | 1 轮 / 1e-4（第 2 节 BERT LoRA 用 2e-5 时验证 AUC 到第 3 轮仍在上升） |

### 运行方式

同一个 Notebook 在 Colab 和 Kaggle 上都能跑：

- Colab：运行时类型选 T4 GPU。把比赛的 `labeledTrainData.tsv.zip`、`testData.tsv.zip`、`sampleSubmission.csv` 放进 Google Drive 的 `MyDrive/word2vec-nlp-tutorial/`，否则 Notebook 用 `kagglehub` 自己下载（Colab「密钥」里有 `KAGGLE_USERNAME` / `KAGGLE_KEY` 时直接下载，没有就弹出登录框，登录后重新运行第一个单元）。输出写在 `MyDrive/imdb_qwen3_sft/`。
- Kaggle：选 GPU T4 x2（只用一张）并打开网络，添加比赛数据 `word2vec-nlp-tutorial`，输出写在 `/kaggle/working/`。
- 先把 `SMOKE = True` 跑一遍查错，再改回 `False` 正式跑。
- 断开后重新从头运行全部单元即可续跑：已完成的阶段直接跳过，训练从最近的检查点（每 250 步一个）继续，推理从缓存继续。配置改了会报错，需要换 `RUN_NAME`。

输出目录里有 `summary.json`（训练前后验证指标、用时、显存、每次会话的环境）、验证集逐条概率、`submission.csv`（正类概率）和 LoRA 权重 `adapter/`（约 130 MB，不在仓库里）。

## 来源

- 六模型 Notebook：数据处理在第 10—28 个单元，训练与导出函数在第 36、37 个，六个模型在第 39—44 个（从 1 开始计数）。
- [Kaggle 比赛](https://www.kaggle.com/c/word2vec-nlp-tutorial)，[评分说明](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/evaluation)为 ROC 曲线下面积；[GloVe](https://nlp.stanford.edu/projects/glove/)，Notebook 用的是 `glove.840B.300d.txt`。数据、GloVe 文件和模型权重不在仓库里。
- Kaggle 分数来自本人提供的提交列表，每条的版本链接见 `metrics/kaggle_submissions.csv`。
- Qwen3-4B Notebook 由 Claude 根据导师的 Unsloth 脚本改写，Pan-Yan-0 在 Colab 单张 T4 上运行；导师脚本不在仓库里。
