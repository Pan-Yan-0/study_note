# IMDB：三个模型一次 Run All

使用 [新版 Notebook](notebooks/imdb-three-models-official-trainer.ipynb)，按顺序自动跑完：

1. **BERT 全量微调**：`bert-base-uncased`。
2. **BERT LoRA**：相同 BERT 底座，训练 LoRA 和 classifier。
3. **DeBERTa-v2-xxlarge LoRA**：`microsoft/deberta-v2-xxlarge`，训练 LoRA、pooler 和 classifier。

全部从公开预训练 checkpoint 重新初始化，每组默认 **3 轮、完整数据**；不需要切换 `MODEL_KEY`，不加载此前的训练结果。训练使用官方 Transformers `Trainer` / `TrainingArguments`、PEFT、Datasets、PyTorch 和 scikit-learn。无需下载仓库代码或使用自定义训练库。

## Kaggle 操作

1. 下载上面的 `.ipynb`，在 Kaggle 新建 Notebook 中导入。
2. 启动新 Session，选择 **GPU T4 × 2**，打开 **Internet**。
3. Add Input 添加比赛 [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data)（`word2vec-nlp-tutorial`）。
4. 点击 **Run All**，等待三个实验结束。

必须有这三个文件，Notebook 会自动定位并校验：

| 文件 | 作用 |
| --- | --- |
| `labeledTrainData.tsv.zip` 或 `labeledTrainData.tsv` | 25,000 条有标签影评 |
| `testData.tsv.zip` 或 `testData.tsv` | 25,000 条比赛测试影评 |
| `sampleSubmission.csv` | 测试 ID 与提交顺序 |

不需要 GloVe、`unlabeledTrainData.tsv`、旧权重或旧提交文件。BERT 和真正的 DeBERTa-v2-xxlarge 由 Hugging Face 下载；公开模型不强制提供 HF_TOKEN。保留 Kaggle 的 CUDA PyTorch，不安装 CPU Torch 替代包。

## 本次实验条件

三组共用一次生成的 **20,000 训练 / 5,000 验证**划分，按标签分层、随机种子 42。数据只去 HTML，保留标点与大小写，再使用对应的官方 tokenizer。

| 实验 | 长度 | batch | 梯度累积 | 有效 batch | 轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT 全量 | 256 | 8 | 2 | 16 | 3 |
| BERT LoRA | 256 | 8 | 2 | 16 | 3 |
| DeBERTa-v2-xxlarge LoRA | 128 | 1 | 16 | 16 | 3 |

学习率 `2e-5`、AdamW、weight decay `0.01`、线性调度、10% warmup。LoRA 的 `r=8`、`alpha=16`、`dropout=0.05`。按每轮**验证 AUC**保存并恢复最佳模型，再做预测。准确率最高的一轮和 AUC 最高的一轮可能不同。

BERT 两组形成相同底座、相同长度的对照；XXL 的底座和截断长度同时变化，结果表会保留这些条件。新划分和调度与旧实验不同，此前成绩不会填进新结果表。

## 两张 T4 与显存

默认只暴露第 1 张 T4 给 Trainer，三个模型顺序运行，各自完成预测并释放对象后才进入下一组。两张 T4 不会自动成为一张 32 GB 显存设备；此处也没有双卡并行或模型拆分。

BERT 使用 FP32 参数 + FP16 AMP。XXL 以 FP16 加载冻结底座，可训练的 LoRA、pooler、classifier 保持 FP32，并开启 nonreentrant 梯度检查点。XXL 使用长度 128、micro batch 1 控制显存，实际峰值以运行记录为准，不能预先保证所有 Kaggle 环境都不会 OOM。

若某一组失败，会保留 `error.txt` 并继续后续实验。总表的 `status` 明确区分成功和失败，最后只在三组全部完成时显示成功。重新从头 Run All 会生成一个新的时间戳目录，不会复用旧检查点；不要在同一 Session 反复执行安装/初始化部分。

## 输出

本次所有文件位于 `/kaggle/working/imdb_official_three_models_时间戳/`：

| 文件 | 内容 |
| --- | --- |
| `results.csv` / `results.json` | 三组状态、验证 accuracy/AUC/loss、最佳轮次、参数量、显存、时间和提交路径 |
| `split_ids.csv` | 本次训练/验证划分 |
| `environment.json` / `experiments.json` | 实际依赖版本、设备和实验配置 |
| 每组 `epoch_validation.csv` | 每轮验证结果 |
| 每组 `training_log.csv` / `log_history.json` | Trainer 原始日志，不虚构逐轮训练准确率 |
| 每组 `training_arguments.json` | Trainer 的完整实际训练配置 |
| 每组 `best_model/` | 完整 BERT 或 LoRA adapter + 任务头，附 tokenizer |
| 每组 `validation_predictions.csv` | 真实标签、预测及正类概率，用于分析误分类 |
| 每组 `submission.csv` | 按官方模板排列的 25,000 条正类概率，供比赛 ROC-AUC 提交 |
| 每组 `test_labels.csv` | 相同顺序的 0/1 预测 |

三份 `submission.csv` 在三个不同实验子目录中。训练完成后下载本次 Output，分别提交 Kaggle；Notebook 不自动提交。`results.csv` 的 Kaggle score 留空，实际排行榜成绩需由真实提交补充。LoRA 输出不包含冻结底座，重新加载仍需要对应的公开 checkpoint。

**验证范围：** 已用本地随机初始化的小模型跑通三组官方Trainer训练、最佳模型选择、保存和预测导出，并检查了LoRA/任务头更新、冻结参数保持、失败后继续和重复运行保护。测试使用CPU与缩小的数据，不代表真实BERT/XXL成绩；真实XXL的T4训练和显存尚需在Kaggle实测。交付的Notebook没有预填训练输出。

## 官方依据

- [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)
- [DeBERTa-v2-xxlarge 模型卡](https://huggingface.co/microsoft/deberta-v2-xxlarge)
- [PyTorch 梯度检查点](https://docs.pytorch.org/docs/stable/checkpoint.html)
