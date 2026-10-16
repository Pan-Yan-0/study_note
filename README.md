# Pan-Yan-0 的学习笔记（note 分支）

机器学习与 NLP 的实验记录、论文笔记和课程笔记。实验和笔记都在 [ML_note/](ML_note/) 下。

```text
ML_note/
├── 实验结果汇总.md        全部实验结果，LoRA 论文结果作参照
├── 实验/
│   ├── IMDB/             README、code/、notebooks/、metrics/、notes/
│   └── SST2/             README、code/、notebooks/、metrics/
├── 原理笔记/
│   └── BERT/  DeBERTa/  LLM/  LoRA/
└── 吴恩达机器学习/         课程各部分的笔记
```

## 实验

| 实验 | 内容 | 主要结果 |
|---|---|---|
| [SST-2 指令学习](ML_note/实验/SST2/README.md) | 《1-使用大模型进行指令学习》任务：DeepSeek 与 Qwen3.5 0.8b / 2b / 4b / 9b 在三个指令模板（basic / cot / role_cot）× 思考开关下的准确率；Mistral、Llama、Gemma、Phi-4 四个系列；DeBERTa-v3-base 全量与 16 / 64 / 256 / 1024 条少样本微调对照 | 验证集 872 条：DeepSeek 95.18%，Gemma 3 12B 94.61%，Qwen 4b 93.81%，Phi-4 93.00%，Llama 3.1 8B 92.43%，Mistral 7B 90.60%；DeBERTa 全量 95.60%，1024 条 92.97%，16 条 61.62% |
| [IMDB 情感分析](ML_note/实验/IMDB/README.md) | Kaggle *Bag of Words Meets Bags of Popcorn*：GloVe + 六种网络；BERT 全量、BERT LoRA、DeBERTa-v2-xxlarge LoRA；Qwen3-4B-Base 指令微调（LoRA，只算答案 token 的损失） | 编码器最高 Kaggle AUC 0.98294（DeBERTa-v2-xxlarge LoRA）；Qwen3-4B 指令微调 Kaggle AUC 0.99493（验证 0.9941，零样本 0.9847） |
| [实验结果汇总](ML_note/实验结果汇总.md) | 本仓库全部实验的结果速览（新的、重要的在前），LoRA 论文的结果放在后面作参照，并说明哪些数字可以比、哪些不能 | — |

每个实验目录下有 `code/`（脚本）、`notebooks/`（Kaggle Notebook）和 `metrics/`（逐条结果、汇总、运行日志）。

## 论文与原理笔记

| 笔记 | 内容 |
|---|---|
| [LoRA 论文笔记](ML_note/原理笔记/LoRA/LoRA论文笔记.md) | 方法、公式与补充推导 |
| [BERT 原理笔记](ML_note/原理笔记/BERT/BERT原理笔记.md) | 嵌入、注意力、一层的数据流，配图和玩具数字脚本 |
| [DeBERTa 与 IMDB 三条路学习笔记](ML_note/原理笔记/DeBERTa/DeBERTa与IMDB三条路学习笔记.md) | AUC，DeBERTa（解耦注意力、EMD、RTD、GDES），微调要点，大模型零样本与少样本 |
| [解码器大模型与路线 C 学习笔记](ML_note/原理笔记/LLM/解码器大模型与路线C学习笔记.md) | 因果掩码与上下文、agent、提示的成本、解码器结构、cot 与概率、LoRA 分类器 / 嵌入 + 逻辑回归 / 蒸馏、指令微调（SFT）与 T4 用时估算、交叉熵与熵、开思考为什么常被截断 |
| [IMDB 六模型实战与概念问答](ML_note/实验/IMDB/notes/IMDB六模型实战与概念问答.md) | GloVe 六模型的概念问答整理 |

## 吴恩达机器学习课程笔记

| 笔记 | 集数 |
|---|---|
| [What Is ML](ML_note/吴恩达机器学习/What_Is_ML.md)、[Supervised Learning](ML_note/吴恩达机器学习/Supervise_Learning.md) | 第一部分开篇 |
| [神经网络直觉](ML_note/吴恩达机器学习/Neural_Network_Intuition_42-51.md) | 42–51 |
| [TensorFlow 训练细节与 Softmax](ML_note/吴恩达机器学习/TF_Training_Softmax_59-70.md) | 59–70 |
| [模型评估与偏差 / 方差](ML_note/吴恩达机器学习/Evaluation_Bias_Variance_71-79.md) | 71–79 |
| [迭代开发与迁移学习](ML_note/吴恩达机器学习/Iterative_Dev_Transfer_80-85.md) | 80–85 |
| [倾斜数据与决策树](ML_note/吴恩达机器学习/Metrics_DecisionTree_86-95.md) | 86–95 |
| [集成方法：随机森林 / XGBoost](ML_note/吴恩达机器学习/Ensemble_RF_XGBoost_96-100.md) | 96–100 |
| [K-means 聚类](ML_note/吴恩达机器学习/KMeans_101-107.md) | 101–107 |
| [无监督学习（聚类、降维、异常检测、推荐系统）](ML_note/吴恩达机器学习/无监督学习笔记_PDF第13-16章.md) | 讲义第 13–16 章 |

[Coursera-ML-AndrewNg-Notes-master/](Coursera-ML-AndrewNg-Notes-master/) 是黄海广整理的 2014 版课程中文笔记，作为参考资料保存。
