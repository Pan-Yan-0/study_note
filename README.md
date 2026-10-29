# Pan-Yan-0 的学习笔记（note 分支）


机器学习与 NLP 的实验记录、论文笔记和课程笔记。实验和笔记都在 [ML_note/](ML_note/) 下。

```text
ML_note/
├── 实验结果汇总.md        全部实验结果，LoRA 论文结果作参照
├── 实验/
│   ├── IMDB/             README、code/、notebooks/、metrics/、notes/
│   ├── SST2/             README、code/、notebooks/、metrics/
│   └── SemEval_DiCo-NLI/ README、code/、notebooks/、metrics/
├── 论文阅读/              按方向分的论文精读笔记
│   └── 模型压缩/          ShortGPT、AdaPruner、Minitron
├── 原理笔记/
│   └── BERT/  DeBERTa/  LLM/  LoRA/  NLI/  反向传播/
└── 吴恩达机器学习/         课程各部分的笔记
```

## 实验

| 项目 | 当前进度 | 主要结果与范围 |
|---|---|---|
| [SemEval-2027 DiCo-NLI](ML_note/实验/SemEval_DiCo-NLI/README.md) | 初步实验阶段完成；进入语义错误诊断和数据扩充设计 | 英语五折OOF sym WF1：Qwen SFT 84.37%；mDeBERTa R-Drop 80.69%、CE 80.03%（五种子）；KD 80.22%、匹配CE 80.23%（三种子）。西语/巴语/跨语言mDeBERTa CE五折三种子OOF：77.97% / 73.45% / 74.95%。Qwen技巧仅英语单折，新方法多语言验证待后续安排 |
| [SST-2](ML_note/实验/SST2/README.md) | 指令学习与微调对照完成 | 验证872条：DeepSeek 95.18%、Gemma 3 12B 94.61%、Qwen 4B 93.81%；DeBERTa全量95.60% |
| [IMDB](ML_note/实验/IMDB/README.md) | 编码器与大模型微调完成 | Kaggle AUC：DeBERTa-v2-xxlarge LoRA 0.98294；Qwen3-4B指令微调0.99493 |

- [全部实验结果汇总](ML_note/实验结果汇总.md)：跨项目结果表、评测口径与历史记录。
- [SemEval完成清单与后续安排](ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md)：当前待办和算力状态。
- [SemEval英语五折分析](ML_note/实验/SemEval_DiCo-NLI/英语五折联合验证结果.md)：配对比较、蒸馏得失和研究瓶颈。
- [SemEval文件导航](ML_note/实验/SemEval_DiCo-NLI/文件导航与维护规则.md)：摘要、登记、原始输出和历史记录的分工。

每个项目的README负责结果、问题、下一步；原始输出保存在metrics，自动登记表从结果生成，不手改。

## 论文阅读：模型压缩

剪枝和蒸馏方向的论文精读，索引和读法见 [论文阅读/README](ML_note/论文阅读/README.md)。

| 论文 | 剪什么 | 进度 |
|---|---|---|
| [ShortGPT](ML_note/论文阅读/模型压缩/ShortGPT.md) | 整层（深度剪枝），按 BI 删层 | 第一遍 ✅，第二遍主体 ✅ |
| [AdaPruner](ML_note/论文阅读/模型压缩/AdaPruner.md) | 层内耦合结构（宽度剪枝），贝叶斯优化搜索校准数据和重要性指标 | 第一遍 ✅ |

## 论文与原理笔记

| 笔记 | 内容 |
|---|---|
| [LoRA 论文笔记](ML_note/原理笔记/LoRA/LoRA论文笔记.md) | 方法、公式与补充推导 |
| [BERT 原理笔记](ML_note/原理笔记/BERT/BERT原理笔记.md) | 嵌入、注意力、一层的数据流，配图和玩具数字脚本 |
| [DeBERTa 与 IMDB 三条路学习笔记](ML_note/原理笔记/DeBERTa/DeBERTa与IMDB三条路学习笔记.md) | AUC，DeBERTa（解耦注意力、EMD、RTD、GDES），微调要点，大模型零样本与少样本 |
| [解码器大模型与路线 C 学习笔记](ML_note/原理笔记/LLM/解码器大模型与路线C学习笔记.md) | 因果掩码与上下文、agent、提示的成本、解码器结构、cot 与概率、LoRA 分类器 / 嵌入 + 逻辑回归 / 蒸馏、指令微调（SFT）与 T4 用时估算、交叉熵与熵、KV cache 存的是激活、开思考为什么常被截断 |
| [DiCo-NLI 句对输入与一致性学习笔记](ML_note/原理笔记/NLI/DiCo-NLI句对输入与一致性学习笔记.md) | SemEval 赛题的四个标签和两个方向，模型怎么分清两段短语（段向量、位置），SoftCons 与 HardCons，对称化，R-Drop 与 KL，方向版 R-Drop |
| [反向传播学习笔记](ML_note/原理笔记/反向传播/反向传播学习笔记.md) | 两层小网络从后往前推：损失对分数的梯度 p − q，传到 W₂、h，穿过 ReLU 到 W₁；反向传播与梯度下降的分工 |
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
