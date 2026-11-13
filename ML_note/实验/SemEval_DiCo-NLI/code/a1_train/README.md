# DiCo-NLI · A1 Kaggle T4×2

A1 在普通初始化的 mDeBERTa 上检验 Q0 教师能否改善 B2 生成数据的监督。五折 × seed 42–44。训练基线是 [Commit 5 的 B2-Haiku](https://github.com/Pan-Yan-0/study_note/blob/bb8606c/ML_note/实验/SemEval_DiCo-NLI/metrics/augment_b/train_c5/README.md)，配方见[冻结计划](https://github.com/Pan-Yan-0/study_note/blob/bb8606c/ML_note/实验/SemEval_DiCo-NLI/更大模型与数据扩充计划.md)。A0 的教师目标来自通过复现检查的 T4 fp16 输出，A0 审计按预定规则让三组入队。

| 组 | 生成行的训练目标 |
|---|---|
| B2-Haiku | 原生成标签，硬 CE |
| B2-RL | 本折 Q0 三种子 sym 概率平均后的 argmax，硬 CE |
| B2-SL | 同一四类平均概率，fp32 soft CE |
| B2-LS | 原生成标签 0.925，其余各 0.025，fp32 soft CE |

四组原始行都使用金标 one-hot，训练行和顺序一致。批大小 32，12 轮，原方向内层验证 WF1 选轮，同分取早。SMOKE 先校验 one-hot soft CE 和硬 CE 的 loss、梯度范数和实际更新，再跑四组各两次实际更新。B2 先按 Commit 5 的两个 batch 原样复现，逐格比较留出概率、验证概率与 logits、末轮 logits，并验证 15 项正式归档。复用通过时跑 45 项，否则本队列重跑 B2，共 60 项。按 Commit 5 的约 10.3 分钟/task，T4×2 理想墙钟约 4 / 5.2 小时，另计安装、SMOKE、评分和打包。

## Kaggle

1. 新建 Notebook，导入 ZIP 里的 `A1_Kaggle_v1.ipynb`。把 `A1_Kaggle_Train_v1.zip` 上传为一个数据集并 Add Input。选 **GPU T4 x2**，打开 Internet 下载锁定 revision 的 mDeBERTa。包内已含英语官方数据和评分代码，训练使用 A0 导出的概率。
2. 保持 `RUN_FORMAL = False`，运行全部 cell。下载 `a1_train_v1_results.zip` 检查 SMOKE。通过后改 `RUN_FORMAL = True`，用 **Save & Run All** 运行正式队列。
3. 运行结束后，在版本 Output 下载 `a1_train_v1_results.zip`。中断时把上一版结果 ZIP 或解压出的结果挂为 Input，在新会话保持 `RUN_FORMAL = True`。身份和文件哈希一致的任务会跳过，未完成的 task 从头开始。保持相同 GPU 型号、精度和 torch / CUDA / transformers 版本。

结果包中 `a1_smoke.json` 记录复用判定和失败原因，`logs/` 记录两个 worker 的执行，`summary.json` 只在完整队列通过检查后写出。主比较是 RL / SL / LS 分别对 B2 的 sym WF1 差，双侧 Bonferroni 98.33% 区间。20,000 次家族配对重抽样，所有组和三个种子共用每次抽样。SL − RL 和 SL − LS 是探索性比较。另报逐类 F1、O 类 P/R/AP、O 与 F/B 的误判流向，以及置信度 ≥ 0.99 的覆盖和错误。

## 重建与复核

在本目录执行以下命令。官方仓库需 checkout 到 `588968e`，本仓库需包含 A0 全量和 Commit 5 的归档。

```bash
python build_bundle.py /path/to/official /path/to/A1_Kaggle_Train_v1.zip
unzip /path/to/A1_Kaggle_Train_v1.zip -d /path/to/input
python run_a1.py preflight --bundle /path/to/input/A1_Kaggle_Train_v1
python test_a1.py /path/to/input/A1_Kaggle_Train_v1
```

解压结果后可以在 CPU 上用 `python run_a1.py score --bundle /path/to/input/A1_Kaggle_Train_v1 --out /path/to/a1_run` 重建 OOF 和区间。打包脚本验证 A0 的 15 份回执、输出和 Q0 归档，用 `a0_analysis.py stats` 重建教师目标并对照封存哈希。代码沿用 Commit 5 的 B2 配方、数据预检、优化器更新和 FP32 评估，并采用 Commit 6 的单卡 worker 模式。
