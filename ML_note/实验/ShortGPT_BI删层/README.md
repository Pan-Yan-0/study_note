# ShortGPT 方法的迁移实验：BI 删层（Qwen3-8B，Kaggle 2×T4）

用 ShortGPT 的 BI（Block Influence）给每层打分，删掉分数最低的 n 层，不做恢复训练，比较困惑度、零样本准确率和推理速度。这是 ShortGPT 方法在 Kaggle 上的**迁移实验 / 教学实现**，不是原文的严格复现。

- Notebook：[notebooks/shortgpt_bi_kaggle.ipynb](notebooks/shortgpt_bi_kaggle.ipynb)
- 论文笔记：[ShortGPT.md](../../论文阅读/模型压缩/ShortGPT.md)

## 结果

### SMOKE（只验证能跑通，数字不作结论）

设置：Qwen3-8B（revision `b968826d`），FP16，`device_map="auto"` 切到两张 T4；校准 4 段 × 512 token（WikiText-2 训练集）；PPL 用 WikiText-2 测试集前 5 个 512 窗口；每个任务只评前 50 题；测速 batch 1、输入 512、输出 128、重复 1 次。全程 3.3 分钟。原始文件在 [metrics/smoke_Qwen3-8B/](metrics/smoke_Qwen3-8B/)。

| 配置 | 层数 | 参数 | PPL | piqa | arc_easy | winogrande | TTFT 估计（s） | decode（tok/s） |
|---|---|---|---|---|---|---|---|---|
| 完整模型 | 36 | 8.19B | 12.81 | 82.0 | 82.0 | 70.0 | 0.474 | 11.8 |
| BI 删 4 层（29–32） | 32 | 7.42B | 26.60 | 72.0 | 66.0 | 68.0 | 0.444 | 13.0（1.10×） |

准确率是 acc_norm（winogrande 只有 acc），单位 %。

**链路检查**：
- 两张卡分别放 16 层和 20 层，没有被放到 CPU。
- 初始 logits、每层 BI、每个 PPL 窗口都是有限值。
- 双卡交界在第 16 层（hook 拿到的输入在 cuda:0，输出在 cuda:1），已在 hook 里先把输入搬到输出所在的卡。
- 测速时每张卡的峰值约 7.1 / 8.5 GiB（每张约 14.6 GiB），正式运行可以加大 batch。
- `bi_from_pair`、`drop_layers` 都通过了自测；删层后的模型 PPL、选择题和带 KV cache 的生成都能正常跑。

**BI 的形状**（只有 4 段校准数据，仅作预览）：第 0 层最高（0.68），第 26–33 层最低（0.025–0.034），最后一层回升到 0.07。和 ShortGPT 报告的"深层冗余"趋势一致。

### 正式运行

设置：校准 64 段 × 2048 token；PPL 用完整 WikiText-2 测试集（146 个 2048 窗口）；三个任务全部题目；测速输入 512、输出 128，batch 1 和 8，各重复 3 次取中位数。全程 61 分钟（Kaggle 后台提交）。结果文件夹 `Qwen3-8B_5fe9d4b6` 在 Kaggle 版本的 Output 里，还没下载进仓库。

| 配置 | 删的层 | 层数 | 参数 | PPL | piqa | arc_easy | winogrande | decode bs1（tok/s） | decode bs8（tok/s） | 加速 bs1 / bs8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 完整模型 | — | 36 | 8.19B | 9.71 | 77.7 | 80.9 | 67.7 | 12.1 | 48.0 | 1.00 / 1.00 |
| BI 删 4 层 | 26, 27, 31, 32 | 32 | 7.42B | 19.47 | 71.4 | 69.6 | 63.8 | 13.2 | 53.9 | 1.09 / 1.12 |
| BI 删 9 层 | 25–33 | 27 | 6.45B | 1073.75 | 62.0 | 42.6 | 54.8 | 15.8 | 63.7 | 1.31 / 1.33 |
| 从后往前删 9 层 | 27–35 | 27 | 6.45B | 23032.47 | 62.0 | 52.0 | 62.2 | 15.5 | 63.4 | 1.28 / 1.32 |

准确率是 acc_norm（winogrande 只有 acc），单位 %；随机水平 piqa、winogrande 50%，arc_easy 25%。TTFT 估计（bs1）：0.541 / 0.480 / 0.393 / 0.391 秒。

**BI**（64 段校准）：第 0 层 0.671 最高；第 25–33 层最低（0.035–0.040）；第 34、35 层回升到 0.056、0.081。和 ShortGPT 的"深层冗余"一致。

**读到的结论**（Pan-Yan-0 回答 notebook 第 7 节讨论题后整理）：
1. **删 9 层不恢复，能力损失很大**：PPL 涨约 110 倍；高出随机水平的部分，piqa 剩约 43%，arc_easy 约 31%，winogrande 约 27%（几乎接近乱猜）。
2. **PPL 和准确率给出相反的结论**：BI 删层的 PPL 远好于从后往前删（1074 对 23032），但从后往前删的 arc_easy、winogrande 更高。PPL 看正确词概率的绝对值，选择题只看选项的相对顺序。从后往前删去掉了第 34、35 层，最后的归一化和 lm_head 接不上前面的输出，概率整体乱掉，但相对顺序还保留着一部分。所以"哪种删法更好"取决于看哪个指标，用什么指标选层也决定了保住哪种能力（Minitron 用 Winogrande 选层，不用 BI）。
3. **加速接近层数之比**：删 9 层，bs1 加速 1.31，bs8 1.33，层数之比 1.33，参数之比 1.27。decode 访存受限，词嵌入每个 token 只读一行，按"实际要读的参数"算，比值约 1.30，和实测吻合。

## 问题

1. **和原文不能直接比数字**：

   | 项目 | ShortGPT | 这里 |
   |---|---|---|
   | 模型 | Llama-2-7B / 13B、Baichuan2 | Qwen3-8B（Llama-2 权限申请被拒） |
   | 校准数据 | PG19 | WikiText-2 训练集 |
   | 评测 | OpenCompass，MMLU、CMMLU 等 | WikiText-2 PPL + piqa / arc_easy / winogrande 零样本，lm-eval |
   | 精度 | 未说明 | FP16（T4 不支持 BF16） |

2. **评测的范围**：PPL 和三个按选项似然打分的选择题，没有检验自由生成和长链推理；测速用的是随机 token，不能说明生成质量。
3. **测速的口径**：TTFT 是"只生成 1 个 token 的 `generate` 端到端时间"，decode 吞吐是两次调用的时间差估算出来的，都是估计值。结论只适用于"HF + 两张卡按层切分"这种部署方式，不能外推到 vLLM。
4. **SMOKE 每个任务只有 50 题**：准确率的标准误约 ±6 个百分点，SMOKE 的数字不作结论。
5. **只有一个模型、一次校准**：BI 的排序和删层结果没有换校准数据或种子验证；正式运行的原始文件还没进仓库。

## 下一步计划

1. 把 Kaggle 输出的结果文件夹下载进 `metrics/full_Qwen3-8B/`。
2. 读完 AdaPruner 第二遍、和导师谈过之后，再决定是否做恢复训练（例如"从后往前删 + 只微调 lm_head 和最后几层"，对应 Reassessing Layer Pruning 的基线），以及是否加入生成和推理类评测。

## 来源

- 方法：ShortGPT（Men 等，arXiv 2403.03853）。
- Notebook 框架（加载、钩子、评测、测速、续跑、实验签名）由 Claude 编写；`bi_from_pair` 和 `drop_layers` 两个函数由 Pan-Yan-0 编写。
- GPT 对第一版做了静态审阅，指出了双卡交界处 hook 的设备问题、结果缓存缺少实验签名、显存峰值的统计窗口、非有限值只打印不停止等问题，第二版已全部修改。双卡问题后来在 Kaggle 上实际出现，证实了审阅意见。
- 实验由 Pan-Yan-0 在 Kaggle 2×T4 上运行。
- 依赖版本：torch 2.11.0、transformers 4.57.6、accelerate 1.15.0、lm-eval 0.4.13、datasets 5.0.1（见 `signature.json`）。
