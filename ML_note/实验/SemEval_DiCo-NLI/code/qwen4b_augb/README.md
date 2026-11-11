# Qwen3-4B 上的 +1 数据扩充队列（qwen4b_augb v1.1）

在 Qwen3-4B-Instruct-2507 的 LoRA CE SFT 上，检验实验 B 的 +1 生成数据有没有用。三组只差训练数据，配方和 JOINT-QWEN 相同，在阿里云 A10 bf16 上跑。代码在 qwen8b_v5.5 上改：训练核心 `upstream_qwen.py` 逐字节不变（`502e6388…`），也就是第六组、JOINT-QWEN 和实验 A 用的同一个核心。

## 设计

| 组 | 每折训练数据 | 每折行数 | 每折计划步数（3 轮） |
|---|---|---|---|
| Q0 | fit | fit（2,186–2,200） | 约 411 |
| Q-B1 | fit + 该折选用的 Haiku +1 生成行（清单 `b380b4a6…`，和 mDeBERTa 的 B1-Haiku 是同一批） | 2·fit | 约 822 |
| Q-step1 | fit + fit 再复制一遍 | 2·fit | 约 822 |

- 共同配置：Qwen/Qwen3-4B-Instruct-2507 快照 `cdbee75f…`，关闭思考，zero 模板，只算答案字母的损失；LoRA r16 / alpha32 / 七个投影 / dropout 0；微批 8 × 累积 2；lr 1e-4，最多 3 轮，按原方向内层验证 WF1 选轮（内层验证只用原数据）；英语五折 × seed 42–44。
- **划分**：worker 用未改动的上游 `make_split` 算五折，运行时要求和实验 B 的 `split_manifest.json`（`f9363290…`）逐折相同（CPU 上已用官方数据核对五折都相同）。
- **隔离**：每条生成行都检查本折可用、上下文父样本都在本折 fit 里、来源是 Haiku；+1 的生成行数等于 fit 行数。
- **行的顺序**：Q-B1 是 fit 在前、生成行按 `fold_selection.csv` 的顺序在后；Q-step1 是 fit 在前、复制的 fit 在后。和 mDeBERTa 的 B1 / step+1 一样。上游用 `group_by_length` 采样，打乱由种子决定。
- **Q0 为什么重跑**：JOINT-QWEN 的 fold 0 来自第六组的上游主入口，fold 1–4 来自 GPT 的联合驱动，正式训练也没有开确定性模式，所以就算环境一样也很难逐位复现。为了让三组在同一套代码、同一环境里配对，Q0 在本队列里重跑（和实验 B 的 B0 不复用第五组 C0 是同一个理由）。Q0 和 JOINT-QWEN 的差只作描述。

## 事先登记的判定（运行前定，看结果后不改）

- **主比较**（sym WF1，同种子差的均值，按家族配对重抽样 2,000 次、种子 20261021）：Q-B1 − Q0、Q-B1 − Q-step1。两条都满足"均值 > 0 且区间下限 > 0"，才算扩充在 Qwen 4B 上有效。
- **分两阶段**：阶段 1 跑 Q0 和 Q-B1（30 项）。Claude 拿到结果包后算 Q-B1 − Q0：
  - 通过：Pan-Yan-0 确认后跑阶段 2（Q-step1，15 项），再按两条规则下结论；
  - 不通过：结论是"+1 在 Qwen 4B 上没有带来提升"，不跑阶段 2（两条判断缺一条就不可能通过）。
- **描述**（不作判定）：plain 方向、SoftCons / HardCons、逐类 F1、O 的 AP；Q0 − JOINT-QWEN（同配方重跑的差）；Q-B1 和 mDeBERTa 的 B1 / B2（系统比较）。

## 运行（阿里云 PAI-DSW，A10）

上传一个包，用 `prepare_upload.py` 在本机生成（需要官方仓库 zip、一个已有的 4B LoRA 目录，比如 JOINT-QWEN 或第四组 A10 的任意一份权重，只用于兼容性检查的初始 adapter）：

```bash
python qwen4b_augb/prepare_upload.py --official-zip <官方仓库 zip> --adapter <4B LoRA 目录> --output a10_upload_qwen4b_augb.zip
```

在实例上解压后：

```bash
bash qwen4b_augb/a10_session.sh prepare   # 解压官方数据、装环境（约 5–10 分钟）
bash qwen4b_augb/a10_session.sh all       # 后台：下载 4B → 最长输入探测 → SMOKE → 兼容性 → 两个回执都通过后自动跑阶段 1
bash qwen4b_augb/a10_session.sh status    # 随时看进度
# 阶段 1 结束后下载 /mnt/workspace/dico_qwen4b_augb/download/ 里的三个包，结果包发给 Claude
# Claude 判定 Q-B1 − Q0 通过、Pan-Yan-0 确认后：
bash qwen4b_augb/a10_session.sh stage2
```

- SMOKE：Q-B1 完整训练集里最长 16 行的探测；Q-B1 在 fold 0 的 SMOKE 子集上（原数据子集 + 同样行数的生成行）做中断、损坏检查点、恢复；Q0 第二个种子的 SMOKE。
- 兼容性：沿用 v5.5 的 `compat.py`：零样本概率和历史 4B 缓存（1,448 个输入）差 ≤ 1e-5、0 个最大类不同，用来确认模型、模板和打分不变；本队列的覆盖入口和上游默认入口在确定性模式下 3 次更新逐位相同。v1.1 起 new 路径不再补写冻结值，只检查 `worker.load()` 覆盖后的值等于冻结值。
- 行数记在哪：`task.json` 的 `split_ids`、`fit_rows`、`fit_sources` 是裁剪或补行之前的候选（探测记的是该组完整训练集，SMOKE 之后还会补最长的原始行）；实际训练的行在 `actual_split.json`，行数在 `done.json` 的 `fit_rows`，回执按这两处核对（v1 在这里有阻断，v1.1 已修）。
- 换实例：`restore` 把上次的三个包恢复到新 ROOT，再运行 `stage1` 或 `stage2`，已完成的任务核对哈希后跳过。

## 估时

按 JOINT-QWEN 在 A10 上每步 2.35 秒估：

| 部分 | A10 时间 | 计算时（6.991 / 小时） |
|---|---|---|
| 验收（下载、探测、SMOKE、兼容性） | 约 1 小时 | 约 7 |
| 阶段 1：Q0 15 项（约 18 分钟 / 项）+ Q-B1 15 项（约 35 分钟 / 项） | 约 13.5 小时 | 约 94 |
| 阶段 2：Q-step1 15 项 | 约 9 小时 | 约 62 |

11 月剩余的额度以控制台为准。阶段 1 放得下的话可以一次跑完。

## 阶段 1 结果的分析（Codex 11-11 审查后补充）

- Claude 从各折 `cache.jsonl` 独立重建对称化概率和标签，和队列写的 OOF CSV 逐条比对，再算分；每组每个种子拼齐 3,042 行。
- 比较按种子配对，三个种子取平均；1,725 个家族重抽样 2,000 次（种子 20261021），两组和三个种子共用同一组抽到的家族，双侧 95% 区间，用未四舍五入的值判定，并报告逐种子差。区间只反映已训练模型下的样本抽样波动。
- O 的 AP 用对称化 `p_O` 在每个种子的完整 OOF 上算，三个种子分别报告；另报 `done.json` 的 `steps`、`fit_tokens`、`best_epoch`。
- 写法：阶段 1 没过线写"按事先规则，未证实 +1 优于 Q0，未运行 Q-step1"；阶段 1 过线、阶段 2 没过线写"相对 Q0 有提升，尚未证实优于原数据重复"。
- 分工：Claude 重建和分析，Codex 复核判定，Pan-Yan-0 转交结果并决定是否进入阶段 2。

## 文件

`common.py`：身份、数据、按组构造训练集、隔离检查；`worker.py`：单任务（v5.5 的 worker 加 `--group`）；`queue_augb.py`：队列、两个阶段、回执、OOF、打包；`compat.py`：4B 一致性（来自 v5.5；v1.1 起 new 路径只检查覆盖值、不补写）；`run_a10.sh`、`a10_session.sh`：A10 入口；`augb_data/`：实验 B 的 +1 清单、逐折选用、五折划分；`baseline/`：历史 4B 零样本缓存；`test_queue.py`：CPU 测试。

## 来源

训练核心由 GPT 编写（第六组 `dico_qwen_methods-2-gpt`），A10 会话外壳和 v5 队列由 GPT 编写、Claude 修订到 v5.5；本队列的分组、数据接入、两阶段和测试由 Claude 编写。生成数据由 Claude 调度 Haiku 5.5 生成（实验 B）。
