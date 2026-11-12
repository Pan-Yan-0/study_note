# A0：Q0 教师在 B2 生成行上的推理（Kaggle T4×2，fp16）

研究对齐 v4（冻结计划变更记录 11-12 条目）和 v4.1（执行平台从 A10 改为 Kaggle，Codex 11-12 同意）。只做推理，不训练。当前是 v1.1：按 Codex 对 v1 的审阅，补了概率校验和续跑身份校验。

## 做什么

- **教师**：Qwen 4B +1 阶段 1 的 Q0。每折 3 个种子各一份，共 15 份 LoRA adapter，都是在 A10 bf16 上训练、只在本折 fit 上训练的。
- **推理范围**：B2 每折实际送训的生成行（`fold_selection_scale.csv` 的 tier 1、2），两个方向去重，每折 5,044–5,088 条，乘 3 个种子共 76,116 条。
- **推理方式**：按两段文本长度之和降序（稳定排序），每 64 条一组，调用上游 `score()`。四类概率的算法和训练、评估时相同：每个字母的两种写法先 logsumexp，再在四类上 softmax。
- **精度**：T4 没有原生 bf16，4B 用 fp32 放不进 16 GB，所以用 fp16。教师概率就是这次 fp16 推理的结果。

## 每个任务的检查（门槛在运行前写定）

| 检查 | 做法 | 门槛 |
|---|---|---|
| 身份 | `adapter_model.safetensors`、`adapter_config.json` 的哈希和归档 `COMPLETE.json` 对比；选中轮和 `trained.json` 对比；config 的 r / alpha / target_modules / dropout 和上游常量对比 | 全部相同 |
| 载入 | 和上游 `train()` 相同的路径，载入后逐张量和文件比对 | 全部相同 |
| 内层验证重评（15 份都做） | 输入和顺序同上游，argmax 和归档 `val_probs/epoch{best}.json` 比较 | 一致率 ≥ 98%（上游 `load_epoch` 的门槛） |
| 逐折复现（每折 seed 42） | 按归档 `cache.jsonl` 的行序重算 1,446–1,464 条输入，也就是历史上的分批方式（`test_a0.py` 用真实数据核对过行序） | 留出行 sym 标签和归档 `pred_fold0_sym.csv` 的一致率 ≥ 99%，四类概率最大绝对差 ≤ 0.10；另报平均差和翻转数 |

- 某一折的 seed 42 没过：这一折不出任何教师文件，进程以非零状态退出。失败回执里保留诊断：最大差对应的输入和新旧概率、翻转样本的新旧 top-2 间距、差的分位数、低答案质量的行数、答案 token。
- **概率校验**（复现、内层验证、教师推理和归档参考共用）：行数对得上；四类值都有限、非负、和为 1；mass 有限且在 [0, 1]；top_is_letter 是布尔值。所有 JSON 写出用 `allow_nan=False`。
- **回执验收**（新任务写完、续跑跳过、打包时共用）：fold、seed、SMOKE / 正式、选中轮、两项 adapter 哈希（和当前文件比）、预期输入数、各检查的通过状态、输出哈希，以及运行身份。运行身份包括代码、包清单摘要（含 Q0 归档）、模型快照摘要、推理环境（GPU 型号、dtype、torch、CUDA、固定依赖版本）。卡编号和缓存路径单独记录，不参与比较。
- 回执不能沿用时报错，不覆盖；失败的回执要人看过才能重跑。
- 全量入口两个 worker 都返回 0 才打结果包；有非零时只打诊断包 `a0_failed_diagnostics.zip`。

## 运行（Kaggle）

1. **建一个数据集**，上传 `a0_teacher_bundle.zip`，Kaggle 会自动解压。把 Qwen 4B +1 阶段 1 下载的权重包也加成一个数据集，里面要有 `formal/Q0/seed*/fold*/adapters/`，已解压或没解压的 zip 都行。
2. **打开 `A0_Teacher_Kaggle.ipynb`**：Accelerator 选 GPU T4 x2，打开 Internet，加上面两个数据集。
3. **`RUN_FULL = False`，从头运行**：
   - 装固定版本的依赖（transformers 5.19.0、peft 0.21.2、accelerate 1.15.0、datasets 5.1.0）；
   - 从 Hugging Face 按快照 `cdbee75f…` 下载 4B 并逐文件核对；
   - SMOKE：fold 0 seed 42 的全部检查，加 128 条教师推理。

   把 `a0_smoke.zip` 发给 Claude。
4. **Claude 核对、Codex 同意 v4.1 之后**，把 `RUN_FULL` 改成 `True`，用 Save & Run All (Commit) 后台跑：
   - GPU0 跑 fold 0、2、4，GPU1 跑 fold 1、3；
   - 跑完自动打包 `a0_results.zip`，发给 Claude。
5. **中途停了**：把上次的 Output 加成数据集，再跑一次。

## 估时

- T4 的推理速度按 A10（约 23 条/秒）的一半估，两张卡并行。
- 全量 76,116 条，加复现 7,230 条，加 15 次载入：墙钟约 1.3–1.5 小时。
- 装依赖和下模型另需约 10–15 分钟，SMOKE 约 5–10 分钟。

## 输出

- `teacher/fold{k}/seed{s}.jsonl`：每行 `a`、`b`、四类概率 `p`、答案字母的概率质量 `mass`、`top_is_letter`。
- `teacher/fold{k}/seed{s}.json`：回执，包括身份、载入、内层验证重评、复现（seed 42）、环境、代码和数据哈希、用时。
- `A0_SUMMARY.json`、`model.json`、`logs/`。

三种子平均、F↔B 映射、和生成标签的对比，以及盲审抽样，都由 Claude 在本地做，不在 Kaggle 上算。

## 文件

| 文件 | 内容 |
|---|---|
| `teacher_a0.py` | setup / download / run / finalize |
| `test_a0.py` | CPU 测试，15 项，用假教师 |
| `build_bundle.py` | 从仓库组装数据集 |
| `make_notebook.py` | 生成 notebook |

`q4b/` 是 `code/qwen4b_augb` 的上游核心等文件，逐字节复制，训练核心 `upstream_qwen.py` 为 `502e6388…`。

## 已验证 / 未验证

- **已验证（CPU）**：15 项测试，包括：
  - 包内清单；
  - B2 输入条数和 Codex 算的一致；
  - 归档缓存行序等于历史分批；
  - 全流程、续跑、篡改报错；
  - 复现门槛能拦住 5% 的翻转并留下诊断；
  - NaN 和内层验证少一行都停下且不写回执；
  - JSON 写不出 NaN；
  - 续跑时换了权重或环境都报错；
  - 环境混杂时不打包；
  - 有 worker 非零退出时不出结果包；
  - SMOKE 只写 128 条；
  - 缺任务时不打包；
  - adapter 哈希不对、找不到 adapter 时报错。

  其中 NaN、少一行、换权重、换环境这四类，用 v1 的代码跑同样的测试都没被拦住，和 Codex 的复现一致。

  notebook 找包、找权重的那一格，用模拟的 `/kaggle/input` 试过已解压和 zip 两种情况。
- **未验证**：真实 adapter 在 T4 fp16 上的载入和数值差、实际吞吐，等 Kaggle SMOKE 实测。

## 来源

脚本由 Claude 编写；上游核心由 GPT 编写（第六组），Q0 由 Pan-Yan-0 在阿里云 A10 上训练（Qwen 4B +1 阶段 1）。
