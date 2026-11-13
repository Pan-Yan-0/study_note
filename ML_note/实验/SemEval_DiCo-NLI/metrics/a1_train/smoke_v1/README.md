# A1 v1 的 Kaggle T4 SMOKE

Pan-Yan-0 在 Kaggle T4×2 实例上运行，SMOKE 绑定一张 T4，使用 fp16、torch 2.11.0+cu128、CUDA 12.8 和 transformers 4.57.6。对应[代码提交](https://github.com/Pan-Yan-0/study_note/commit/ae00b270c71b5a1d0331e7d69036a4bc7a1963e2)与锁定的 mDeBERTa revision `a0484667…`。

## 结果

| 组 | 实际更新 | fp16 跳步 | 内层验证行 | 留出行 |
|---|---:|---:|---:|---:|
| B2-Haiku | 2 | 0 | 244 | 608 |
| B2-RL | 2 | 0 | 244 | 608 |
| B2-SL | 2 | 0 | 244 | 608 |
| B2-LS | 2 | 0 | 244 | 608 |

- Codex 用发布的训练包重算正式入口门槛。四组配置、输出文件哈希和每折 A0 教师目标哈希一致。独立重建四组留出对称化预测及验证 WF1。
- one-hot soft CE 与硬 CE 各执行两次更新，loss、梯度范数和参数的最大绝对差均为 0。
- B2-Haiku 的留出预测、验证预测、额外 logits（608 / 244 / 852 行）与 Commit 5 SMOKE 逐字节一致。15 个 B2 正式任务的配置与输出均通过复用检查。

## 下一步

用同一训练包和 Notebook，将 `RUN_FORMAL` 设为 `True`，在 Kaggle 选 Save & Run All。同环境下复用 Commit 5 的 B2，正式训练 RL、SL、LS 共 45 项。主比较使用冻结方案中的 20,000 次家族配对重抽样与双侧 Bonferroni 98.33% 区间。

训练包 SHA-256 为 `70fc1e01624947a476190c6b890806fbe8e213febeec7bead2663cc0c7886332`。回传的完整结果包 `a1_train_v1_results.zip` SHA-256 为 `ecba6eb32bfb4815ce5d019ffbac706841b735bf039360e6f0f902bb5fc3a304`。此目录记录回执与配置，逐条预测留在回传包中。
