# 西语、巴语、跨语言全量的 Kaggle 驱动

[metrics/mdeberta_multilang_oof](../../metrics/mdeberta_multilang_oof/README.md) 那次运行的代码，由 GPT 编写。

- `semevaltask2.ipynb`：在 Kaggle（T4 ×2）上运行的 notebook，已清掉输出，Dataset 路径里的用户名换成了占位符。它把 `runner.py` 写进运行目录，再依次执行 `prepare`、`smoke`、`run`。
- `runner.py`：从 notebook 里原样取出的驱动。两张 T4 各跑一个独立任务（赛道 × 种子 × 折），训练精度固定为 fp16 autocast，每个任务完成后核对指纹、精度和文件哈希再打包；全部完成后拼出每个赛道、每个种子的折外预测并用官方脚本评分。
- `official_files.json`：官方仓库 40 个文件的 SHA-256。
- 训练本身调用第五组的 mDeBERTa 脚本，原样使用，不改代码：[joint_validation/dico_mdeberta_methods.py](../joint_validation/dico_mdeberta_methods.py)（SHA-256 `83bfe561…`）。

任务记录里的代码身份 `8554642853d6…` = SHA-256（`runner.py` 的哈希 + 训练脚本的哈希 + `official_files.json` 的哈希），用本目录的文件可以算出同一个值。
