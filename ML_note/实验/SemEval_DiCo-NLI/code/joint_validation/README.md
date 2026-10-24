> 本目录归档代码与说明；官方数据、旧结果和权重在对话提供的完整上传ZIP中，不在GitHub重复上传。

# A10 联合实验运行说明（已执行，供复现）

原队列及补跑已全部完成，见[执行验收](../../SFT与蒸馏验证计划.md)。以下命令用于复现，不是当前待执行清单；原32条教师检查的故障与修复见[补跑代码](../joint_recovery/recover_joint.py)。

## 本次运行内容

1. 第五组 mDeBERTa：复用第二阶段已有16项，补109项，最终得到五种设置、五折、五种子，共125项及25套OOF。
2. Qwen CE SFT：复用已核验的fold0三个种子结果和LoRA权重，新增fold1–4、种子42/43/44共12项。最终15项、3套OOF。
3. 用每折自己的Qwen CE老师导出fit及val的正反输入概率。val仅用于核验，学生训练只读取fit。
4. 新的统一教师组KD：五折、三个种子，共15项及3套OOF；和第五组同折、同种子的CE配对。旧的第四组教师KD单独保留，不混入本组。

所有超参数沿用既有配方，不在此队列做搜索。训练核函数来自原始第五组v3和第六组v2，源码逐字保留；新增驱动负责折、教师、恢复、队列和验收。新旧fold0输出的命名差异由汇总器明确处理，数据按外层fold目录隔离。

不包含：新的Qwen dropout/MoCo跨折训练、四个赛道的全面KD、交叉拟合教师、数据扩充。此包完成当前规划的英语验证，不能称为整个课题全部完成。

## 开机前

- 实例：一张A10，Python 3.11镜像，与此前实验保持相同卡和精度。
- 工作盘建议至少预留30 GiB。新环境安装到独立venv_joint。
- 上传国内实例时切换到适合国内上传的网络。包内已包含官方数据、旧结果、旧教师概率和fold0 LoRA权重，无需再找散文件。
- 原运行按94 calculation预算规划，主队列78、保留16；这不是当前余额。已找到fold0权重，省去三个SFT重训。建议仍留出10–11小时、并确保结束时能下载和释放实例；这不是耗时保证。
- 本队列已通过A10 SMOKE和正式运行；新实例复现仍需先做SMOKE。本地未运行完整4B模型。

## 1. 上传并解压

将 a10_joint_validation.zip 上传到 /mnt/workspace，在终端依次执行：

```bash
cd /mnt/workspace
python -m zipfile -e a10_joint_validation.zip /mnt/workspace
cd /mnt/workspace/a10_joint_validation
bash run_a10.sh setup
```

setup会核对上传文件、原始结果与权重的SHA256、五折师生划分、第五组恢复任务和运行环境。预期看到“第五组第二阶段已有16/125项”和“Qwen fold0三个种子的权重可直接复用”。已恢复过的实例可能显示更多完成项，这是正常的。

## 2. 跑SMOKE

```bash
cd /mnt/workspace/a10_joint_validation
bash run_a10.sh smoke
```

顺序为原第五组SMOKE、Qwen fold1小规模训练与教师导出、对应学生KD及KD续跑检查。原始第五组SMOKE使用旧教师做原路径自检，新联合KD SMOKE使用刚导出的新教师，输出目录分开。

耗时通常约15–25分钟，首次下载模型另计。SMOKE在前台运行，期间保持终端连接；正式训练会在后台运行。详细进度在 dico_joint/logs/ 下，主终端在每个子任务结束时报告状态。

结束应看到“SMOKE全部通过。正式任务尚未开始。”以及基于测速的预计时长。若预算估计超过78 calculation，脚本返回失败并暂不启动正式队列；将 budget_estimate.json 和相关日志发给GPT重新安排，不能把失败当成通过。预算估算包含当前安装/SMOKE已耗时间，但不是实时账单，也不是自动释放功能。

## 3. 后台运行完整队列

确认SMOKE通过后：

```bash
bash run_a10.sh run
```

看到“已在后台开始，PID …”后可以关闭网页，后台继续运行。不要重复执行其他训练脚本；本包的运行锁防止自身重复启动，但不能阻止手工启动不使用同一锁的旧脚本。

```bash
bash run_a10.sh log
```

查看总进度。Ctrl+C只退出日志查看，不终止后台实验。总日志只显示每个任务的开始和结束；某个训练任务的细节可用：

```bash
tail -n 40 -f /mnt/workspace/dico_joint/logs/mdeberta_stage2.log
```

随后Qwen的日志形如 qwen_fold1_seed42.log，KD的日志形如 kd_fold1_seed42.log。也可以查看状态：

```bash
bash run_a10.sh status
```

每个fold完成后更新结果包和权重包。文件已经出现不等于全部完成。任何失败会写日志；独立任务继续，相应教师失败会阻断对应KD；最终有缺项时不会写COMPLETE。

## 4. 完成验收与下载

队列结束、不再持有锁后运行：

```bash
bash run_a10.sh check
```

必须看到：

```text
COMPLETE：125个第五组任务、15个SFT、15份教师、15个KD；31套OOF已复算。
```

它会重新读取逐条概率/logits、核对任务和教师身份、调用官方评分器复算，并生成完整OOF，随后更新两个下载包。OOF每个种子3042行，五折WF1不会被简单平均冒充OOF分数。

从 /mnt/workspace/dico_joint/download/ 下载：

- dico_joint_results.zip：发给GPT，含各任务结果、老师概率、原始预测/logits、日志、COMPLETE.json和结果汇总.md。
- dico_joint_weights.zip：自己保存，含15个老师选中轮次的LoRA权重；约2 GB量级，以实际文件为准。
- DOWNLOAD.json：两个包的大小和SHA256，建议一并保存。

核对下载文件大小、能正常打开，并保存完整权重后再手动释放实例。脚本不会自动释放实例；训练结束后实例仍然占用资源。

## 出错后继续

同一实例：修复具体错误后重新运行 bash run_a10.sh run。完成任务会核对后跳过；SFT未完成时从完整epoch检查点恢复；导出概率从完整缓存继续；KD单个任务较短，未完成的任务从头跑。不会因为某个分数低而自动重训或选择另一个种子。

需要先下载现有输出时，在队列结束后运行 bash run_a10.sh pack。部分包没有COMPLETE，不能冒充最终结果。

换实例：保留本次同版本上传包，将已下载的 dico_joint_results.zip 和 dico_joint_weights.zip 一起放入解压后的 a10_joint_validation 文件夹，再按setup、smoke、run执行。setup核对结果包内清单并恢复已完成输出；未完成SFT的检查点不进结果包，换实例时这一个任务需要重跑。模型底座会重新下载，完成的LoRA权重从权重包恢复。

不要手动删除done.json、替换老师、修改fold或损失系数来绕过报错。先发对应日志，避免把不同配置混成同一实验。

## 来源与验证边界

原始第五组和第六组训练代码由Claude编写、GPT审阅并修正，已有A10运行记录。本联合驱动、教师身份协议、预算估计、验收和打包由GPT/Codex编写。实验由Pan-Yan-0运行。

本地测试使用官方数据、既有输出和明确标记的合成测试数据，覆盖完整矩阵验收、错误教师拒绝、缺项拒绝、原评分复算、导出续跑和KD接口。合成测试结果不放入实验结果或交付输入包。具体测试数量及源码哈希见 local_validation.json。
