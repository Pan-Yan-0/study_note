# 英语五折联合结果目录

## 结果

完整验收：125项mDeBERTa、15项Qwen SFT、15份教师、15项KD；31套OOF各3042条。核心分数见[原运行汇总](结果汇总.md)，解释与配对分析见[五折分析](../../英语五折联合验证结果.md)，整个课题各阶段见[项目总览](../../README.md)。

| 文件或目录 | 内容 |
|---|---|
| COMPLETE.json | 完成数量、31套OOF分数与配对差 |
| independent_audit.json | 62组官方分数复算、各类指标、KD错误得失 |
| oof/ | 五折拼接的逐条预测、参考标签和官方分数 |
| mdeberta/ | 第五组正式任务配置、训练记录、logits、预测 |
| formal/qwen/ | 五折三种子SFT缓存、配置、选轮记录 |
| formal/teachers/ | 教师概率、权重来源哈希、三项补跑诊断 |
| formal/kd/ | 匹配教师KD正式输出 |
| recovery/ | 补跑来源与成功状态 |
| archive_selection.json、ARCHIVE_MANIFEST.json | 选取归档范围及原包校验值 |
| error_examples.json | 24条按ID取样的描述性核对，不代表总体分布 |

## 问题与下一步

本目录矩阵已完成，不依据旧阶段缺项或历史失败记录重新运行。Qwen其他技巧仍是单折，多语言新方法未全面验证。分析与下一步统一见[当前状态](../../实验状态与算力计划.md)。模型权重单独由运行者保存，不在本目录。

原始输出保留字节；README是维护索引，不参与原归档清单。复核命令：`python code/joint_results_audit.py <官方仓库目录>`。实验由Pan-Yan-0运行，GPT独立复算归档。
