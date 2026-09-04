# AGENTS.md · 潘岩学习项目协作档案

> 本文件是 agent 的持久工作约定，持续更新。每次会话开始时先读本文件。

## 用户档案
- 潘岩（GitHub: Pan-Yan-0），桂林电子科技大学研究生新生（2025 级），导师：王津老师。
- 职业目标：**AI Infra / 大模型推理优化**（CUDA、推理引擎方向）。
- 当前主线：执行导师的 ML 学习路径（吴恩达 2022 版 ML 课程 142 集 + IMDB 情感分析项目），同时自学 CUDA。

## 核心偏好与约定（用户明确要求，必须遵守）
1. **先全局地图，后细节**：解释任何算法/代码前，先给全流程图（数据处理 vs 模型两大部分），讲清每步产出什么（数据还是预测）。
2. **重要问题大开大合**：发现之前方案/教程有结构性问题，直接重写，不做修修补补；不用先征求同意。
3. **教学原则：让用户自己写**：教程采用脚手架式（任务→提示→自己写→对照答案），不是「给完整代码让用户读」。
4. **教程必须完整覆盖**：从数据处理到模型输出的全流程，不留黑盒；老师给的所有模型/脚本都要纳入，不能只做一部分。
5. **三遍学习法**：第 1 遍跟写理解 → 第 2 遍盲写骨架 → 第 3 遍复盘/改造。
6. **零基础假设**：解释数学/概念时假设零基础，逐个定义符号。
7. **笔记工作流**：用户口述笔记和疑问 → agent 整理成结构化 Markdown + 逐题答疑 → 推送到 GitHub `Pan-Yan-0/study_note` 的 `note` 分支（ML_note/ 目录）。
8. 中文交流，教程/页面用中文。

## 项目资产
- **学习网站**（/mnt/agents/output/app，website_version_manager 发布，type=html）：
  - index.html 主路线图 / trial.html 试水期 / tracker.html 142 集追踪器（localStorage 存笔记，键 ml-tracker-v1）
  - kaggle.html 作业指南 / kaggle-code.html 代码精讲（含★全流程地图）/ link.html 课程×项目联动表
  - imdb.html **IMDB 手写实战·完整流水线**（Part A 手写数据处理替代 imdb_process.py / Part B 通用训练框架 train_model+predict_submit / Part C 六模型 M1 CNN→M6 Transformer / Part D DistilBERT 微调）
- **GitHub 仓库**：
  - `word2vec-nlp-tutorial`：Kaggle 作业代码（词袋 0.84576 / Word2Vec 0.84396）
  - `imdb-sentiment-models`：导师 IMDB 脚本包的运行指南/FIXES（15 个原始脚本由用户自行网页上传）
  - `study_note`（note 分支）：课程笔记 + 本文件
- **本地备份**：关键交付物同步复制到 /mnt/agents/output/（中文命名）
- **参考书**：黄海广《斯坦福2014机器学习个人笔记》v5.52（336 页 PDF 已上传）——只精读第 9 章（反向传播，新版课删掉的部分）；71–87 集时对照第 10–11 章；PCA/SVM/大规模机器学习挂起备用。

## 关键技术事实（避免重查）
- Kaggle 提交正确姿势：`sample.copy()` 模板 + assert id 对齐（quoting=3 会让 test.id 带字面引号）。
- gensim 4：glove2word2vec 已删，需手动加首行「词数 维度」头；model.wv.*；size→vector_size；支持 `word in model`。
- Kaggle 环境：29GB RAM，GPU 需手机验证（用户尚未通过）；GloVe 840B 用公开数据集挂载，不下载。
- 导师 IMDB 脚本包：GloVe 系（cnn/lstm/gru/attention_lstm/cnnlstm/transformer）依赖 pickle；BERT 系（bert/distilbert/roberta × native/trainer）独立、需联网或 hf-mirror。设备 cuda:0 硬编码、lr=0.8+SGD——教程版已改为自动设备检测 + Adam。
- OOM 三连：batch_size 减半 → max_len 256 → 换 DistilBERT。
- 用户手机验证未过 → 本地 VSCode/WSL 备选；脚本支持 DATA_DIR 环境变量覆盖。

## 进度快照（更新于用户最新进度 JSON）
- 课程：已看 1–36、42–70（共 65 集）；**缺 37–41（过拟合与正则化）**，做 IMDB 过拟合实验前必须补。
- Kaggle 作业：Part 1–3 全部完成并提交成功。
- IMDB：进入手写阶段（Part A→B→M1 先行，CNN 知识已齐）。
- 待办（用户侧）：Kaggle 手机验证、三遍法第 2 遍盲写、补看 37–41、网页上传 15 个原始脚本到 imdb-sentiment-models。

## 工作流程备忘
- 页面发布：website_version_manager(build_version, project_dir=/mnt/agents/output/app, type=html)，只返回版本 ID。
- GitHub 推送用 push_files（内容内联，大文件不行）。
- 改 tracker.html 的 DATA：格式为 `const DATA = [{"id":1,"sec":1,"title":"...","dur":秒,"part":N,"mod":"..."}, ...]`（单行 JSON）。
- 页面 head/CSS 复用：改页面时从现有页提取 <head>，空白填通用类：.blank/.task/.shape/.tag/details。
