# AGENTS.md · Pan-Yan-0 学习项目协作档案

> 本文件是本对话/agent 的持久工作约定，持续更新。每次会话开始时应先读本文件。

## 学习画像与教学 skill（2026-10-15）

学习画像和教学方法只放在 Pan-Yan-0 账号的私有 skill `cognitive-tutoring` 里（本仓库公开，不存画像）；讲解、出题、批改、规划学习和整理笔记时按它做，画像更新也写回那个 skill。仓库里的 `skills/cognitive-tutoring/` 是旧版，「核心偏好与约定」第 1、3、4、5 条和「进度快照」是历史记录，都以私有 skill 为准。

## 用户档案
- Pan-Yan-0（GitHub 用户名；文档里用这个名字，不用中文名）。2026-09-20 最新确认：正式主导师为云南大学信息学院张学杰教授，目前主要由王津教授指导；用户说明王津曾是张学杰的学生（用户陈述，未将共同署名当作师承证明）。此前志愿顺序为张学杰、王津、张有，导师归属不再标为未确定。旧档案中的桂电/2025级信息未核实，不用于推断个人培养要求。
- 职业目标：**AI Infra / 大模型推理优化**（CUDA、推理引擎方向）。
- 当前主线：执行导师的 ML 学习路径（吴恩达 2022 版 ML 课程 142 集 + IMDB 情感分析项目），同时自学 CUDA。

## 核心偏好与约定（用户明确要求，必须遵守）
1. （教学类约定，已移入私有 skill `cognitive-tutoring`）
2. **重要问题大开大合**：发现之前方案/教程有结构性问题，直接重写，不做修修补补；不用先征求同意。
3. （教学类约定，已移入私有 skill `cognitive-tutoring`）
4. （教学类约定，已移入私有 skill `cognitive-tutoring`）
5. （教学类约定，已移入私有 skill `cognitive-tutoring`）
6. **笔记工作流**：用户口述笔记和疑问 → agent 整理成结构化 Markdown + 逐题答疑 → 推送到 GitHub `Pan-Yan-0/study_note` 的 `note` 分支（ML_note/ 目录）。每次提交笔记时同时按私有 skill 第 3 节更新画像（2026-10-15 用户要求）。每晚约 21:51（北京时间）有定时任务「每晚提交学习笔记」在学习对话里自动整理并提交，每次运行先预约下一晚。
7. 中文交流，教程/页面用中文。
8. 学习安排写入现有 HTML，按阶段、任务、交付物与验收标准组织；不设具体日期或固定时长，允许中断后通过“下一步”记录接续。
9. **提交署名**（2026-10-12 用户要求）：推送到 GitHub 的提交以用户为作者：`git commit --author="Pan Yan <112330427+Pan-Yan-0@users.noreply.github.com>"`，提交者仍是 Claude，信息末尾保留 Claude 的署名；提交内容含 GPT 写的代码、Notebook 或文字时，再加 `Co-authored-by: GPT6.1sol <noreply@openai.com>`。文档的“来源”一节写清谁运行了实验、谁写了哪部分代码。已推送的提交不为补署名而强制改写历史。

## 项目资产
- **导师研究画像（2026-09-20）**：研究全局图第09部分 `#advisor`，源 `.research-src/advisor_section.html`；含王津相关8篇论文PDF角色、张学杰云边资源研究2篇贡献声明与历史硕博作者证据。研究方向为公开样本综合判断，正式指导安排依用户最新确认；不把通讯作者自动当学位导师。
- **NLP 研究全局图（2026-09-17）**：`app/research.html` 与根目录 `NLP研究全局图.html`；6 类方向、6 篇 2025–2026 论文与 1 篇 GKD 基础锚点、研究与发表流程、国内培养要求边界、个人无日期研究路线和选题草稿。编辑源 `.research-src/`，修改后运行 `build.py`；`integrate.py` 维护旧页面入口。草稿独立键 `nlp-research-map-v1`，支持 JSON 恢复与 Markdown 导出，不读取或认证课程掌握。仅本地交付，未发布 Kimi。离线结构与交互检查通过，浏览器本地文件访问受策略限制，未做真实视觉验收。
- **无日期阶段计划**：伴学页默认进入 `#plan`，7 个阶段、34 条自检标准；`#plan/阶段id` 定位详情。来源 `.d2l-src/plan_content.py`、交互 `.d2l-src/plan.js`。计划保存在原 `d2l-companion-v1` 的可选 `plan` 字段，兼容旧课程备份；不改 `ml-tracker-v1`，不自动把自检当作掌握证据。
- **Windows 本地李沐伴学（2026-09-16 新增）**：`app/d2l.html`，另有根目录 `李沐深度学习伴学.html`；18 个单元，独立存储键 `d2l-companion-v1`，支持 JSON 备份与 Markdown 笔记导出。编辑源在 `.d2l-src/`，修改后运行 `build.py` 同步两份 HTML。当前仅更新本地网站，未发布到 Kimi。使用说明见 `李沐伴学使用说明.md`。
- **学习网站**（/mnt/agents/output/app，website_version_manager 发布，type=html）：
  - index.html 主路线图 / trial.html 试水期 / tracker.html 142 集追踪器（localStorage 存笔记，键 ml-tracker-v1）
  - kaggle.html 作业指南 / kaggle-code.html 代码精讲（含★全流程地图）/ imdb.html IMDB 手写实战·完整流水线 / link.html 课程×项目联动表
  - bridge.html 训练闭环补桥计划（①小网络训练 ②注意力计算 ③Transformer 编码器 ④BERT 与 IMDB 微调，四阶段完整教学）
- **GitHub 仓库**：
  - `word2vec-nlp-tutorial`：Kaggle 作业代码（词袋 0.84576 / Word2Vec 0.84396）
  - `imdb-sentiment-models`：导师 IMDB 脚本包的运行指南/FIXES（15 个原始脚本由用户自行网页上传）
  - `study_note`（note 分支）：课程笔记
- **本地备份**：关键交付物同步复制到 /mnt/agents/output/（中文命名）

## 关键技术事实（避免重查）
- Kaggle 提交正确姿势：`sample.copy()` 模板 + assert id 对齐（quoting=3 会让 test.id 带字面引号）。
- gensim 4：glove2word2vec 已删，需手动加首行「词数 维度」头；model.wv.*；size→vector_size。
- Kaggle 环境：用户已注册并确认可以使用 GPU（2026-09-16 最新陈述）；具体 GPU 型号与内存以实际会话为准。GloVe 840B 用公开数据集挂载，不下载。
- 导师 IMDB 脚本包：imdb_process.py 先跑（GloVe→pickle）；CNN/LSTM/GRU/Attention/Transformer 系依赖 pickle；BERT 系独立。设备 cuda:0 硬编码。OOM 三连：batch_size 减半 → max_len 256 → 换 DistilBERT。
- 后续训练实验可以使用 Kaggle GPU；本地 VSCode/WSL 仍可用于编辑与小例子练习。脚本支持 DATA_DIR 环境变量覆盖。

## 进度快照（更新于用户最新进度 JSON）
- 2026-09-19 用户要求依据云南大学王津教授论文的通讯作者、学生作者与学位身份分析团队方向。推断须区分论文合作、通讯作者责任、正式指导关系与当前学位身份，不用作者排序替代导师关系证据。
- 2026-09-17 用户转述导师提到 on-policy distillation 中反向 KL 的意义，以及“年底 SemEval”、RL、Agent、NLP；用户已确认导师尚未给具体任务编号或论文链接。具体论文、SemEval 届数/任务编号与研究题目尚未确认；不能据此认定这些全部为必修或用户已经掌握。研究入口应以具体任务、基线、评价指标与小型可验证问题为中心，不要求先学完整个 NLP 或做完全部推理优化。
- 2026-09-16 最新交流：用户转述导师建议重点看 BERT 和 Transformer。后续学习安排围绕这两个主题按需补基础；不将完整学完李沐课程或 LSTM 作为前置门槛。此条记录学习方向，不代表已掌握这些模型。
- 2026-09-20 IMDB 实战重大进展：六个模型（CNN 0.88880 / LSTM 0.89128 / GRU 0.89996 / CNNLSTM 0.88580 / AttLSTM 修复后 0.89660 / Transformer 0.86944，均 5 epoch public LB）全部跑通并提交 Kaggle。AttLSTM 曾因 bug 得 0.50004（随机水平）后修复。epoch=20 实验观察到 GRU/AttLSTM 分数下降（候选解释：过拟合；未经训练/验证曲线确认，不据此断定最佳轮数）。概念问答经通义千问网页版进行，Kimi 校验版笔记存 `ML_note/IMDB六模型实战与概念问答.md`。
- 2026-09-20 新增学习网站页面 `bridge.html`（训练闭环补桥计划）：GPT 建议路线（小网络一次训练 → 注意力 → Transformer → BERT）的教师版授课页；阶段①含固定数值小网络 T1 手算前向 / T2 手算梯度+autograd 核对 / T3 backward vs step，数字经 autograd 验证（∂L/∂W1[1,0]=0.084）；与 d2l 4.7 节对照。
- 2026-09-20 bridge.html ②③④ 由预告扩为完整教学（应用户「不够深入」反馈；d2l 10.3/10.5/10.6/10.7/14.8/15.6 经 web_open_url 通读）：②注意力计算（h₁h₂h₃+q 贯穿例：点积→softmax→加权 [0.4223,0.1554,0.4223]、dim=-1 沿键、masked_softmax 先于 softmax 填 −1e6、加性 vs 缩放点积参数对比、√d 方差论证 d=512 std≈22.6）；③Transformer 编码器（3×3 自注意力矩阵 [0.2119,0.2119,0.5761] 陷阱题、复杂度对比表、d=4 位置编码手算、LN vs BN [[1,2],[2,3]]、四投影层参数定位、残差梯度通路、编码器块形状跟踪图、M6 对照）；④BERT 微调（三种嵌入相加、[CLS]/[SEP]、MLM 15%+80/10/10 防错位、1.1 亿参数 vs 分类头 1538、MLM/NSP 头微调时丢弃、Kaggle 五步协议：切分→训练→验证集选点→提交→错例分析≥10 条）。所有数字 Python 验证。网站版本 db9f2ba。
- 当前目标（用户 2026-09-20 陈述）：按补桥计划学习，先完成阶段①小网络训练闭环，再进入注意力 → Transformer → BERT。②③④ 教材已就位，待用户审批修正。
- 2026-09-21 导师新任务（用户转述）：跑 `imdb_bert_trainer.py` 与 `imdb_bert_scratch.py` 出结果，再改 roberta-large 与 deberta-xxlarge。已交付 `/mnt/agents/output/bert_kaggle/`（四脚本+README，zip 已打包）：两个 Kaggle 兼容修补版（load_metric→numpy accuracy；evaluation_strategy 版本自适应；DATA_DIR 环境变量；自动建 result 目录）、imdb_roberta_large_trainer.py（fp16+batch8×累积2+lr1e-5+max_len256）、imdb_deberta_xxlarge_trainer.py（9 亿参数 16GB 四件套：梯度检查点+adamw_bnb_8bit+fp16+batch1×累积16）。关键事实：bert_scratch 实为「手写分类头+预训练编码器」，非从零训练。四脚本 py_compile 通过；训练须在 Kaggle GPU 执行，结果记录表待用户回填。
- 课程：已看 1–36、42–107（共 102 集，第二部分完结，第三部分进行中）；**缺 37–41（过拟合与正则化）**，计划中。
- Kaggle 作业：Part 1–3 全部完成并提交成功。
- IMDB 脚本包：六模型全部跑通并提交（见上）。
- 待办（用户侧）：三遍法第 2 遍盲写、补看 37–41、补桥计划阶段①验收、导师 BERT 四模型实验回填结果表。Kaggle 注册与 GPU 使用权限已就绪，不再列为待办。

## 工作流程备忘
- 页面发布：website_version_manager(build_version, project_dir=/mnt/agents/output/app, type=html)，只返回版本 ID。
- GitHub 推送用 push_files（内容内联，大文件不行）。
- 改 tracker.html 的 DATA：格式为 `const DATA = [{"id":1,"sec":1,"title":"...","dur":秒,"part":N,"mod":"..."}, ...]`（单行 JSON）。
