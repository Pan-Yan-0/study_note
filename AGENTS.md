# AGENTS.md · 潘岩学习项目协作档案

> 本文件是本对话/agent 的持久工作约定，持续更新。每次会话开始时应先读本文件。

## 用户档案
- 潘岩（GitHub: Pan-Yan-0）。2026-09-20 最新确认：正式主导师为云南大学信息学院张学杰教授，目前主要由王津教授指导；用户说明王津曾是张学杰的学生（用户陈述，未将共同署名当作师承证明）。此前志愿顺序为张学杰、王津、张有，导师归属不再标为未确定。旧档案中的桂电/2025级信息未核实，不用于推断个人培养要求。
- 职业目标：**AI Infra / 大模型推理优化**（CUDA、推理引擎方向）。
- 当前主线：执行导师的 ML 学习路径（吴恩达 2022 版 ML 课程 142 集 + IMDB 情感分析项目），同时自学 CUDA。

## 核心偏好与约定（用户明确要求，必须遵守）
1. **先全局地图，后细节**：解释任何算法/代码前，先给全流程图（数据处理 vs 模型两大部分），讲清每步产出什么（数据还是预测）。
2. **重要问题大开大合**：发现之前方案/教程有结构性问题，直接重写，不做修修补补；不用先征求同意。
3. **教学原则：让用户自己写**：教程采用脚手架式（任务→提示→自己写→对照答案），不是「给完整代码让用户读」。用户读现成代码吃力且记不住。
4. **三遍学习法**：第 1 遍跟写理解 → 第 2 遍盲写骨架 → 第 3 遍复盘。
5. **零基础假设**：解释数学/概念时假设零基础，逐个定义符号。
6. **笔记工作流**：用户口述笔记和疑问 → agent 整理成结构化 Markdown + 逐题答疑 → 推送到 GitHub `Pan-Yan-0/study_note` 的 `note` 分支（ML_note/ 目录）。
7. 中文交流，教程/页面用中文。
8. 学习安排写入现有 HTML，按阶段、任务、交付物与验收标准组织；不设具体日期或固定时长，允许中断后通过“下一步”记录接续。

## 本项目教学 skill
- 教学、学习计划、练习批改与笔记答疑时，读取 [cognitive-tutoring/SKILL.md](cognitive-tutoring/SKILL.md)，再按任务读取其中的参考文件，不必一次加载全部资料。
- 可编辑源文件位于 `cognitive-tutoring/`；`cognitive-tutoring.skill` 是对应分发包，`cognitive-tutoring.original.skill` 保留本轮修订前版本。修改源文件后需同步重新打包，不能把旧包当作最新规则。
- 学习能力与进度记录以用户最新陈述和实际作答为准；旧进度快照不能覆盖新信息。看完课程、跟写运行与独立掌握分别记录。

## 项目资产
- **导师研究画像（2026-09-20）**：研究全局图第09部分 `#advisor`，源 `.research-src/advisor_section.html`；含王津相关8篇论文PDF角色、张学杰云边资源研究2篇贡献声明与历史硕博作者证据。研究方向为公开样本综合判断，正式指导安排依用户最新确认；不把通讯作者自动当学位导师。
- **NLP 研究全局图（2026-09-17）**：`app/research.html` 与根目录 `NLP研究全局图.html`；6 类方向、6 篇 2025–2026 论文与 1 篇 GKD 基础锚点、研究与发表流程、国内培养要求边界、个人无日期研究路线和选题草稿。编辑源 `.research-src/`，修改后运行 `build.py`；`integrate.py` 维护旧页面入口。草稿独立键 `nlp-research-map-v1`，支持 JSON 恢复与 Markdown 导出，不读取或认证课程掌握。仅本地交付，未发布 Kimi。离线结构与交互检查通过，浏览器本地文件访问受策略限制，未做真实视觉验收。
- **无日期阶段计划**：伴学页默认进入 `#plan`，7 个阶段、34 条自检标准；`#plan/阶段id` 定位详情。来源 `.d2l-src/plan_content.py`、交互 `.d2l-src/plan.js`。计划保存在原 `d2l-companion-v1` 的可选 `plan` 字段，兼容旧课程备份；不改 `ml-tracker-v1`，不自动把自检当作掌握证据。
- **Windows 本地李沐伴学（2026-09-16 新增）**：`app/d2l.html`，另有根目录 `李沐深度学习伴学.html`；18 个单元，独立存储键 `d2l-companion-v1`，支持 JSON 备份与 Markdown 笔记导出。编辑源在 `.d2l-src/`，修改后运行 `build.py` 同步两份 HTML。当前仅更新本地网站，未发布到 Kimi。使用说明见 `李沐伴学使用说明.md`。
- **学习网站**（/mnt/agents/output/app，website_version_manager 发布，type=html）：
  - index.html 主路线图 / trial.html 试水期 / tracker.html 142 集追踪器（localStorage 存笔记，键 ml-tracker-v1）
  - kaggle.html 作业指南 / kaggle-code.html 代码精讲（含★全流程地图）/ imdb.html IMDB 手写实战·完整流水线 / link.html 课程×项目联动表
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
- 2026-09-17 用户转述导师提到基础不足、on-policy distillation 中反向 KL 的意义，以及“年底 SemEval”、RL、Agent、NLP；用户已确认导师尚未给具体任务编号或论文链接。具体论文、SemEval 届数/任务编号与研究题目尚未确认；不能据此认定这些全部为必修或用户已经掌握。研究入口应以具体任务、基线、评价指标与小型可验证问题为中心，不要求先学完整个 NLP 或做完全部推理优化。
- 当前 LSTM 概念卡点：用户曾把门控理解为 good 加好评、not 遗忘；已解释 GloVe 表示、门控状态与最终分类的区别，尚未获得用户独立作答证据。计划加入短补桥，但不把完整 LSTM 实现作为 Transformer 前置。
- 2026-09-16 最新交流：用户转述导师建议重点看 BERT 和 Transformer。后续学习安排围绕这两个主题按需补基础；不将完整学完李沐课程或 LSTM 作为前置门槛。此条记录学习方向，不代表已掌握这些模型。
- 2026-09-20 IMDB 实战重大进展：六个模型（CNN 0.88880 / LSTM 0.89128 / GRU 0.89996 / CNNLSTM 0.88580 / AttLSTM 修复后 0.89660 / Transformer 0.86944，均 5 epoch public LB）全部跑通并提交 Kaggle。AttLSTM 曾因 bug 得 0.50004（随机水平）后修复。epoch=20 实验观察到 GRU/AttLSTM 过拟合下降。概念问答经通义千问网页版进行，Kimi 校验版笔记存 `ML_note/IMDB六模型实战与概念问答.md`。概念理解证据来自问答转述，尚未经独立作答验证。
- LSTM 概念卡点更新：经多轮问答，用户已能口述三门职责、CEC 加法保梯度、"只有 h_t 外传"等要点；旧"good 加好评"误解已被纠正性问答覆盖，仍待独立复现确认。
- 当前目标（用户 2026-09-20 陈述）：开始学习 Transformer 架构与代码。路线按李沐伴学阶段计划：注意力 → Transformer 本体 → BERT 微调。
- 课程：已看 1–36、42–107（共 102 集，第二部分完结，第三部分进行中）；**缺 37–41（过拟合与正则化）**，计划中。epoch=20 过拟合实验已完成，37–41 补上后可做正式偏差/方差复盘。
- Kaggle 作业：Part 1–3 全部完成并提交成功。
- IMDB 脚本包：六模型全部跑通并提交（见上）。
- 待办（用户侧）：三遍法第 2 遍盲写、补看 37–41。Kaggle 注册与 GPU 使用权限已就绪，不再列为待办。

## 工作流程备忘
- 页面发布：website_version_manager(build_version, project_dir=/mnt/agents/output/app, type=html)，只返回版本 ID。
- GitHub 推送用 push_files（内容内联，大文件不行）。
- 改 tracker.html 的 DATA：格式为 `const DATA = [{"id":1,"sec":1,"title":"...","dur":秒,"part":N,"mod":"..."}, ...]`（单行 JSON）。
