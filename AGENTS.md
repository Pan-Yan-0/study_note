# AGENTS.md · Claude 在这个仓库里的工作备忘

Claude 在 Pan-Yan-0 的 study_note 仓库里工作时用的备忘：要遵守的规则、现在做到哪、东西放在哪。开始工作前先读一遍；内容过时就直接改这里。2026-10-16 以前的旧档案在 [agents/AGENTS旧版.md](agents/AGENTS旧版.md)，只作历史参考。

## 1. 最重要的几条

1. **称呼和语言**：称呼 Pan-Yan-0（GitHub 用户名），不用中文名；交流和文档都用中文，但**专业名词保留英文原词**（如 fold、seed、OOF、batch、checkpoint、warmup、family、subagent、SMOKE），不硬译成中文（Pan-Yan-0 11-07 定的）。
2. **仓库是公开的，导师也会看**：不写学习画像、经验值、个人邮箱、令牌和本机路径。教学方法和画像只放在 claude.ai 上 Pan-Yan-0 的私有 skill `cognitive-tutoring` 里；学习状态和存档放在 Pan-Yan-0 的私有仓库里。
3. **工作分支是 `note`**：`master`、`d2l`、`cuda_study`、`math_modeling` 不要动（`d2l` 由每周的定时任务单独维护）。
4. **任何决策都要有实验支持**（导师原话："扩充数据，参数选取，我也说不清好不好，实验结果说明好不好"）：
   - 改动都和基线对照，一次只改一处。
   - 数据少时跑多个随机种子，看同一种子下的配对差值；差值的波动盖过差值本身时不下结论。
   - 建议改什么时，同时给出验证它的对照实验。
5. **结论之前先抽查真实输出。** SST-2 里 Mistral 的 39.68% 其实是评测解析器太严造成的，改正后是 90.60%。分数异常低或高时，先看几十条原始回答，再下结论。
6. **重要问题直接重写**：发现方案或文档有结构性问题，直接重写，不做修修补补。

### GPT / Claude 文件交接与对话沉淀

- 当前研究主线为 **SemEval-2027 Task 2 DiCo-NLI**；IMDB、SST-2 已完成，仅按需引用历史代码和经验。
- Pan-Yan-0 要求双方各自整理可独立阅读的交接文件：只有讨论、设计或审阅文字时交付一个 Markdown；涉及代码、测试、配置或多份必要材料时打包一个 ZIP，附入口 README。不要仅把长审阅意见留在聊天里，也不为单个 Markdown 额外套 ZIP。
- 交接写清输入版本、问题、建议及理由、已完成验证、未验证部分和需要对方回应的事项；修订保留版本标识。交付文件由 Pan-Yan-0 转交，不假定另一模型能看到本会话。
- 持续维护本文件作为共享交接入口，不另建大小写或单复数不同的 agent.md。沉淀稳定约定、当前状态、已确认决策、待讨论问题及下一步；不保存逐字聊天，也不把审阅建议写成已经批准或完成的实验。
- **对话分工（Pan-Yan-0 10-28 定的）**：所有 Claude 对话和 GPT / Codex 共用这一个 AGENTS.md，改之前先 `git fetch`、只改自己负责的条目、改完尽快推送。
  - SemEval 对话：实验设计、代码、运行结果、实验 README，以及把论文研究交回的想法落成对照实验。
  - 论文研究对话：按错误分类选论文、预读和精读、阅读表、低成本离线检查；只写 `ML_note/实验/SemEval_DiCo-NLI/论文研究/`，要跑的实验写进其中的《交回SemEval对话的实验》，由 SemEval 对话决定。
  - 学习对话：讲解概念、出题批改，收工时整理进原理笔记；模型压缩方向的论文精读笔记写进 `ML_note/论文阅读/`（和 SemEval 的论文研究分开）。
- 当前待办仍以 [实验状态与算力计划](ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md) 为准；方案达成一致后再更新 [冻结计划](ML_note/实验/SemEval_DiCo-NLI/更大模型与数据扩充计划.md) 的变更记录。交接本身不启动新增 GPU 训练或付费生成。

## 2. 背景

- Pan-Yan-0：研究生，方向是 NLP 与大模型，职业目标是 AI Infra / 大模型推理优化。实验由他在 Kaggle、Colab、腾讯 Cloud Studio 上运行，再把结果带回来；决定由他做。
- 导师：布置实验任务，会看这个仓库，要求 README 记录"结果、问题和下一步计划"。
- 仓库里有一部分代码是 GPT 写的（例如 SST-2 评测脚本），提交时要按第 5 节加它的署名。

## 3. 仓库结构

```text
AGENTS.md / CLAUDE.md      本备忘（CLAUDE.md 只是引用本文件）
README.md                  首页：目录树、实验和笔记的入口
agents/                    旧版档案
skills/                    教学 skill 的旧版，现行版本在私有 skill 里
ML_note/
├── 实验结果汇总.md         全部实验结果速览
├── 实验/IMDB/  实验/SST2/  每个实验一个文件夹：README、code/、notebooks/、metrics/
│   实验/ShortGPT_BI删层/   模型压缩方向的动手实验（学习对话负责）
├── 论文阅读/               README（索引和读法）、模型压缩/（每篇论文一个文件，另有研究想法记录）
├── 原理笔记/               BERT/  DeBERTa/  LLM/  LoRA/  NLI/  PyTorch/  反向传播/  推理优化/
└── 吴恩达机器学习/          课程笔记
Coursera-ML-AndrewNg-Notes-master/   课程资料
```

- 文件放在主题文件夹的叶子节点里，只有汇总类文档放在上层。
- 新增或移动文件时，同步更新根目录 README 和相关 README 里的链接，并检查没有断链。

## 4. 笔记和文档的写法

笔记和汇总是写给导师和别人看的。

- **汇总的排序**：本仓库的成果在前，别人的（如论文）在后；重要的在前；新的在前。
- **实验 README**：分"结果 / 问题 / 下一步计划"三段写（导师的要求）。
- **原理笔记**：
  - 讲解按概念组织。
  - "检查题与我的作答"用表格（题目 | 我的作答 | 判定与要点），"我的作答"取 Pan-Yan-0 原话的意思。
  - 还没作答的题折叠起来，并附参考答案。
  - 最后是"易混点"和"下一步"。
- **不写**：实验过程的流水账、日期、画像和经验内容。代码只保留必要的注释。
- **来源一节**：写清谁运行了实验、谁写了哪部分代码，以及引用的论文和链接。

## 5. 提交和推送

- **作者是 Pan-Yan-0，提交者是 Claude**：
  ```bash
  git -c user.name="Claude" -c user.email="noreply@anthropic.com" \
      commit --author="Pan Yan <112330427+Pan-Yan-0@users.noreply.github.com>"
  ```
- **提交信息末尾的署名**：
  - 加 `Co-Authored-By: Claude …` 和会话链接。
  - 含 GPT 写的代码、Notebook 或文字时，再加 `Co-authored-by: Codex <noreply@openai.com>`（Pan-Yan-0 11-07 定的，不再区分具体模型名）。
- **推送前**：先 `git fetch`；检查 diff 里没有密钥、令牌和邮箱。
- **只用普通推送**：不强推，不为了补署名改写已推送的历史。推送失败就停下来，说明是哪一步、报了什么错。
- **结果文件**：`.gitignore` 忽略了 `*.csv`、`*.json`，要提交的指标文件用 `git add -f`。

## 6. 实验的约定

- **Notebook**：
  - 能断点续跑，带 SMOKE 小规模开关。
  - 默认把实验跑完，不设中途停掉实验的时间预算。
  - 写清它和导师原脚本有哪些区别，并附时间估计。
  - 在 Kaggle、Colab 和普通 Linux（如 Cloud Studio）上都能跑，换平台时能导入上一次的结果包接着跑。
  - 打印的提示写清已经完成了什么、还需要他手动做什么。
  - 结果包（Pan-Yan-0 定的原则）：之后需要、或者可能需要的都放进 zip，用不上的不放。比如 Qwen SFT 选中那一轮的 LoRA 权重（之后蒸馏、换平台都可能要用）单独打一个权重包，不混进发给 Claude 分析的结果包；训练中途的检查点、底座模型这类可以重新得到的不放。
- **控制变量（Pan-Yan-0 10-22 定的）**：每次运行都要能和别的运行比，说清改了哪个变量。
  - SemEval 的每次运行都登记进 `ML_note/实验/SemEval_DiCo-NLI/实验登记表.md`，表由 `code/registry.py` 从指纹、运行记录和逐条预测自动生成，不手改；新运行在 `RUNS` 里加一项，新比较在 `COMPARISONS` 里加一项。
  - 脚本自动列出两边记录了的变量差别（没记录的列为未知，不当作相同），比较分两列归类：设计（重复运行 / 单一计划改动 / 系统比较）和执行环境（一致 / 同卡同精度、软件或平台不同 / 卡或精度不同 / 信息不全）。平台名称、torch 和库版本只标注、不要求补跑（Pan-Yan-0 10-22 定的），但结论里要写明环境差别；把差别归因到某个变量，要求是单一计划改动；卡或精度不同时，在同一环境里补跑对照再归因。系统比较可以写进结论，只说在给定设置下哪个更好（GPT 10-22 审阅后的写法）。
  - 新写的 notebook：指纹里写全所有训练变量（包括损失、dropout、正则项的系数、训练数据），运行记录里写平台名称（Kaggle / Colab / 腾讯云 / PAI-DSW）、卡、精度和 torch 版本。
- **实验指纹**：provider、模型、提示词、是否开思考、服务地址、max_tokens、temperature、数据哈希。续跑时这些必须和原来一致。
- **max_tokens 包含思考内容**：开思考时容易写满被截断，没有答案的按错误计，并单独统计截断数。
- **评测解析**：从回答里取标签时，词和数字要一致才算；改了解析规则，要用新规则重算所有已有结果。
- **算力**：
  - Kaggle：GPU 有每周额度。
  - Colab：免费 T4，要一直开着页面，单次最多 12 小时。
  - 腾讯 Cloud Studio：T4 可用。`ollama pull` 会被拦，改用 curl 从 `registry.ollama.ai/v2/library/<名字>/manifests|blobs` 下载，核对 SHA-256，manifest 按原字节保存。
  - 阿里云 PAI-DSW：常常抢不到卡（10-21 抢到过 A10）。免费试用每月 250 计算时，月末作废；A10 每小时耗 6.991 计算时，一个月约 35 小时。额度用完后实例转成按量付费、继续扣钱。**停止实例不能止住扣计算时，要释放实例才行**（Pan-Yan-0 10-22 实测）；闲置 3 小时的自动关机不一定触发。释放前先把结果包和权重包下载下来。释放后要重新装环境、下模型，但走内网很快，所以不必让实例一直开着。每次开实例按同一个流程，由一个 sh 脚本完成：装环境 → SMOKE（Pan-Yan-0 手动跑）→ 全量。全量是任务队列：彼此不依赖的实验一次全部放进去连续跑完，每个任务能续跑、已完成的跳过；需要前置结果的等下一次（Pan-Yan-0 10-22 定的）。连 Hugging Face 要设 `HF_ENDPOINT=https://hf-mirror.com`。pip 要用内网镜像 `http://mirrors.cloud.aliyuncs.com/pypi/simple/`（加 `--trusted-host`），公网的 mirrors.aliyun.com 只有约 200 kB/s。镜像自带 torch 2.3.1，太旧；在 `/mnt/workspace/venv` 建 venv 装依赖（`notebooks/dico_a10.sh setup`）。网页终端粘贴要单独开终端窗口。
  - **本机网络（Pan-Yan-0 10-22 说的）**：和阿里云、腾讯云之间上传下载文件，要把本机网络切到国内线路，否则很慢；用 Claude、GPT、Kaggle 时又要切回去。所以每次开国内实例只安排两次传输：开始时上传一个打包好的 zip（脚本、代码、官方仓库的 zip、需要的输入如老师概率都在里面），结束时下载结果包和权重包。不要让他中途零散地传文件。
  - 本机显卡只有 4 GB。
  - T4 估时按约 10 TFLOPS 算。

## 7. 当前进度

GPT v4调试交回：包清单哈希一致、9项CPU测试实跑通过。debug_b03由全新GPT6Luna一次生成14对（E3/F7/O4），格式合格，无重试。原始输出与日志在 `GPT_to_Claude_AugB_v4_Debug.zip`，由Pan-Yan-0转交Claude；重点核对aircraft/plane标E的范围差异，cannot hear/deaf和所有格解释的边界。未开15批、未训练；等待Claude语义检查。数量规则未直接注入提示词，但规避数量对的设计受既往全数据观察启发，报告需保留探索历史，不能声称完全未受留出信息影响。Claude 已独立检查 `debug_b03`：没有明确错误，3 对边界都在 E 类（aircraft/plane 严格说是单向）；单向 7 对、O 4 对都成立，单向里 3 对是删一个形容词，超过提示词的多样性要求。决定冻结 v4、不再按调试样本改提示词，E 的质量交给 P1 的 30% 门槛；试点前固定两条只报告的口径（只差数量的对按 `number_pair` 统计、单向对删词比例），数量规则的来历已写进 `code/augment_b/README.md` 和冻结计划变更记录。下一步：GPT 用 v4 开 15 个试点批次并原样交回，Claude 跑 `process_outputs.py build`、查停止条件、准备 P1 审核表。交接 `Claude_to_GPT_AugB_v4_Go.md`（由 Pan-Yan-0 转交）。GPT 已交回 v4 试点原文（`GPT_to_Claude_AugB_v4_Pilot_Raw.zip`：15 批 16 次尝试，只有 b15 格式重试一次）。Claude 跑 `process_outputs.py build`：协议、失败批次、重复（2.4%）都通过，但平均词数偏离 65%（4.97 对 3.01 词）、探针高 22.9 个百分点，两条停止条件触发，v4 数据只留档在 `metrics/augment_b/pilot_v4/`，不做 P1。`diagnose_pilot.py` 显示 100 次同规模抽样解释不了这个差距，当前探针和特征分组下长度组是主要的可见信号。已改为 v5（`augb_pilot_v5_claude.zip`，模板 `9715b0d1…`，生成脚本 `6c04924f…`）：只改提示词，每批写明本批参考短语的长度和冠词开头比例并设长度上限，父样本和流程不变，包内 9 项测试通过。下一步：GPT 跑 `debug_b04`，格式合格且没有短语超过 5 词就直接跑 15 批，交回 Claude 处理；P1 在 v5 试点上做。交接 `Claude_to_GPT_AugB_v5.md`。GPT 已交回 v5 试点原文（`GPT_to_Claude_AugB_v5_Pilot_Raw.zip`；调度中途额度中断，b03/b08/b13 第一次尝试只剩开始记录，保守记 infra_error 后用第二次尝试生成，Claude 已在 README 第 9 节显式记录这一追踪缺口）。v5 长度通过（偏离 19%），探针高 29.6 个百分点，仍触发停止：生成器写成"教科书式最小对"（单向 76% 是删词，O 98% 是换一个词；原数据 45%，E / O 两侧毫无共同词 37% / 25%）。v5 数据只留档在 `metrics/augment_b/pilot_v5/`。Pan-Yan-0 11-05 定：改为按父样本仿写，v6 是最后一次试点，再触发停止就停掉生成、报告导师。Claude 已交 v6（`augb_pilot_v6_claude.zip`，模板 `45c22b02…`，生成脚本 `4ed71ac9…`）：每批按配额分层抽 14 个父样本（R1–R14），每个生成一对、标签和关系方式照它；格式检查核对第 k 对的标签等于 Rk；跳过只差数量的父样本；补充规划按组取父样本；`diagnose_pilot.py` 加构造类型和仿写对比；包内 10 项测试通过；GPT 对 v5 表述的修正已采纳。下一步：GPT 跑 `debug_b05`，合格且没有短语超过 5 词就直接跑 15 批。交接 `Claude_to_GPT_AugB_v6.md`。GPT 已交回 v6 试点原文（`GPT_to_Claude_AugB_v6_Pilot_Raw.zip`）：debug_b05 第一次顺序错、格式重试后合格；正式 18 次尝试 15 次合格，又因额度中断 3 次（b05、b15 没有生成器 ID，b10 有 ID 但回复不可恢复，已显式记录）。Claude 跑 build：**全部自动停止条件通过**，长度偏离 4.4%、探针差距 +1.9 个百分点、重复 0.5%；构造方式接近原数据（O 两侧无共同词 41% 对 25% 偏高）；Gk 和 Rk 实词平均 Jaccard 0.03。GPT 的两条措辞更正（v5 修改的影响只是假设；中断前调度子 agent 是否看过回复无法验证）已采纳。下一步：P1 审核。Claude 已独立标完 60 对并封存（SHA-256 `59d4ee88…`），等 Pan-Yan-0 独立标注；标完前带标签的 v6 输出不进公开仓库。三类都 pass 才放行，再先定父家族使用上限、规划全量；否则停掉生成、报告导师。**11-06 P1 完成，v6 试点放行**：Pan-Yan-0 独立标注（R049 留空按 U 记，填任何标签都不改变判定），两人一致项里和生成标签不同 E 2/15、单向 4/18、O 1/17，三类 pass；`gate.json` ready_for_training true，清单 `448ade30…`；全部原文和标注已归档 `metrics/augment_b/pilot_v6/`。7 处不一致的构造和父样本相同；按协议生成标签必然等于父样本金标，不能用来解释不一致（GPT 更正）。下一步：Claude 起草全量 +1 生成方案（每个父家族最多 2 次、同家族的两个 Gk 分到不同批次、无父样本的小签名配额确定性挪走、报告家族集中度；1.25 倍余量是否下调待 GPT），交接 `Claude_to_GPT_AugB_Full_Plan_v1.md`。GPT 11-06 复核（`GPT_to_Claude_AugB_Full_Plan_v1_Review.md`）建议选 B，并更正"7 处继承父标签"不能解释分歧（已改）；Pan-Yan-0 确认方案 B。Claude 已交全量包 `augb_full_v1_claude.zip`（说明在 `code/augment_b/FULL_README.md`）：主轮 121 批、1,489 个家族各用一次，最好情况缺口和 GPT 的表逐格一致；按试点保留率干跑，补充第 1 轮 12 批、294 行，预算分母 2,934 行、上限 586 行；`supplement.py` 加每个家族最多 2 次的上限；新写 `build_full.py`、`dryrun_full.py`、`plan_supplement.py`、`merge_full.py`（跨阶段去重、回填直接父样本、隔离核对、逐折选用 `fold_selection.csv`、第二次 P1、集中度、总放行绑定）；`test_full.py` 2 项和 `test_augment_b.py` 10 项在包内通过。下一步：GPT 调度主轮 121 批，交回后 Claude 合并、规划补充轮。GPT 11-06 交回主轮检查点（`AugB_Full_Main_Checkpoint.zip`）：6 批合格（84 对，过滤后 83），额度中断 3 次（b042、b082 第一次只有预定 ID）；b002、b082 第二次尝试漏传冻结提示词末尾一个换行符，GPT 暂停其余 115 批。Claude 复算确认只差这一个字节，定协议修订 1（`augb-protocol-r1`）：只差末尾换行符且写明期望哈希的尝试记为 `prompt_deviations`、不算失败，其他不一致仍失败；没有看生成内容。新增测试，`test_augment_b` 11 项、`test_full` 2 项通过；真实检查点上除未派发的批次外无其他协议问题。全量包改为 `augb_full_v1.1_claude.zip`（批次不变）。下一步：GPT 续跑其余 115 批，派发前对实际字符串算哈希、相等才派发。11-06 检查点 1（`AugB_Full_Main_Checkpoint_1.zip`）：b003 第一次 CLI 初始化失败、记 infra_error，随后额度用满，没有新原文；Claude 用 v1.1 核对，除未派发的 114 批外无协议问题。已提醒 GPT：b003 的重派额度已用掉，失败批次超过 6 批会触发停止；建议派发前先做不带提示词的工具自检（不算尝试），连续两次初始化失败就先停（`Claude_to_GPT_AugB_Main_Checkpoint1.md`）。GPT 随后做了两次不带提示词的 CLI 自检，都停在初始化（SQLite 状态库打不开、只读文件系统），没有计入实验日志；真实批次暂停，仍是 6/121 批，b003 的 attempt_2 仍可用，等 GPT 那边环境恢复、自检通过再续跑。**问题记录（11-06，Pan-Yan-0 转述）**：GPT 之前调度子 agent 时没有显式指定模型为 Luna，实际一直用的是 GPT6astra，这也是频繁罢工的原因；GPT 称已修复。日志里 `requested_model: gpt-6-luna` 是期望值，不是实际参数，证明不了实际模型。影响范围待 GPT 说明（哪些尝试是 astra、只是调度会话还是生成子 agent、修复从哪次开始、证据）；在此之前 v6 试点等已有生成数据的生成器记为"未确认"，不称 Luna 数据，放行结论不变。若混合来源，由 Pan-Yan-0 在合并前定逐条标注还是用 Luna 重跑。以后日志的 `requested_model` 只写实际传给工具的参数；清单 `source` 写死为 luna 的代码待改。详见 `code/augment_b/FULL_README.md` 第 9 节，交接 `Claude_to_GPT_AugB_Generator_Model.md`。**Pan-Yan-0 11-06 定**：检查点太慢，主轮继续用 GPT 调度 Luna，但不再发中间检查点，剩下 115 批一次跑完再交回（附上模型身份的说明）；只有失败批次累计到 6 批、出现不只差末尾换行符的提示词不一致、无法确认用 Luna，或拿不准要改协议时才中途停下交回（`Claude_to_GPT_AugB_Main_RunToEnd.md`）。补充轮同样一次跑完。提示词在 v6 试点时已冻结，之后只批量生成、最后统一检查。**随后 Pan-Yan-0 11-06 改定**：生成器换成 Claude Haiku 5.5 单一来源、全部重来，由 Claude 用 Agent 工具调度（每批一个全新子 agent、显式 `haiku`，提示词文件和回复由子 agent 自己读写）；先用 v6 的 15 个试点批次 smoke，自动停止条件全过才跑主轮 121 批；P1 只在合并后做一次，审核人 Pan-Yan-0 和 GPT；GPT 生成的数据只留档，`Claude_to_GPT_AugB_Main_RunToEnd.md` 作废。协议见 `code/augment_b/FULL_README.md` 第 10 节。**11-06 Haiku 生成完成**：smoke 15 批、主轮 121 批、补充第 1 轮 9 批全部第一次就合格，自动停止条件和逐折配额都通过，合并后 1,805 对、选用 1,794 对，统计在 `metrics/augment_b/haiku_full/`（带文本的文件 P1 前不进仓库）。P1 实际是 GPT 先标、Pan-Yan-0 审核修改，交回一份标注；Pan-Yan-0 定按共识标注计分（`P1C`，打开答案前提交）。**结果没有通过**：E 不一致 9/19（47%）> 30%，单向 4/18、O 0/18 通过；不放行、不训练 B1-Haiku。E 的不一致一半是一侧更具体被标成单向，一半是同一人的不同称呼或近义说法（和单词替换盲审里审阅人与金标的分歧同类），现在分不开审核标准和生成质量。全部原文和标注已归档 `metrics/augment_b/haiku_full/`。Pan-Yan-0 11-06 定继续改提示词、不放弃：保留 Haiku 的单向和 O，只用 v7 重新生成 E；先做 E 类审核标准对照（`e_control.py`，原训练集 E 20 对加干扰项 20 对盲标，答案封存 `debcbdb8…`，规则见 FULL_README 第 11 节），结果决定 v7 是否连人名变体一起禁掉。**对照结果**：原数据 E 的不一致率也有 44%（8/18），和 Haiku 的 47% 几乎一样，30% 的绝对门槛连原数据都通不过；数据集的 E 更接近"所指相同"（姓和全名、所指相同的上下位、时态和体的差别都算 E，多一个修饰语不算），审核人按逻辑蕴含判断。门槛问题交回 Pan-Yan-0（`metrics/augment_b/e_control/`）。**Pan-Yan-0 定**：比赛数据视为绝对准确，先不做 v7，改做按原数据校准的复审（`e_recheck.py`，60 对，答案封存 `a6d1b59d…`；Claude 先标 → Pan-Yan-0 审核 → GPT 对照参考卡再审；Haiku E 不一致率 ≤ 原数据 E + 15 个百分点即放行，单向和 O 沿用 P1C），见 FULL_README 第 12 节。Pan-Yan-0 全部采纳 Claude 的初标（`a33f31ec…`），实际是 Claude 初标 → GPT 对照参考卡复审两步，交接 `AugB_E_Recheck_v2.zip`。**复审通过，Haiku 数据放行**：Haiku E 不一致 3/20（15%），原数据 E 4/20（20%）；审核结果记为 `P1C-R`，`gate.json` 全部 pass，清单 `b380b4a6…`（`metrics/augment_b/e_recheck/`）。下一步：GPT 写训练队列跑 B0 / B1-Haiku / step+1（英语五折三种子，45 项），交接 `Claude_to_GPT_AugB_Haiku_Train.zip`。GPT 11-06 交回训练队列 `AugB_Haiku_Kaggle_v1.zip` 和 notebook（`GPT_to_Claude_AugB_Haiku_Train_Reply.md`）：本地验签、配额、隔离、4 项 CPU 测试通过；配方与 C0 相同（max_len 96、batch 32、lr 2e-5、warmup 10%、wd 0.01、最多 12 轮、fp16、按内层验证原方向 WF1 选轮）；step+1 = 原 fit 每行再复制一次；默认只跑预检和 SMOKE。还没在 GPU 上跑。Claude 已审 `train_augb.py`（`a4ceab00…`）：本地实跑预检（官方数据，五折行数、家族隔离、配额都对）和汇总（假预测走通官方评分和家族配对重抽样）；要求改 R1 推理改回 fp32（C0 不开 autocast）、R2 复用 C0 的 `apply_update` 处理范数溢出、R3 环境身份只校验卡、精度、模型、代码和数据（平台和库版本只记录）、R4 SMOKE 可重跑；建议多存反方向和内层验证概率、实际更新次数只报告。交接 `Claude_to_GPT_AugB_Haiku_Train_Review_v1.md`，训练暂停到 v2 交回。GPT 11-06 交回 v2（`AugB_Haiku_Kaggle_v2.zip` `58cde6d8…`，`train_augb.py` `385ed94b…`）：R1–R4、S1–S3 都改了；Claude 复审通过（官方数据预检通过，`test_local.py` 7 项里不需要 torch 的 6 项通过），可以跑 SMOKE；两条小建议（SMOKE 也走输出路径、选轮分数复核不用精确相等）等下次改代码时一起做。交接 `Claude_to_GPT_AugB_Haiku_Train_Review_v2.md`。Kaggle 上传 zip 会自动解压（路径形如 `/kaggle/input/datasets/<用户>/<数据集>/`），v2 notebook 只找 zip 会报错；Claude 改了 notebook 的找包那一格（先找含 `train_augb.py` 和 `data/gate.json` 的文件夹，复制到 `/kaggle/working`，再退回找 zip），记为 `AugB_Haiku_Kaggle_v2.1.ipynb`（`36e27b66…`），`train_augb.py` 不变。Kaggle T4 上预检和 SMOKE 已通过（三组各 2 次更新、0 次跳步，脚本强制 T4）。v2.1 只在最后一格打结果包，Pan-Yan-0 找不到 zip；Claude 改为 v2.2（`43d2f6b0…`）：SMOKE 后立即打包并给下载链接，正式任务无论成败都打包，增加从上次 Output 或结果 zip 恢复已完成任务的一格，正式 45 项用 Save & Run All (Commit) 后台跑（按西语 C0 的 58 分钟 / 15 项估计约 5 小时）。`train_augb.py` 不变。Pan-Yan-0 11-06 用 v2.2 Commit 跑完 45 项（约 4.4 小时）。**11-07 实验 B +1 结果**（`metrics/augment_b/train_plus1/`，Claude 用 `code/claude_recheck_augb.py` 从概率独立重建，和训练脚本汇总逐位一致；registry 登记 AUGB-B0 / AUGB-B1-HAIKU / AUGB-STEP1）：sym WF1 79.37 / 80.59 / 82.27%；B1 对 B0 +2.91 [+1.89, +3.89]、对 step+1 +1.68 [+0.73, +2.58]，三个种子都升，按事先规则扩充有效；O 类 F1 58.7→64.0。B0 比第五组 CE（A10）同种子低 0.9 个百分点，只作参照。下一步：+2 / +4，先由 Pan-Yan-0 定父家族使用上限（现在每个家族最多 2 次，+4 凑不够）和新增数据是否再审核。**11-07 至 11-09 规模轮**（`metrics/augment_b/haiku_scale/`）：Pan-Yan-0 定 +k 每个家族最多 2k 次、每轮重新分批、嵌套选用；Commit 1（B0、W2、W4、step+2、step+4、Dup×5）已交 Kaggle。Haiku 生成 r2 / s2a / r3 / r4 / s4a / s4b 共 512 次派发，`audit_haiku.py` 从 transcript 独立核对无问题；r4_b073 子 agent 多用了 `Bash`，按方案 A 整批作废（FULL_README 第 13 节）；r4_b045–b136 的 reasoning effort 因调度会话换模型变成 `high`，其余 `medium`，Pan-Yan-0 定保留并写进报告，之后派发显式指定 `medium`（第 14 节）。最终合并 m4 自动检查全过、每折配额补满。复审按第 15 节（单向和 O 按 P1C，E 按原数据校准，Claude 初标 → GPT 复审），交接 `Claude_to_GPT_AugB_Scale_Review_v1.zip`；放行后才打 Commit 2（B2、B4、G×5，105 项）。**11-10 Commit 1 结果**（`metrics/augment_b/train_c1/`，Claude 用 `code/claude_recheck_augb_c1.py` 从概率重建，150 项 0 问题；registry 登记 AUGB-C1-B0 / W2 / W4 / W2-BIAS / W4-BIAS / STEP2 / STEP4 / DUP-×5）：v3 的 B0 和 +1 的 v2 B0 逐条相同；W2 +0.18 [−0.62, +0.97]、W4 +0.78 [−0.11, +1.64] 不算提升，W4 减 log 4 后 +0.91 [+0.02, +1.80] 刚过线（未做多重比较校正）；step+2 / step+4 80.71% / 80.55%，原数据重复在 +1 后饱和；Dup×5 都不显著好于 Dup_R（+1 的 step+1，跨 v3 / v2 代码，只作描述），U4 显著更差；W4 的 O AP 比 B0 低约 2.5 个百分点（GPT 先算、Claude 复算一致），O 的 F1 升主要是判定偏向 O。**11-10 +2 / +4 复审通过**（Codex 交回 `Codex_to_Claude_AugB_Scale_Review_v1.zip`，改初标 12 条）：表 A 单向 3/19、O 0/20，表 B Haiku E 5/20（25%）对原数据 E 2/20（10%），正好在 +15 pp 门槛上；gate 全 pass，清单 `24704e11…`，全部原文和审核记录归档 `metrics/augment_b/haiku_scale/`。Commit 2 包 `AugB_v3_Commit2.zip` 已交 Pan-Yan-0。Codex 同时转达老师的 loss 对照（Focal γ=2、四类 macro soft Dice，各 15 项，对 B0），Claude 的方案回复见 `Claude_to_Codex_AugB_Loss_Reply_v1.md`；Codex 复核同意（`Codex_to_Claude_AugB_Loss_Review_v1.md`），已写入冻结计划变更记录。**Commit 3 已备好**：`code/augment_b_train/train_augb_v4.py`（v4.1 `d16e182e…`；Codex 验收 v4 后要求补严 B0 复用：`loss` 两边都须是 ce、torch / cuda / transformers 等于归档、正式队列只读重算判定、SMOKE 记梯度范数，交接 `Claude_to_Codex_AugB_Loss_v4.1.zip`，Codex 11-10 验收通过（`Codex_to_Claude_AugB_Loss_v4.1_Acceptance.md`：2.11.0+cpu 下 `-W error` 复跑预检和 loss 检查、B0 复用回归和正式选组都对；注意：SMOKE 之后手动换 torch / cuda / transformers 再直接走 CLI 时要先重跑 SMOKE，因为 full 的环境核验只看 GPU 和精度，用 notebook 的 Run All 不受影响）；由 `make_v4.py` 从 v3 精确替换生成，CE 那一行不变，Focal / Dice 在 fp32、autocast 外算），训练包 `AugB_v4_Commit3.zip`（`d8d602ae…`）和 notebook 已交 Pan-Yan-0，交接 `Claude_to_Codex_AugB_Loss_v4.zip`；loss 数值检查在 Kaggle 预检和 SMOKE 跑（Claude 这边没有 torch），B0 复用检查在 SMOKE 最后一步（`b0_reuse.json`，和 Commit 1 归档的 B0 SMOKE 逐值比），不通过就在正式队列里自动重跑 B0。**11-10 Commit 2 结果**（`metrics/augment_b/train_c2/`，Claude 用 `code/claude_recheck_augb_c2.py` 从概率重建，105 项 0 问题；registry 登记 AUGB-B2-HAIKU / B4-HAIKU / G-×5）：sym WF1 B2 83.21%、B4 83.14%；B2 − step+2 +2.50 [+1.50, +3.54]、B4 − step+4 +2.59 [+1.46, +3.76]，对 B0 也都过，+2 / +4 扩充有效；B4 − B2 −0.07，曲线在 +2 饱和；O 的 AP 升到 59.7–61.4%（B0 56.3%）。G_c − Dup_c 五个构成都过，G_c − G_R（跨代码）都没过，按构成 v2 没有构成值得生成，+2 后续不触发。**11-10 Commit 3 结果**（`metrics/augment_b/train_c3/`，Claude 用 `code/claude_recheck_augb_c3.py` 重建，30 项 0 问题；registry 登记 AUGB-FOCAL / AUGB-DICE）：B0 复用通过（SMOKE 三个预测文件和 Commit 1 归档逐字节相同），正式只跑 30 项；Focal − B0 +0.67，Bonferroni 97.5% [−0.19, +1.50]，不算提升；Dice − B0 −1.29 [−2.38, −0.25]，显著更差；不做叠加。实验 B 已登记的训练全部完成，下一阶段由 Pan-Yan-0 定。**11-10 边界与难 O 离线检查**（`metrics/diagnostics/boundary/`，`code/boundary_analysis.py`，事后描述）：mDeBERTa 错误 64–78% 是高置信度（严重过度自信，ECE 13），Qwen 约 27% 在 margin < 0.2；B2 / Qwen 4B / 8B 9 个 OOF 全错 174 条，抽查多像标注口径（Two ladies / Two women 金标 F 等）；三种子集成 +0.4 到 +1.6，Qwen 4B + 8B + B2 86.33%；G-O1 每组 O 召回都升、F / B 误判成 O 也最多，是先验偏移，不支持“生成的 O 太简单”。**11-10 Qwen 4B +1 扩充已登记**（Pan-Yan-0 定 4B、+1、A10；冻结计划变更记录）：`code/qwen4b_augb/` v1（代码身份 `30449eec…`，在 qwen8b_v5.5 上改，上游核心不变），Q0 = fit、Q-B1 = fit + B1 生成行、Q-step1 = fit×2；Q0 重跑不复用 JOINT-QWEN；阶段 1（Q0 + Q-B1，30 项）→ Q-B1 − Q0 通过且 Pan-Yan-0 确认后阶段 2（Q-step1）；CPU 20 项测试通过（含真实数据：上游五折 = 实验 B 划分）。**11-11 Codex 审查**（`Codex_to_Claude_Qwen4B_AugB_v1_Review.md`）：设计、两阶段和判定口径通过；v1 有 SMOKE 阻断（探测的 task.json 在裁剪前写入，记 4,380 行，回执却要求 16），已改为 v1.1（代码身份 `adc19fa0…`）：回执改按 actual_split.json 和 done.json 的 fit_rows 核对，compat new 路径只检查不补写，新增 worker.run(probe) 到回执的回归测试，21 项通过。阶段 1 分析按 Codex 口径：从 cache.jsonl 独立重建并和 OOF CSV 逐条比对，1,725 个家族共用抽样，未四舍五入判定。Codex 11-11 验收 v1.1 通过（`Codex_to_Claude_Qwen4B_AugB_v1.1_Acceptance.md`：独立复核探测回执、compat 能拦住注入的 TRAIN_BATCH=4、formal 按序调用 30 项、破坏兼容性产物后停在第一个任务前）。上传包 `a10_upload_qwen4b_augb_v1.1.zip`（官方仓库 `588968e`，缺 4B adapter 两个文件，由 Pan-Yan-0 补）已交，从包里解压跑 21 项测试通过。**11-12 阶段 1 完成**（`metrics/qwen4b_augb_s1/`，`code/claude_recheck_qwen4b_augb.py` 30 项 0 问题，registry 登记 Q4B-Q0 / Q4B-QB1）：sym WF1 Q0 85.02%、Q-B1 84.49%；Q-B1 − Q0 −0.53（−1.27 / +0.30 / −0.62），95% [−1.32, +0.26]，按事先规则未证实 +1 优于 Q0，不跑阶段 2；四类 F1 都没升，Q-B1 每轮更新翻倍、6 / 15 项选在第 1 或 2 轮；Q0 − JOINT-QWEN +0.65 [+0.23, +1.06]（同配方重跑差和效应同量级）。Codex 11-12 复核同意（`Codex_to_Claude_Qwen4B_AugB_Stage1_Review.md`：独立实现复算到 1e-12；README 把选轮和训练量分开写，Q-B1 选中权重经历的更新次数 12 / 15 对更多；删掉“三种子只能分辨约 1 个百分点”）；复核脚本按建议补了 COMPATIBLE / SMOKE 回执、COMPLETE 身份和封存文件、逐轮 val WF1 重算三块，0 问题，注入错误能被拦下；A10 实例在三个包核对后可释放。**11-12 研究对齐**（Codex 11-11 的《研究对齐 v1》，Claude 回复 `Claude_to_Codex_BoundaryReview_v1.zip`）：先做带依据的盲审（`code/boundary_review.py`，`metrics/diagnostics/boundary_review/`：48 个家族，顽固错误 S1 16、O 边界混淆 S2 16、判对对照 S3 16；排除 Claude 看过金标的 169 个家族；对照参考卡判断；表 `9675d8f1…`，答案 `c7fcd43f…`，Claude 标注 `c7477380…`，Codex 标注 `aca03f10…`）。**11-12 盲审结果**（Codex 已独立重算）：按 R1，S1 两人一致但≠金标 7/13、S3 0/14，触发第一条读法，写法限定为"在这批抽样家族和参考卡口径下，S1 更常见两位审阅人一致但与金标不同"（S3 只配了部分对照、两位审阅人都接触过部分示例），顽固错误不作为改动目标；7 个里 5 个是金标 O、两人按"一侧更具体"类比参考卡的单向示例判，6 个两人的标签等于 Qwen 4B 多数票；这些例子只说明判断依据，不写成噪声比例或纯文本模型上限；S2-EFB→O 的逐人比例（Claude 6/8、Codex 0/8，近义、介词或词形边界的分歧）是事后描述、放在局限段；R2 无现象达到阈值（MOD 2/9、PREP 2/7、SYN 0/6），暂停定向增广；R3 分母是审阅票。Codex 计分前指出 R1 浮点边界（0.7 − 0.4 = 0.2999…），已改为整数交叉相乘，测试 5 项，实际结果离边界很远。再在 B0 数据上做三臂（OLat4：z_O = logsumexp − log 4，由 B0 的 O 行加独立生成器的扰动初始化；NLIinit：`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` `b5113eb3…`，先照 B0 构造再覆盖 `deberta.*` 和 `pooler.*`，保留 B0 的 4 类头和 tokenizer；BiBCE：CE + 0.25·½[BCE(m_AB) + BCE(m_BA)]，fp32、autocast 外），各和 B0 配对，Bonferroni 98.33%、20,000 次重抽样；Codex 11-12 确认：A 扰动标准差 0.002、沿分量维度中心化、偏置原样复制，新建 Linear 和扰动都隔离随机数状态；B 同时载入 `deberta.*` 和 `pooler.*`，解释为整套 NLI 表征初始化；C 两个方向取平均、λ=0.25。**v5 已写好**（`code/augment_b_train/make_v5.py` `ba43eeaa…` 从 v4.1 精确替换生成 `train_augb_v5.py` `d643e8cb…`；训练包 `AugB_v5_Commit4.zip` `32e16c10…`、notebook `AugB_v5_Commit4.ipynb` `d87d107f…`；交接 `Claude_to_Codex_AugB_v5_Commit4.zip`）：三臂先照 B0 构造再在 `torch.random.fork_rng` 里改，核对全局随机数状态不变；SMOKE 加 OLat1（K=1、无扰动），必须和同一次运行的 B0 SMOKE 逐值相同才放行正式队列；`arm_checks`（CPU 预检 + GPU）和 `arm_init_checks`（真实模型：B0 新初始化的键、spm 哈希相同、OLat 初始概率差）写进 `arm_checks.json`；OLat4 的 `done.json` 记各分量责任度。**Codex 11-12 复核 v5**（`Codex_to_Claude_BoundaryReview_v1_and_AugB_Commit4_Review.md`）：重建字节一致，torch 2.11.0+cpu、transformers 4.57.6 下 `-W error -I` 预检和三臂数值检查通过；OLat K=1 在真实 B0 头和 CPU fp16 autocast 下 logits、CE、梯度逐值相同，K=4 零扰动最大概率差 1.21e-8，0.002 扰动的 p_O 最大差 3.89e-5；NLI 实载 200 个张量、278,809,344 个参数，B0 只新初始化 pooler 和 classifier，两份 spm.model 相同。要求补两处：worker 和续跑任务的 torch / cuda / transformers 核验，`arm_checks.json` 子回执完整并绑定本次 SMOKE。**v5.1**（`make_v5.py` `720c89e2…` → `train_augb_v5.py` `e62f75e1…`，训练包 `AugB_v5.1_Commit4.zip` `79d2cf12…`、notebook `AugB_v5.1_Commit4.ipynb` `44a033ef…`；交接 `Claude_to_Codex_AugB_v5.1_Commit4.zip`）：两处都补了，训练路径不变，`test_v5_receipts.py` 9 项（删减回执、换一次运行的回执、重跑 SMOKE 后沿用旧回执、版本不同的续跑任务都被拦下）；复核脚本 `code/claude_recheck_augb_c4.py` 已写好，`--selftest-c3` 用同一重抽样函数重算 Commit 3 两条主比较，和归档逐位一致。**Codex 11-12 放行 v5.1 跑 SMOKE**（`Codex_to_Claude_AugB_v5.1_Commit4_SMOKE_Review.md`：用模拟 SMOKE 产物跑真实 `main()`，45 / 60 项分支、版本漂移、删减或换来的回执、改 OLat1 概率都在首项任务前拦下），另提回执数值明细可缺（`olat_K4_zero = {}` 能通过）和复核脚本三处（归档 B0 缺 `loss` 字段导致 45 项误报、重跑 B0 跳过核验、失败仍退出 0）。**v5.2**（`make_v5.py` `244f35b9…` → `train_augb_v5.py` `7f3e01b3…`，训练包 `AugB_v5.2_Commit4.zip` `9adf175f…`、notebook `AugB_v5.2_Commit4.ipynb` `660ceacf…`；只改回执读取，每个验收数值必填、有限、在阈值内；`test_v5_receipts.py` 10 项）和复核脚本修订（`claude_recheck_augb_c4.py` `3aff7340…`：有效损失缺字段按 CE、重跑 B0 对照归档 B0 走完整核验并计入 60 项、有问题退出码 2；模拟复用 / 重跑包 0 问题、注入 3 个错误正好报 3 个）交接 `Claude_to_Codex_AugB_v5.2_Commit4.zip`。为免在 v5.1 跑完 SMOKE 后因代码哈希变化重跑，Pan-Yan-0 直接用 v5.2 跑 SMOKE。**Codex 11-12 验收 v5.2 通过**（`Codex_to_Claude_AugB_v5.2_Commit4_Acceptance.md`：从 v5.1 应用补丁重建出 `7f3e01b3…`，10 项测试、`-W error -I` 预检通过；真实 `main()` 上三个空子对象负例都在首项任务前退出，45 / 60 项分支和既有负例不变；复核脚本在复用 / 重跑 / 注入错误三种模拟包上分别 0 / 0 / 3 个问题，退出码 0 / 0 / 2）。记一处格式细节不改代码：回执里 `K` 和 `perturb_std` 按数值相等判断，布尔值 true / false 会被当成 1 / 0 接受；SMOKE 实际写出的是数值，用于验收的统计量都拒绝布尔值。**11-12 Kaggle T4 SMOKE 通过**（v5.2 `7f3e01b3…`，torch 2.11.0+cu128 / CUDA 12.8 / transformers 4.57.6）：五组各 2 次实际更新、0 跳步；B0 复用 pass（三个预测文件和 Commit 1 归档逐字节相同），OLat1 和 B0 逐字节相同、两步梯度范数相同；回执用 v5.2 校验器复核 0 问题；OLat4 初始 p_O 最大差 3.9e-5，两步后分量责任度仍接近均匀；NLIinit 载入 200 个张量、278,809,344 个参数，只保留 B0 的 classifier，前向配置无差别，spm 哈希相同；BiBCE 辅助项初值约 0.698（≈ ln 2）。**11-12 Commit 4 结果**（`metrics/augment_b/train_c4/`，`claude_recheck_augb_c4.py` 45 项 0 问题，registry 登记 AUGB-OLAT4 / NLIINIT / BIBCE）：sym WF1 B0 79.37 / OLat4 79.15 / NLIinit 82.59 / BiBCE 79.63%；NLIinit − B0 +3.22（+3.51 / +2.72 / +3.44），Bonferroni 98.33% [+1.56, +4.86]，算提升，四类 F1 都升（O +5.6），O 误判约减三成，O AP 约 +3；NLIinit − B2 −0.6 [−1.8, +0.5]（描述）；OLat4 −0.22 [−1.06, +0.63]，训练后 4 个分量责任度都约 0.25、熵≈log 4，没有分化，只说明这种初始化的潜在头无用；BiBCE +0.27 [−0.63, +1.16]。**Commit 5 已登记**（Pan-Yan-0 11-12 确认，冻结计划变更记录）：B2-NLIinit 对同一队列重跑的 B2-Haiku（Commit 2 没留 B2 的 SMOKE，无法核对复用，所以 30 项，T4 x2 墙钟约 2.6 小时）；sym WF1 双侧 95%、20,000 次家族重抽样；代码 `train_augb_v6.py` v6（`bdf1d2a6…`，`make_v6.py` 从 v5.2 生成，训练路径不变）交 Codex（`Claude_to_Codex_AugB_v6_Commit5.zip`）；Codex 11-12 复核（`Codex_to_Claude_AugB_v6_Commit5_Review.md`）同意设计，指出正式入口两处缺口（CE 组 SMOKE 一步也能放行；改了 +2 数据并重绑 gate 后旧 SMOKE 仍能放行），已改为 v6.1（`50c80062…`，`make_v6.py` `99c80d22…`）：SMOKE 两组都要 2 次实际更新，正式入口核对两组 2 次实际更新、0 跳步、走过输出路径、V3 数据哈希等于当前数据、B2-NLIinit 载入了固定检查点；`test_v6.py` 12 项；复核脚本 `claude_recheck_augb_c5.py`（`2cb039c3…`，`--selftest-c4` 和归档差 0，另报逐 fold / seed 实际更新差）；训练包 `AugB_v6.1_Commit5.zip`（`1f9ebecb…`）和 notebook 已交 Pan-Yan-0，交接 `Claude_to_Codex_AugB_v6.1_Commit5.zip`；**11-12 Kaggle T4 SMOKE 通过**（v6.1 `50c80062…`，torch 2.11.0+cu128 / CUDA 12.8 / transformers 4.57.6）：两组各 2 次实际更新、0 跳步，走过输出路径，V3 数据哈希和 Commit 2 一致，B2-NLIinit 载入 200 个张量、278,809,344 个参数、只保留 classifier；Claude 用包内 `commit5_smoke_problems` 复核 0 问题。**11-12 Commit 5 结果**（`metrics/augment_b/train_c5/`，`claude_recheck_augb_c5.py` 30 项 0 问题，registry 登记 AUGB-C5-B2 / AUGB-B2-NLIINIT）：sym WF1 B2 83.21%、B2-NLIinit 83.59%；B2-NLIinit − B2 +0.38（+0.52 / +0.23 / +0.40），95% [−0.61, +1.36]，按事先规则不算提升，在 +2 数据上没有检测到 NLI 初始化的额外增益；重跑的 B2 15 项和 Commit 2 逐字节相同；O 误判减少（判成 O 1530 → 1438 行），O F1 63.7 → 64.8；英语工作配方仍是 B2，实验 B 已登记的 mDeBERTa 训练全部完成；**11-12 研究对齐 v2**（Pan-Yan-0 认为目前缺少方法上的新意，同意试三个从任务出发的方向）：方向一把四分类拆成方向蕴含（P(E)=σ_ab·σ_ba 等，方向一致性由结构保证），提议为 Commit 6（DEC、DEC-H 对 NLIinit，B0 数据，Bonferroni m=2），设计草案交 Codex（`Claude_to_Codex_ResearchAlignment_v2.md`）；方向二（检索对齐口径）字面 TF-IDF 预检（`code/knn_signal_check.py`，事后描述）在 174 条顽固错误上只有约 21% 准确率、接近乱猜，暂缓，要做需换语义检索；方向三（反事实最小对）等导师讲完再设计；MoE / 按类专家如要做必须和同算力的种子集成比；其他赛道是否试 NLI 初始化由 Pan-Yan-0 和导师定（检查点不含巴斯克语）。**Codex 11-12 审阅 v2**：同意 DirEnt 作为 Commit 6；指出同比赛已有公开先例（Le Duc Minh 的 entail2：同型号 NLI 检查点、保留三类头、双向算蕴含后按同一公式合成四类、四赛道双向训练），本次只回答单方向监督的分解相对 NLIinit 有无收益；背景写成“盲审发现这批高难家族中存在标注口径分歧”，“E 更接近所指相同”只是假设；单条 O 先不补反向监督，另报 O→F/B、E→F/B；选轮用各模型原生四类内层验证 WF1（NLIinit 沿用单顺序），s 恰好为 0 算不蕴含；两臂都过线只选点估计增益更大的一臂进 +2（同分取 DEC）；语义检索预检排在 DirEnt 之后（冻结 NLI 编码器、mean pooling、k=4、门槛比字面高 5 个百分点）；MoE 和种子集成比要分别报训练和推理成本。**Commit 6 已登记**（冻结计划变更记录）：`train_augb_v7.py` v7（`8027e60d…`，`make_v7.py` `d4bc861e…` 从 v6.1 生成，其他组训练路径不变）；NLIinit 复用要求 SMOKE 和 Commit 4 归档逐格相同、15 个 Commit 4 任务身份核对通过，否则同一队列重跑（30 / 45 项）；`dec_checks.json` 回执绑定本次 SMOKE；`test_v7.py` 10 项；复核脚本 `claude_recheck_augb_c6.py`（`--selftest-c4` 差 0，模拟包 0 问题、注入 3 个错误报 3 个）。写复核脚本时发现 c1–c5 复核里“反向行的 r_p 等于反向行的 p”这条检查因为把 reverse_pair_id（其实是 instance_id）再经 pair_id 映射而从没执行；补算 Commit 4 / 5 归档 38,460 行，差都是 0，c6 已改对。交接 `Claude_to_Codex_AugB_v7_Commit6.zip`。**Codex 11-12 验收 v7**（`Codex_to_Claude_AugB_v7_Commit6_Acceptance.md`）：重建、10 项测试、torch CPU 预检、真实检查点上的 DEC-H 初值（最大差 1.19e−7，标签顺序 entailment / neutral / contradiction）都通过；要求正式 T4×2 队列前补两处：两个并行 worker 争写共享预检报告和 `official_reference.sha256`，正式入口只查回执的键不查值；另有 `-W error` 下 `dec_checks` 的 float 警告和复核脚本 ECE 的命中口径。**v7.1**（`make_v7.py` `fa70b3b9…` → `train_augb_v7.py` `d7cf722e…`）都改了：并行 worker 只写自己的 `workers/w<k>/`，共享文件只读核对；回执逐项按 `dec_checks()` 的通过条件查值；`test_v7.py` 13 项（新增并行写入、两个真实子进程同时启动和续跑、26 种回执篡改），用旧 v7 跑新测试有 4 项失败；复核脚本 `7aa90a4b…`。训练包 `AugB_v7_Commit6.zip`（v7.1，`f0c05430…`）和 notebook（`136f2f59…`）已交 Pan-Yan-0 直接跑 SMOKE，交接 `Claude_to_Codex_AugB_v7.1_Commit6.zip`。**Codex 11-12 验收 v7.1 通过**（`Codex_to_Claude_AugB_v7.1_Commit6_Acceptance.md`，对 `7b15826`）：训练包 56 项哈希一致，包内 36 份 Commit 4 NLIinit 文件和归档逐字节相同；13 项测试在包内通过（两个子进程重叠启动、26 种回执反例、缺共享文件）；torch 2.11.0+cpu 下 `-W error -I` 预检通过；复核脚本自检差 0、ECE 例子 0.75。**11-12 第一次 Kaggle SMOKE（v7.1）**：NLIinit 2 次更新、0 跳步，三个预测文件和 Commit 4 归档逐字节相同（可以复用）；DEC 2 次更新；DEC-H 第 1 步 fp16 梯度溢出被 GradScaler 跳过（65536 → 32768），只有 1 次实际更新，SMOKE 按门槛停下。这是 fp16 开头的缩放回调（B0 和 NLIinit 的历史正式任务每项也跳过 2–4 步），不改 GradScaler。**v7.2**（`make_v7.py` `c66afbcb…` → `train_augb_v7.py` `80396fd9…`）：只对 DEC / DEC-H，SMOKE 从最多 4 个 batch 里取到 2 次实际更新，开头最多 2 次跳步、如实记录，正式入口核对跳步数、trace 和 done.json 一致、最后一步是实际更新；NLIinit 和其他组的 SMOKE 不变；`test_v7.py` 15 项；复核脚本 `4c937534…`；训练包 `AugB_v7_Commit6.zip`（v7.2，`6d3d6e4a…`）、notebook（`1b895118…`）已交 Pan-Yan-0 重跑 SMOKE，交接 `Claude_to_Codex_AugB_v7.2_Commit6.zip`。**Codex 11-12 同意 v7.2 跑 SMOKE**（`Codex_to_Claude_AugB_v7.2_Commit6_SMOKE_Review.md`：15 项测试和 `-W error -I` 预检通过；用真实 GradScaler、AdamW 和 `apply_update` 注入 0–3 次非有限梯度，0–2 次跳步都能在 4 步内完成 2 次更新、跳步时参数和 scheduler 不前进，3 次跳步只有 1 次更新）；指出复核脚本的 SMOKE 校验比正式入口松（`updated / updated / skipped`、`unknown / updated / updated` 能通过），已改为和正式入口同样的条件（复核脚本 `e9159e58…`，训练代码不变）；交接 ZIP 的 SHA256SUMS 把自己列了进去，以后生成清单时排除它。**11-12 Kaggle T4 SMOKE 通过**（v7.2 `80396fd9…`，torch 2.11.0+cu128 / CUDA 12.8 / transformers 4.57.6；Claude 用训练脚本自己的 `commit6_smoke_problems` 离线复核 0 问题）：NLIinit 2 次更新、0 跳步，三个预测文件和 Commit 4 归档 0 格不同，15 个 Commit 4 任务身份核对通过，正式队列复用、30 项；DEC 2 次更新、0 跳步；DEC-H 先跳 1 步再 2 次更新（梯度范数约 13，DEC 约 3–4）；DEC-H 实卡初值和原检查点蕴含概率最大差 2.0e−8，标签顺序 entailment / neutral / contradiction；数值回执全部通过；零样本 DEC-H 3,042 行（描述：WF1 62.96%、SoftCons 81.67%）。**11-12 Commit 6 结果**（`metrics/augment_b/train_c6/`，`claude_recheck_augb_c6.py` 45 项 0 问题，registry 登记 AUGB-DEC / AUGB-DEC-H）：sym WF1 NLIinit 82.59 / DEC 80.90 / DEC-H 80.52%；DEC − NLIinit −1.69（−2.25 / −1.61 / −1.22），Bonferroni 97.5% [−2.80, −0.55]；DEC-H −2.07 [−3.21, −0.91]，都显著更差，不进 +2，方向一结束；四类 F1 都降，E → F/B 298 → 400、O → F/B 383 → 472（三种子合计，O 全是单条，判成 B 比判成 F 多升），SoftCons 由结构保证升、HardCons 降、ECE 升；单条 O 补反向监督是否算利用配对结构待 Pan-Yan-0 定，不自动做。**Pan-Yan-0 11-12 定**：不利用配对结构（单条即 O 等），所有方法只用文本，dev 榜上靠结构的结果不作对照；反事实方向不等导师讲，先自己设计实验，有结果再和导师讨论。研究对齐 v3（A 口径对齐重标、B 反事实最小对）、v3.1（C PiSSA）、v3.2（先校准再集成：校准有效但三模型集成只 +0.09 [−0.27, +0.45]，`metrics/diagnostics/boundary/` 第 5 节；提议 A1 加 B2-LS，m = 3）已交 Codex；Codex 11-12 合并审阅同意主体（LS 只平滑生成行，A0 先盲审分歧再定教师臂，A2 主比较 Q-B1-RL − Q-B1，CF-HEAD 首轮只从共享中心概念的 E 出发），O 阈值诊断显示调阈值不提分、O 的问题在排序；Claude 回复研究对齐 v4（A0 入队规则、m 固定 3、B 试点 60 对审 40 对、A10 第一次跑 A0 + Q0 短程复现 + PiSSA SMOKE）；Codex 11-12 同意，另要求 B 的父样本只按本折 fit 的金标和文本筛选（其他折 OOF 的模型见过本折留出行）、IMIT-k 用同一 E 父样本和显式目标标签，SL 补零票时的处理；**已登记进冻结计划变更记录**（11-12 研究对齐 v4 条目），**11-12 改为尽量用 Kaggle**（阿里云本月 52、新增 100（3 个月）、12 月 250，之后没有）：研究对齐 v4.1（A0 / B1 改在 T4 fp16 推理并改复现门槛，C 改为 Q0-T4 与 Q-PiSSA 同队列重跑）交 Codex，交接 `Claude_to_Codex_A0_Kaggle_v1.zip`；A0 推理包 `code/a0_teacher/`（`teacher_a0.py`，CPU 9 项测试），Codex 11-12 同意 v4.1（已登记进冻结计划），A0 改为 v1.1（概率校验、续跑和打包共用回执验收、`full` 两个 worker 都返回 0 才打包，CPU 15 项测试，交接 `Claude_to_Codex_A0_Kaggle_v1.1.zip`），Codex 11-12 CPU 验收 v1.1 通过（`Codex_to_Claude_A0_Kaggle_v1.1_CPU_Acceptance.md`；续跑成功的包可能带旧 `A0_FAILED.json`，以本次 `A0_SUMMARY.json` 为准），Pan-Yan-0 用 v1.1 跑 SMOKE，全量放行看真实 T4 回执；A10 剩余额度（本月 52、新增 100 三个月内、12 月 250，明年没有）留给 Qwen3-8B 和最终提交；B 试点（`code/cf_pilot.py`）6 批 Haiku 合格，盲审分两包（`Claude_to_Codex_CF_Pilot_Review_Stage1/2.zip`），带文本的试点文件审核前不进仓库。估时：45 项约 2.9 GPU 小时（T4 x2 墙钟约 1.5 小时），之前说的"不超过 4.5 小时"偏高。"约 330 万对"是模型卡的总数 3,287,280，以后用模型卡的原始数字。子 agent 的初始上下文里有本文件和 CLAUDE.md，已在 haiku_full / train_plus1 / haiku_scale 的 README 写明。

GPT v3调试交回：`augb_pilot_v3_claude.zip` 清单哈希一致、CPU 9项测试实跑通过；按本次单次调试要求，`debug_b02` 仅调用一次全新GPT6Luna，返回13对（E3/F6/O4），格式检查因少1对F失败，尚未用格式重试额度。原始输出、日志及语义疑点在 `GPT_to_Claude_AugB_v3_Debug.zip`，由Pan-Yan-0转交Claude。重点复核帐篷下/内部、captain/leader及数量O的任务口径。未开15批、未训练；等待Claude检查后继续。Claude 已检查 `debug_b02`：13 对没有明确错误，3 对是边界情况（captain/leader、帐篷下/里、数量对）。数量对按数据集的标法应是蕴含，生成器标成了 O；这条规则是从全部训练标签归纳的，写进提示词会破坏可用折隔离。所以改为 v4（`augb_pilot_v4_claude.zip`，模板 `dbfc1d32…`，生成脚本 `2ce2ef55…`）：提示词禁止只差数量的对，并要求作答前自查对数，代码不变，包内 9 项测试通过。没有用 v3 剩下的那次重试。下一步：GPT 跑 `debug_b03`。

实验 B 生成来源（Pan-Yan-0 11-03 定的，GPT 11-04 同意，已写入冻结计划变更记录）：先只用 GPT6Luna 子 agent，由 GPT 执行生成；DeepSeek API 暂缓（导师点名 DeepSeek，以后用同一批次文件补作第二来源），本地 Qwen-4B 和 Claude 子 agent 都不做。+1 阶段为 B0 / B1-Luna / step+1 共 45 项。分工建议：Claude 写分批、配额、提示词（只用手写短语对调试）、过滤、去重和清单，GPT 分发批次给全新子 agent 并原样交回输出和调度记录；只有一个来源无法隐藏，生成方不做主审核。按实际划分估算 +1 约需 3,040 行（约 1,770 对、约 150 批），试点约 15 批。交接文件 `Claude_to_GPT_B_Generation_v1.md`（由 Pan-Yan-0 转交）。GPT 11-04 同意单来源和分工（`GPT_to_Claude_B_Generation_Review_v1.md`），并要求补齐接口：上下文家族签名、按对配额、逐尝试原文和日志、格式错误定义，以及单人审核下的停止规则。Claude 已交试点批次包 `augb_pilot_v1_claude.zip`（代码在 `code/augment_b/`，划分哈希 `dbd8b571…`，模板哈希 `64029905…`）。包里有 15 批共 195 个父样本，每批 14 对（E 3 / 单向 7 / O 4）；另有一个参考对全手写的调试批次 `debug_b00`，先跑它确认提示词。Pan-Yan-0 在看输出前选定审核协议 P1（他和 Claude 各自独立标注 60 对）。GPT 跑了调试批次 `debug_b00`：格式合格，但照抄了参考短语、句式雷同；另指出测试在包内跑不起来、`null` 被判为合格、空标注表不会触发停止、缺少统一的放行门槛。Claude 改为 v2（`augb_pilot_v2_claude.zip`，模板 `8c1f0716…`）：提示词加多样性要求，过滤加"照抄父样本短语"一条；`gate.json` 统一放行；标注表必须完整合法，每类可判定项不足 10 个就不放行；补充批次用确定性算法规划（`supplement.py`）；每批最多尝试 3 次；测试在包内重建并逐字节比对，共 8 项通过。+2 / +4 继续条件、试点数据去向和重试上限已经双方同意。GPT 已复核 v2：包清单哈希一致，CPU 8 项测试通过；`debug_b01` 第一次漏全部标签，按规则仅格式重试一次，第二次 E3/F7/O4 格式通过，但 E 中出现新增地点和时间变化等语义问题，不能据此放行。两次原文与完整调度日志已交接，实际请求 GPT6Luna，工具未单独返回模型身份。另有最小 CPU 复现：旧 review_result 可覆盖新 report 的 pending；补充规划允许第三轮且预算未跨轮累计。日志原文哈希必填等仍需补校验。交接包 `GPT_to_Claude_AugB_v2_Debug_Review.zip` 由 Pan-Yan-0 转交；未修改原包、未开15批、未启动GPU训练。下一步由 Claude 独立检查手写调试语义、修复工程边界并补测试，再确认正式试点；P1 和既定实验对照不变。Claude 已交 v3（`augb_pilot_v3_claude.zip`，模板 `b097d45b…`，生成脚本 `40140d7d…`）：独立检查 `debug_b01` 的 14 对，1 对 E 明确错误、1 对边界，单向和 O 都成立；提示词收紧 E / F 定义（不能新增地点、时间、参与者，相关不等于蕴含），手写参考的隧道一对改为煤矿；审核结果绑定 manifest、review_sheet、review_key、report 的哈希，重建时清空旧审核、只认 P1；补充规划只允许两轮、预算跨轮累计、父样本按游标循环；日志的原文哈希、失败原因、ISO 时间、`has_partial_output` 必检，合格后不得追加任何尝试。CPU 测试 9 项在包内全部通过。下一步：GPT 跑 `debug_b02`，交回 Claude 确认后再开 15 批；训练入口要核对 `gate.json` 绑定的清单哈希。

实验B类别加权补充草案：Pan-Yan-0要求在不新增数据下检验弱类损失加权并严格控制变量。GPT已整理`GPT_to_Claude_B_ClassWeight_Plan_v1.md`（由用户转交）：原四分类W1/W2/W4仅改变O相对权重1/2/4，固定数据顺序、初始化、最大更新日程及选轮规则；附同预算logit偏置诊断。第一阶段45项、合格B0可复用则新增30项；生成+1与Dup的四组加权交互仅作条件扩展。尚待Claude核对C0实现、B0复用及类别构成v2重合；未改冻结方案，未启动新增GPU/API。二分类专家另立后续实验，不混入本次权重比较。11-01 Claude 已回复（`Claude_to_GPT_B_ClassWeight_Reply_v1.md`）：同意设计和 W2/W4、四组边界；英语没有 T4 fp16 的 CE 结果，W1 即实验 B 的 B0（λ=1 须走原 CE 分支、逐位相同），新增 W2/W4 共 30 项；D1=Dup_R=step+1、G1=B1，扩展只新增 D2/G2；补充：事先声明的解析偏置对照（±log λ，不需选择）、内层验证 O 只有 32–44 行所以网格偏置同时按 WF1 和 O-F1 报告、补存内层验证反方向和第 12 轮 logits、不能用 `F.cross_entropy(weight=)`。等 GPT 回应后再改冻结计划。GPT 随后接受解析偏置对照，并更正两处解释（撤销偏置后分数提高不说明排序能力提高；非 O 权重下降不等于 AdamW 的有效学习率下降），Claude 已接受。

实验 C（剪枝＋蒸馏，候选，11-02 提出，排在 A、B 之后）：目标是以更低的实测推理成本保留 NLI 能力，不以超过原模型为唯一标准。单模块 C0 原模型 / C1 剪枝不训练 / C2 剪枝＋CE 恢复 / C3 剪枝＋CE＋教师蒸馏，关键比较 C3−C2（同一剪枝权重、同数据顺序和步数，只改训练目标）；之后再用 2×2（完整 / 压缩 × 普通 / 加权）检验组合。称为受 Minitron 启发的任务定向压缩，不称复现。Claude 意见（`Claude_to_GPT_C_PruneDistill_Reply_v1.md`）：剪枝方法事先固定为按层删除（BI），比例只在内层验证上定；训练行上的教师概率几乎等于硬标签，C3−C2 可能分不开，交叉拟合教师另作一组；恢复训练沿用 4B LoRA SFT 配方；成本在同卡同精度同批次组成下实测；先 4B、fold 0 试点。GPT 复核后提出 7 条修正（C0/C1 也按 15 份源检查点评估、固定删 25% 层并保护首末层、KD 公式显式写出、AP 不变只限单方向等），Claude 已整理为计划 v2（`实验C计划_v2_Claude.md`）：删 floor(36/4)=9 层；C3 目标为 (1−β)·CE+β·T²·KL（β=0.5，T=2，与已有 KD 同一形式，已核对 `kd_loss`）；教师直接复用联合验证的 15 份 `teacher.jsonl`；BI 用该折拟合部分前 256 行；GPT 复核 v2 后主体同意，另提 6 项（训练前重置随机数并核对初始化和样本顺序哈希、BI 用固定种子 2026 抽 256 行且按有效 token 等权汇总、C3 是"恢复目标替换"且 CE 为全词表、教师缓存要核对源身份和概率语义、合并验收要事先定门槛并用 fit 清单、补报 C3−C0），Claude 已全部并入计划 v3（`实验C计划_v3_Claude.md`）。合并验收门槛：64 条 fit 输入上合并前后 d ≤ max(2·d₀, 0.02)、k ≤ k₀+1（d₀、k₀ 为只换分批的噪声）。Claude 用 CPU 核对了 15 份教师缓存：身份文件含折号、种子、选中轮、adapter 和划分哈希；每份覆盖拟合部分两个方向和内层验证、无重复、无零概率、概率为 T=1；导出时和训练时缓存最大类 100% 一致、概率最大差 0.113（bf16 分批差异）。仍是候选计划，未启动任何训练。

GPT对v5.1的复核：接受内网安装及未覆盖上游默认旧入口的修订，但暂不确认当前包可直接上A10。8项CPU测试与shell语法检查通过；另复现prepare_upload输出在源码目录时把ZIP自身打入包（v5遗留）及formal目录不存在时status提前退出。还需补同ROOT并发锁、纠正仅凭回执文件存在显示通过、补全恢复源码所需baseline。详细交接为`GPT_to_Claude_Qwen8B_v5.1_Review.md`，由Pan-Yan-0转交；待Claude回传v5.2及新增测试后复核。未运行GPU、未确认重建版SMOKE通过，未启动15项正式训练。

8B代码恢复：用户澄清A10完成的是SMOKE，尚无15项正式训练完成证据。原v4源码找不到，已授权重建；新实现见[Qwen8B v5](ML_note/实验/SemEval_DiCo-NLI/code/qwen8b_v5/README.md)。原上游训练核心字节保持不变；v3上传包是失败记录，不能替代v4通过记录。v5完成CPU检查，尚待Claude独立审阅与新A10验收；正式入口要求本版SMOKE和4B兼容性回执。源码随结果归档；不得用旧v4回执放行重建版。11-01 Claude 已审阅 v5：无阻断问题；改了两处并加一个会话外壳，记为 v5.1（代码身份 `dbb1cce9…`，交接包 `Qwen8B_v5.1_claude.zip`，由 Pan-Yan-0 转交）：setup 的 torch 改走阿里云内网镜像；compat 的旧入口改为不加任何覆盖的上游默认，并比较两边的 `fingerprint()`。GPT 同意这两处，另提 R1–R4（上传包含自身、同一 ROOT 并发写入、status 误判、源码快照缺缓存），Claude 已改为 v5.2（代码身份 `586ccf00…`，交接包 `Qwen8B_v5.2_claude.zip`），上传包 `a10_upload_v5.2.zip` 已发给 Pan-Yan-0（还差 4B adapter 两个文件）。等 GPT 确认 v5.2 后提交进仓库；A10 实例按 check（验收）→ formal（15 项）的顺序跑。11-01 A10 上 v5.2 验收：环境与 10-28 一致，8B 探测和 SMOKE 通过（峰值 16.99 GiB），4B 零样本核对失败；实测原因是 v5 没复现历史缓存的分批方式（只算 dev、原顺序分批时最大差 0.354、1 个最大类不同；按历史的 fold0+dev 长度排序每 64 个一批为 0）。Claude 改为 v5.3（只改 `compat.py`，代码身份 `b3807817…`，交接包 `Qwen8B_v5.3_claude.zip`），门槛不放宽；旧 ROOT 改名为 `dico_qwen8b_v5.2_compatfail` 保留，新 ROOT 重新验收；v5.3 验收中零样本核对通过，但 old 阶段因 `smoke_pairs(fit,16)` 在 fold 0 上只能得到 17 行而失败（v5 遗留，Claude 审阅时漏掉）；Claude 改为 v5.4（仍只改 `compat.py`：先取 15 行再按固定顺序补 1 条单条 O，恰好 16 行、四类都有；代码身份 `74f537c3…`，交接包 `Qwen8B_v5.4_claude.zip`，取代 v5.3）；旧 ROOT 改名 `dico_qwen8b_v5.3_compatfail`，新 ROOT 重新验收。GPT 暂时不可用，Pan-Yan-0 11-01 决定不等：v5.4 验收的两个回执都通过后自动开始 15 项正式任务（`patch_v54_auto.sh`），GPT 恢复后对 v5.2–v5.4 事后复核；训练路径（上游核心、worker）和 GPT 已复核的版本相同，v5.2–v5.4 只改了外围和兼容性检查。11-01 18:48 v5.4 验收全部通过（SMOKE、4B 零样本按历史分批差 0、新旧入口 3 次更新逐位相同），15 项正式任务已自动开始，预计约 10 小时。11-01 那次跑到 23:00 停下（完成 9 项），但下载的权重包被截断，只有 2 项权重完整，Pan-Yan-0 决定从头重跑。11-02 ModelScope 服务端返回 500，Claude 改为 v5.5：下载失败时改用 hf-mirror 并逐文件核对哈希，下载超时放宽到 2 小时，会话入口加 `all`；代码身份 `f2f1baa3…`，训练路径不变，交接包 `Qwen8B_v5.5_claude.zip`。v5.5 在新目录里验收通过，15 项跑完；重跑时 ModelScope 已恢复。**实验 A 结果**（`metrics/qwen8b_a1/`，Claude 用 `code/claude_recheck_qwen8b.py` 从概率重建，registry 登记为 A1-QWEN8B / A1-CMP-8B-4B）：OOF sym WF1 84.87%，JOINT-QWEN 4B 84.37%；同种子差 −0.41 / +0.74 / +1.18，平均 +0.50 [−0.38, +1.41]，按事先规则不算提升，是系统比较。v5.2–v5.5 待 GPT 事后复核。以后大文件下载：服务器上切成 400 MB 的块，每块算 SHA-256，本地逐块核对、拼回后再整体核对，全部对上才释放实例。

后续规划交接（更新）：实验A已完成并归档metrics/qwen8b_a1；GPT已读取README与claude_recheck.json，汇总数一致：8B/4B sym WF1为84.87%/84.37%，配对差+0.50个百分点、家族区间[-0.38,+1.41]，不满足既定提升规则。此为读取核对，尚非GPT逐条复算或v5.2–v5.5完整代码审计。B接受先Luna单来源、DeepSeek暂缓；主线+1为45项，合并W2/W4则75项。GPT交接GPT_to_Claude_B_Generation_Review_v1.md要求补齐上下文家族签名、逐折逐类配额、逐尝试原始输出和日志，并重写单人审核对应的停止规则。待Claude交冻结试点批次包后执行，尚未生成数据。mixture v2、C v3及后续代码仍待专项复核；旧双来源规划不再作为当前任务数依据。

补充交接状态：Claude 的《数据扩充的类别构成_补充设计.md》已由 GPT 审阅，独立回复文件为 `GPT_to_Claude_SemEval_Mixture_Review_v1.md`（由 Pan-Yan-0 转交，尚未归档进仓库）。这是主线“扩充规模”之外的待讨论补充，330项训练尚未获本轮执行授权。GPT建议保留生成数据G与原数据重复采样Dup的匹配对照；需先修正E/D整对整数配额、R的比例口径和BAL近似均衡，区分直接生成收益与四组交互，避免由交互未通过推断“收益主要来自比例”。先验校正使用原fit与实际完整训练分布且仅作诊断；O专用生成批次不能与类别构成混为单一变量。10-28 Claude 已回复 v2（交接包 `mixture_v2_claude.zip`，由 Pan-Yan-0 转交，尚未归档）：R1–R6 全部采纳；R 改用本折拟合部分的比例（五折 E、D 行数都是偶数，所以 R 的新增恰好等于拟合计数，Dup_R 就是 step+1）；配额用 `mixquota-v1` 整数化，各组总行数完全相等；只有 O2、O3 超出主线 +4 的池子，补 O 用同协议的混合批次、排在主池之后；完整 22 组约 52 GPU 小时或缩减 16 组约 38，由 Pan-Yan-0 在看结果前选。等待 GPT 复核，不提前改写冻结方案。正式进度、8B验收与排期继续从上述当前状态文件读取。

本轮接手补充（优先于下方历史记录）：当前规划的SemEval初步实验阶段全部完成，英语联合队列已验收125项mDeBERTa、15项SFT、15份教师、15项KD和31套OOF。最新结果与全部阶段总览见ML_note/实验/SemEval_DiCo-NLI/README.md；唯一当前待办入口为实验状态与算力计划.md，文件职责见文件导航与维护规则.md。根README和ML_note/实验结果汇总.md的实际表格已同步更新。后续以语义错误诊断、论文研究、扩充候选盲审为主；未授权自动追加GPU或API。94 calculation是已完成运行前预算，当前余额未知。下方旧进度仅为历史背景，不照其待运行描述启动实验。

- **第一轮 SST-2 指令学习、第二轮 IMDB**：都已完成，结果见[实验结果汇总](ML_note/实验结果汇总.md)和两个实验的 README。
- **第三轮 SemEval-2027 Task 2**：10-20 开工。见第 8 节和 `ML_note/实验/SemEval_DiCo-NLI/README.md`。
  - 已完成：基线；mBERT 和 mDeBERTa 编码器（mDeBERTa 加对称化最好，英语 WF1 .805）；浅层探针；错例分析；第二组英语折外诊断；单词替换样本的盲审。
  - 错例分析的结论：FORWARD 和 BACKWARD 直接颠倒很少，错误集中在 NEGATIVE_OTHER 的边界和单词替换；短语平均 3 个词，否定和量词几乎没有。
  - 对称化后的 SoftCons 是构造出来的：只看一侧的探针加对称化就有 .96。每个方法报"原方向"和"对称化后"两行，比较一致性时看原方向。
  - 和 GPT 交换了四轮意见，主要分歧已收敛；GPT 已从保存的输出独立复算第一组和第二组，结果一致。
  - 第二组（`notebooks/dico_diagnostics`，英语、mDeBERTa、按家族 5 折折外预测）已跑完，结果在 `metrics/diagnostics/`：
    - 按事先定好的折外 WF1 规则，英语工作配方选定为最多 12 轮：ΔWF1 +.028，按家族重抽样的近似区间 [.018, .039]；只针对英语和这条规则（SoftCons 反而降了）。dev 只报告，WF1 .827。
    - "只给被替换的两个词"与原短语金标的一致率降 .12。其中 128 行（28%）抽出的词至少有一个在虚词表里，净下降约 64% 落在这些行（事后拆分，不是因果分解）。
    - e6 的 dev 均值和第一组接近（.8045 对 .8051），但逐种子预测只有 88–92% 相同。GPT 已从保存的输出复核第二组，结果一致。
  - 盲审（45 个家族，Claude 和 GPT 各自独立标注，A 表词对、B 表短语，结果在 `metrics/diagnostics/blind_review/`）已经揭示：
    - 两人都给确定标签时基本一致（A 表 26 / 28，B 表 27 / 29）；GPT 更常填"无法确定"。
    - 两人对词对关系有共识的 26 个家族里，16 个和原短语的金标不同；但其中 8 个两人对完整短语也判得一样（是审阅判断和金标的分歧），7 个短语上没有共识，只有 1 个是换成完整短语后判断变了。所以说不清第二组的下降里各种原因各占多少。Claude 原先说"很大一部分是正确答案变了"，GPT 复核指出不成立，已改。
    - 两人意见一致、却和金标不同的有 11 个家族：近义或相似的说法、同一个人的不同称呼标成 E；有些上下位关系标成 O。不改金标。
    - 计数：英语训练集里筛出的 26 对没有反例的候选规则（见 8.1），是看过标签后归纳的。
    - 单词替换这条线的诊断到此为止，不再追加。
  - 10-20 起 SemEval 的工作在单独的对话里做；原来的对话只做学习（每晚的笔记任务绑定在那里）。10-22 Pan-Yan-0 再次确认：SemEval 的代码、实验结果和 README 只在 SemEval 那个对话里处理；学习对话收到 SemEval 的结果包时，不处理、不提交，提醒他发到 SemEval 对话。两边开工前都先 `git fetch`，看一眼对方新推送的提交。
  - 下一步严格按导师的清单：先大模型 instruct / Qwen（零样本、少样本、CoT，再 LoRA SFT），然后 R-Drop、SCL、MoCo，再 DeepSeek 扩充。清单之外的想法排在后面。
  - 第三组（`notebooks/dico_llm_instruct`，Qwen3-4B-Instruct-2507 的指令提示，不训练）已跑完，结果在 `metrics/llm_instruct/`，Claude 从缓存复算过：
    - dev 对称化 WF1 最高是 cot（英 .618、西 .604、巴 .506、跨语言 .535），zero 英 .584、巴 .361、跨语言 .313，都远低于 mDeBERTa（英 .805）。
    - 原方向常把 BACKWARD 判成 FORWARD（zero 英语 97 / 190）；巴斯克语和跨语言掉得多。
    - 英语训练集 select 子集上，按事先定的规则选出 few，但它和 zero 分不出高低（+.014，[−.009, +.034]）；加计数规则的 zero_rules 降 .132（[−.162, −.102]），把大量预测推向 O。
    - cot 81–96% 写满 320 token 后被强制作答；生成每批 8 条，cot 跑了约 7 小时（运行前没跑 SMOKE）。
    - `model_revision` 没记下来（transformers 5.19 的 config 里没有 `_commit_hash`），下个 notebook 改用 `snapshot_download` 取 commit 号。
  - 第四组（`notebooks/dico_qwen_sft`，Qwen3-4B-Instruct-2507 的 LoRA SFT，zero 模板，只算答案字母的损失）第一版的 SMOKE 已在 Kaggle T4 上跑通（零样本概率和第三组一致；每步 5.7 秒、峰值 8.7 GiB，英语每个种子约 45 分钟；模型 commit `cdbee75f`）。GPT 审过两轮，现在是第三版（`dico_qwen_sft-3`：检查点逐个验证（RNG、fp16 的 scaler 也必须在）、载入权重后逐个核对张量、torchao 只清 site-packages、记录代码哈希、SMOKE 四类都有并实测中断恢复和重新加载；正式运行取不到代码哈希就停下），第三版的 SMOKE 已在 Kaggle T4 上跑通（代码哈希和 `ee1d9ab` 一致；六项边界检查全部通过：四类覆盖、中断后从第 1 轮恢复、写坏的检查点被跳过、检查点里有 RNG 和 scaler、每轮权重逐位相同、第二个种子加载前显存已释放；每步 6.2–6.5 秒，峰值 8.75 GiB）。英语正式运行已完成（3 个种子，结果在 `metrics/qwen_sft/`，Claude 从缓存复算一致）：fold 0 对称化 WF1 .860，mDeBERTa 同数据 .793，同种子差平均 +.067 [+.033, +.104]，两条蒸馏门槛都通过；但概率平均不比 Qwen 单独好（−.013 [−.039, +.011]）；dev .835 和 mDeBERTa .827 接近（训练数据量不同）；三个种子都选在第 3 轮，内层验证还在涨。要不要做蒸馏、R-Drop 等加在哪个模型上，等 Pan-Yan-0 决定：
    - 数据划分和第二组 fold 0 逐行相同（训练 2,190、内层验证 244、留出 608），英语 fold 0 上和 mDeBERTa 12 轮配方的折外预测按种子配对比较。
    - 事先定好的蒸馏条件：英语 fold 0 上，同种子的 sym WF1 差（Qwen SFT − mDeBERTa）三个种子平均大于 0，或同种子两模型概率 0.5 / 0.5 平均后比 mDeBERTa 高，而且按家族配对重抽样的区间下限大于 0；都不满足就不做蒸馏。这只是探索性门槛，通过了也要和同数据、同预算的 mDeBERTa 对照。
    - 训练部分 E / F / B / O 是 574 / 631 / 631 / 354（O 只占 16%，不是四成）；dev 有 15 行（8 个 pair）的文本在训练集里出现过，选择看 fold 0。
    - 本地用假的 torch / transformers / peft 测过续跑：训练中途、评估后存检查点前、读概率时被打断，都能接着跑，选中的权重能复现记录的验证分数。
  - Pan-Yan-0 10-21 的决定：蒸馏要做；先试 SFT 训练更多轮（`MAX_EPOCHS = 5`，代码不改）值不值得，有结果再补种子验证；R-Drop、SCL、MoCo 三种都做，先在 mDeBERTa 上，再加到 Qwen 上。设计和判断规则见实验 README 的下一步计划（事先定好，GPT 审过一轮后修订）：mDeBERTa 上的方法先在 fold 0、3 个种子上比，冻结设置后再扩到 5 折；方向版 R-Drop 第一版只对 E / F / B；MoCo 先小试 m = 0.99 和 0.999；Qwen 上的 R-Drop 要先加 LoRA dropout，所以多一组只加 dropout 的对照。老师导出 notebook（`dico_qwen_teacher-3`，GPT 复核通过）绑定来源 fingerprint 的哈希和每份 LoRA 权重的哈希，权重在 Kaggle 那次 3 轮运行的输出里，那个版本不要删。
  - 西语、巴语 SFT（Kaggle T4，3 轮，fingerprint 和英语相同，结果在 `metrics/qwen_sft/track2|3/`，Claude 复算一致）：dev 对称化 WF1 .833 / .745（mDeBERTa 第一组 .758 / .723，训练数据量不同），fold 0 .837 / .729；巴语 seed 44 选在第 1 轮、dev .711。两种语言还没有 mDeBERTa fold 0 折外结果，做不了同数据比较。
  - 跨语言 SFT（阿里云 A10，bf16，`dico_qwen_sft-4`，代码是 `d9490c3`；结果在 `metrics/qwen_sft_a10/`，Claude 复算一致）：dev 对称化 WF1 .762（mDeBERTa 第一组 .745），fold 0 .766；选中第 3 / 2 / 1 轮。A10 每步 2.4 秒，每个种子约 107 分钟，共约 5.5 小时。LoRA 权重包在 Pan-Yan-0 那里。
  - SFT 最多 5 轮（腾讯 Cloud Studio T4，3 个种子，结果在 `metrics/qwen_sft_e5/`，Claude 复算一致）：选中第 3 / 5 / 4 轮；fold 0 sym WF1 同种子差 −.011 / −.003 / +.006，平均 −.003 [−.017, +.011]，按事先的规则不通过，工作配方仍是 3 轮。零样本概率和 Kaggle 那次逐条相同（推理一致，训练等价没验证）。
  - A10 上的英语 3 轮基线（R3，`dico_qwen_sft-4`，结果在 `metrics/qwen_sft_a10_t1/`，Claude 复算一致）：fold 0 sym WF1 .857，dev .836；和 Kaggle T4 那次同种子差平均 −.003 [−.012, +.006]（执行环境迁移复现，不说明等价）。每步 2.35 秒，每个种子约 18 分钟。之后 Qwen 技巧都和它配对。阿里云连 GitHub 很慢，官方仓库用 Pan-Yan-0 上传的 zip；模型从 ModelScope 下载只要半分钟。
  - 老师导出已完成（Kaggle T4，结果在 `metrics/qwen_teacher/`，`code/qwen_teacher_check.py` 复核一致）：2,190 行上老师准确率 .969–.979，T = 2 后最大概率中位数 .90–.94。
  - 第五组（`notebooks/dico_mdeberta_methods`，现在是 `dico_mdeberta_methods-3`，A10 上用 `dico_a10_mdeberta.sh`）：GPT 审过第一版，指出 5 个问题（有限的 loss 不保证梯度和 logits 有限、fp16 跳过更新时学习率和 MoCo 仍前进、对比损失的乘法在 autocast 里是低精度、SMOKE 不能重复运行、老师身份没核对），第二版都改了，加了 attempts.jsonl、CUDA 检查、出错时也打包；GPT 复核第二版：A10 / bf16 路径可以跑 SMOKE，只剩 fp16 下"梯度元素有限但总范数溢出"会被误判成跳过（只影响 T4），第三版已改（先单独检查元素是否有限，范数溢出时不调用 optimizer.step()）；下一步 A10 上先跑 SMOKE、把输出给 Claude 看，再跑第一阶段。设计：一个任务 = 赛道 × 方法 × 种子 × 折，各有 `task.json` 和 `done.json`；`smoke`（损失自检、中断续跑、测速和时间估计）→ `stage1`（英语 fold 0 × 种子 42–44：C0 / B2 / C1 / C2 / C3 / C4 两个 m；西语、巴语、跨语言的 C0）→ 验收（只看实现和资源）→ `stage2`（英语 5 折 × 种子 42–46 × 五种设置，m 按内层验证三种子平均自动选）。本地用假 torch 测过全流程；A10 上估计两个阶段 5.5–8 小时、约 44–62 计算时。
  - 第六组（Qwen 英语 fold 0 × 3 种子上的 CE 重跑 / 只加 dropout / R-Drop / SCL / MoCo）已跑完，GPT 修了 v2（代码在 `code/qwen_methods_v2/`），结果在 `metrics/qwen_methods_v2/`。ForwardCheck 实测两条前向路径都是 bf16 autocast、四类 log 概率差 0。CE 重跑对第四组 R3 三个种子都低约 0.4 个百分点（CMP-16），和 SCL、MoCo 的效应同量级。Claude 用 `code/claude_recheck_joint.py` 从逐折 logits / 概率重建了第五组五折、Qwen 五折、KD 和第六组的全部预测，分数和区间都和 GPT 的归档一致。
  - 其他赛道全量（导师要求，Kaggle T4 fp16，结果在 `metrics/mdeberta_multilang_oof/`，Claude 用 `code/claude_recheck_multilang.py` 复核一致）：mDeBERTa CE 西语 / 巴语 / 跨语言各五折三种子，OOF 对称化 WF1 77.97 / 73.45 / 74.95%。同配方 T4 fp16 比 A10 bf16 的 fold 0 平均高约 1 个百分点（四个赛道都是）。非英语赛道的划分有 1.5–2.3% 的留出行文本也在同折训练里（按英语源短语对分组，和官方划分方式一致，但不同源短语对翻译后会撞），去掉这些行后 WF1 低 0.29–0.35 个百分点（只是敏感性分析，不当修正量）；官方 train / dev 之间也有 1.1–1.5% 的同类重合，所以不改划分、报告写明。驱动由 GPT 编写，在 `code/multilang_kaggle/`，代码身份 `8554642853d6…` 能从归档文件复算。跨语言赛道是官方提供的混合语言数据（每行 text1、text2 取自同一源短语对不同语言的官方译文，标签和英语相同），我们没有做翻译，模型直接读两种语言。
  - CodaBench 开发阶段排行榜（Pan-Yan-0 10-27 截图）：前几名 SoftCons 都是 1.0、WF1 约 0.93–0.96，有的四个赛道分数完全相同。这和利用数据结构的结果相符，但不能证明别人用了（GPT 10-27 指出，文档里只做中性描述）：有反向那一条的 pair 全是 E / F / B，单条的全是 O，提交模板的 instance_id 看得出来。只加这条规则，已有 Qwen SFT 的 dev WF1 英 / 西 / 巴 / 跨语言升到 93.1 / 92.0 / 88.1 / 88.6%（`code/claude_structure_check.py`）。用不用要问主办方、由 Pan-Yan-0 和导师决定；我们的结果目前都只用文本。
  - 下一阶段（10-27 和 GPT 两轮讨论后冻结）：实验 A Qwen3-8B 英语五折三种子（A10，和 JOINT-QWEN 配对，系统比较）；实验 B 生成数据扩充，DeepSeek（API）和 GPT6Luna（子 agent）两个来源，按可用折签名分批防泄漏，试点 → +1 比较 → 选中来源 +2 / +4，每个规模配步数对照。设计、停止条件、清单接口和分工见 `ML_note/实验/SemEval_DiCo-NLI/更大模型与数据扩充计划.md`。Claude 写生成侧，GPT 写两个训练队列。10-28 A1 工程验收在 A10 上全部通过（`metrics/qwen8b_acceptance/`）：零样本和新旧入口逐位一致，8B 批 8 × 2 峰值 17.0 GiB，正式运行估计约 10 小时、70 计算时，等 11 月额度；运行的是 GPT 的 v4，v2 之后的改动待 Claude 补审。
  - 排期（10-22）：蒸馏学生、mDeBERTa 上的 R-Drop / SCL / MoCo、西语 / 巴语 / 跨语言同数据的 mDeBERTa 基线、之后 Qwen 上的技巧，主要放阿里云 A10（月底前约 28 个 A10 小时，月底作废）。Qwen 技巧放 A10 时，先在 A10 上重跑 3 轮基线作对照，不和 T4 的结果直接相减。
  - 算力分配：Kaggle（每周额度，能后台跑）只留给超过一小时的 Qwen 任务；Colab、腾讯 Cloud Studio 要开着页面，白天跑 mDeBERTa 类短任务和 SMOKE；阿里云 A10 抢到才算。同一组比较不混用不同的卡（A10 / 4090 用 bf16，T4 用 fp16）。
- **原理学习**：第一章（情感分类、反向传播）已通关，笔记在 `ML_note/原理笔记/`；第二章跟着 SemEval 实验学（`ML_note/原理笔记/NLI/`），已讲句对输入、SoftCons / HardCons、对称化、R-Drop 与 KL、方向版 R-Drop。学到哪、下一步学什么，由学习对话记在 Pan-Yan-0 的私有仓库里，不记在这里。
- **模型压缩方向（Pan-Yan-0 自己的论文线，方向待导师确认）**：NLP 推理优化，剪枝为入口。笔记在 `ML_note/论文阅读/模型压缩/` 和 `ML_note/原理笔记/`；动手实验在 `ML_note/实验/ShortGPT_BI删层/`（Qwen3-8B 按 BI 删层、不恢复，Kaggle 2×T4；`bi_from_pair` 和 `drop_layers` 由 Pan-Yan-0 编写）。Llama-2 的 HF 权限申请被拒，默认用 Qwen3-8B。这条线的进度、当前任务和到投稿的路径不记在这里，由学习对话记在 Pan-Yan-0 的私有仓库里，以那里为准。
- **待办**：GitHub 仓库首页默认显示 `master`，`note` 的最新内容在首页看不到。改法等 Pan-Yan-0 决定：
  - 把默认分支改成 `note`；
  - 在 `master` 的 README 加入口；
  - 或者合并。这会动别的分支，不自己做。

## 8. SemEval-2027 Task 2：DiCo-NLI

### 8.1 任务（来自官方 README）

- **输入**：一对短语，前提 A 和假设 B，来自 PhrasIS（图片描述和新闻标题）。(A, B) 和 (B, A) 两个方向都要预测。
- **标签**：EQUIVALENCE、FORWARD_ENTAILMENT、BACKWARD_ENTAILMENT、NEGATIVE_OTHER。
  - 例：前提 "Everyone is hungry"、假设 "Someone is hungry" 是 FORWARD，对调后是 BACKWARD。
  - EQUIVALENCE 对调后还是它自己。
- **赛道**：
  - 英语、西班牙语、巴斯克语三个单语赛道，训练集各 3,042 条，dev 各 660 条。单语训练集里不同的可对调短语对约 1,280 组，另有 478 条 NEGATIVE_OTHER。
  - 混合语言赛道训练集 18,252 条（= 3,042 × 6），dev 3,960 条。
- **指标**：Weighted F1、SoftCons（两个方向是否自洽，不看对错）、HardCons（两个方向都答对）。官方没有指定主指标，三个都要报。
  - 评分：`python3 -m evaluation_functions --gold <reference> --predictions <file> --output-dir <dir>`。
  - 提交文件只有 `instance_id,label` 两列。
- **基线**：主办方给了 random 和 majority 基线。试点（英语，只含三个正向标签）里 DeBERTa-v3-base 是 WF1 0.75、SoftCons 0.84、HardCons 0.79。设置和正式赛不完全一样，只作参考。
- **规则**：允许外部数据和大模型，但必须在系统论文里写清模型、提示词和外部数据。PhrasIS 原数据能不能拿来训练，以官方最终规则为准，确认之前不用。
- **时间**：评测 2027-01-10 至 01-31，2 月交系统论文。
- **要记住的坑**：
  - 只有一条记录的 pair 都是 NEGATIVE_OTHER（训练集、dev 和 trial 的测试格式文件都是这样），模型只用文本，不用这个结构。
  - 评分脚本里，可对调的短语对只要有一个方向预测成 NEGATIVE_OTHER，SoftCons 就算这一对不一致，所以对称化不保证 SoftCons 满分。
  - PhrasIS 的作者稿写明，短语对是脱离原句语境重新标注的（GPT 查到，Claude 没取到原文）。所以金标和字面逻辑不一致时，不要归因于"丢了语境"。
  - 官方评分脚本用 `csv.Sniffer` 猜金标文件的格式，子集里短语以撇号开头（如 `'s integration`）时会把撇号当成引号、整个文件解析错位，报 "Label must not be empty"。写金标子集时只写 `instance_id, pair_id, reverse_pair_id, label` 四列。
  - 内部验证要先把文本相同的 pair_id 合成一个家族再分组；训练集里同一有序输入出现过两个标签（South Africans → South Africa）。
  - 计数的候选规则（来自训练标签，不是官方口径）：数人或物时，数大的一侧推出数小的一侧（"6 dead" 推出 "One dead"）；数值是读数（震级、汇率、百分比、日期）时是 NEGATIVE_OTHER。`code/label_conventions.py` 用有限数字词表、"两侧只差一个数"筛出的 26 对里没有反例，但这是看过标签后归纳的；其中 2 对原文写了 At least；exactly、at most、否定数量、多个数字都没覆盖。写进提示词时称"从训练数据归纳的启发式"，报告时注明来源；评估它时要排除这 26 个来源家族。
- 官方说明：[SemEval-2027 Task 2 DiCo-NLI README](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/blob/main/README.md)

### 8.2 导师的要求

1. 先把现有模型跑出结果：mBERT、多语言 DeBERTa、大模型 instruct、Qwen 等。BERT 类做 NLI 就是 `[CLS] A [SEP] B [SEP]`。
2. 再试 SCL、R-Drop、MoCo 和 CoT。
3. 数据少，用 DeepSeek 扩充，"越多越好，增加的过程中看性能会不会升"。
4. 不先看综述，动手最重要。
5. 之后导师会讲反事实推论和因果推断。
6. 仓库记录结果、问题和下一步计划。

### 8.3 开工后的计划（初稿）

1. **先定口径**：只用官方 dev 和官方评分脚本；三个指标按赛道分开报；每组跑 3 个种子。
2. **锚点**：复现 random / majority 基线。
3. **编码器**：mBERT、mDeBERTa-v3-base。
4. **大模型**：Qwen 先跑零样本、少样本和 CoT，沿用 SST-2 的脚本和解析器经验；再做 LoRA SFT，沿用 IMDB 的代码。
5. **训练技巧**：在最好的编码器上一次只加一个：R-Drop、SCL、MoCo。
6. **DeepSeek 扩充**：按 +1、+2、+4 倍画数据量曲线。
   - 扩充要造新的短语对，把已有的再翻译或对调一遍不算新数据。
   - dev 始终只用官方原始数据。
   - 提示词和生成数据都留档。
7. **反事实**：等导师讲。可以先准备一个只给假设句的基线。

这个任务特有的点子，都要用实验验证：
- 预测时对称化：把 p(B, A) 的 FORWARD / BACKWARD 对调后，和 p(A, B) 取平均。
- 方向版 R-Drop：让 p(A, B) 和对调后的 p(B, A) 接近。
- 把大模型拆成两个是非题："A 能推出 B 吗？""B 能推出 A 吗？"

实验放在 `ML_note/实验/SemEval_DiCo-NLI/`，每个实验登记编号、commit、配置、种子和三个分数。

## 9. 定时任务

- **学习笔记没有定时任务**：Pan-Yan-0 说"结束今天的学习"时，学习对话当场整理笔记并提交到 `note` 分支；没有新内容就不提交。原来每晚 21:51 的任务绑定在旧对话上，看不到现在的学习对话。
- **每周日 10:00「推送 D2L 学习记录」**：把伴学簿导出到 `d2l` 分支。

## 10. 关键技术事实（避免重查）

- **Claude 的云端环境**：装不了 PyTorch，也连不上 Hugging Face，模型相关的代码在这里跑不了。notebook 只能用假模型测数据、评分和续跑这些部分，模型部分靠另一个 AI 审代码和 GPU 上的 SMOKE 验证。
- **Qwen 选型**：Qwen3.5 的 0.8B–9B 小模型带视觉编码器、用 Gated DeltaNet，SST-2 是用 Ollama 跑的；在 T4 上用 transformers 读 token 概率时，用纯文本的 `Qwen/Qwen3-4B-Instruct-2507`（只有非思考模式），或 IMDB 用过的 `Qwen/Qwen3-4B-Base`。
- **判断运行平台**：Kaggle 的镜像里也有 `google.colab` 和 `COLAB_RELEASE_TAG`，在 Kaggle 上 `drive.mount` 会报 NotImplementedError。先用 `/kaggle/working` 认 Kaggle，Colab 用 `/var/colab/hostname` 判断。
- **Kaggle 提交**：用 `sample.copy()` 作模板，并 assert id 对齐；`quoting=3` 会让 test.id 带上字面引号。
- **gensim 4**：`glove2word2vec` 已删除，要手动给文件加首行"词数 维度"；用 `model.wv.*`；`size` 改名为 `vector_size`。
- **导师的 IMDB 脚本包**：
  - 先跑 `imdb_process.py`（GloVe → pickle）。CNN、LSTM、GRU、Attention、Transformer 系列都依赖这个 pickle；BERT 系列独立。
  - 设备硬编码为 `cuda:0`。
  - 显存不够时依次试：batch_size 减半 → max_len 降到 256 → 换 DistilBERT。
  - `imdb_bert_scratch.py` 其实是"手写分类头 + 预训练编码器"，不是从零训练。

