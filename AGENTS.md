# AGENTS.md · Claude 在这个仓库里的工作备忘

Claude 在 Pan-Yan-0 的 study_note 仓库里工作时用的备忘：要遵守的规则、现在做到哪、东西放在哪。开始工作前先读一遍；内容过时就直接改这里。2026-10-16 以前的旧档案在 [agents/AGENTS旧版.md](agents/AGENTS旧版.md)，只作历史参考。

## 1. 最重要的几条

1. **称呼和语言**：称呼 Pan-Yan-0（GitHub 用户名），不用中文名；交流和文档都用中文。
2. **仓库是公开的，导师也会看**：不写学习画像、经验值、个人邮箱、令牌和本机路径。教学方法、画像和经验存档只放在 claude.ai 上 Pan-Yan-0 的私有 skill `cognitive-tutoring` 里。
3. **工作分支是 `note`**：`master`、`d2l`、`cuda_study`、`math_modeling` 不要动（`d2l` 由每周的定时任务单独维护）。
4. **任何决策都要有实验支持**（导师原话："扩充数据，参数选取，我也说不清好不好，实验结果说明好不好"）：
   - 改动都和基线对照，一次只改一处。
   - 数据少时跑多个随机种子，看同一种子下的配对差值；差值的波动盖过差值本身时不下结论。
   - 建议改什么时，同时给出验证它的对照实验。
5. **结论之前先抽查真实输出。** SST-2 里 Mistral 的 39.68% 其实是评测解析器太严造成的，改正后是 90.60%。分数异常低或高时，先看几十条原始回答，再下结论。
6. **重要问题直接重写**：发现方案或文档有结构性问题，直接重写，不做修修补补。

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
├── 原理笔记/               BERT/  DeBERTa/  LLM/  LoRA/  NLI/  反向传播/
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
  - 含 GPT 写的代码、Notebook 或文字时，再加 `Co-authored-by: GPT6.1sol <noreply@openai.com>`。换了别的 GPT 模型（如 GPT6astra），就换成实际参与的那个名字；拿不准时问 Pan-Yan-0。
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
  - 其他赛道全量（导师要求，Kaggle T4 fp16，结果在 `metrics/mdeberta_multilang_oof/`，Claude 用 `code/claude_recheck_multilang.py` 复核一致）：mDeBERTa CE 西语 / 巴语 / 跨语言各五折三种子，OOF 对称化 WF1 77.97 / 73.45 / 74.95%。同配方 T4 fp16 比 A10 bf16 的 fold 0 平均高约 1 个百分点（四个赛道都是）。非英语赛道的划分有 1.5–2.3% 的留出行文本也在同折训练里（按英语源短语对分组，和官方划分方式一致，但不同源短语对翻译后会撞），约偏高 0.3–0.4 个百分点；官方 train / dev 之间也有 1.1–1.5% 的同类重合，所以不改划分、报告写明。驱动由 GPT 编写，在 `code/multilang_kaggle/`，代码身份 `8554642853d6…` 能从归档文件复算。跨语言赛道是官方提供的混合语言数据（每行 text1、text2 取自同一源短语对不同语言的官方译文，标签和英语相同），我们没有做翻译，模型直接读两种语言。
  - 排期（10-22）：蒸馏学生、mDeBERTa 上的 R-Drop / SCL / MoCo、西语 / 巴语 / 跨语言同数据的 mDeBERTa 基线、之后 Qwen 上的技巧，主要放阿里云 A10（月底前约 28 个 A10 小时，月底作废）。Qwen 技巧放 A10 时，先在 A10 上重跑 3 轮基线作对照，不和 T4 的结果直接相减。
  - 算力分配：Kaggle（每周额度，能后台跑）只留给超过一小时的 Qwen 任务；Colab、腾讯 Cloud Studio 要开着页面，白天跑 mDeBERTa 类短任务和 SMOKE；阿里云 A10 抢到才算。同一组比较不混用不同的卡（A10 / 4090 用 bf16，T4 用 fp16）。
- **原理学习**：第一章（情感分类、反向传播）已通关，笔记在 `ML_note/原理笔记/`；第二章跟着 SemEval 实验学（`ML_note/原理笔记/NLI/`），已讲句对输入、SoftCons / HardCons、对称化、R-Drop 与 KL，下一课是方向版 R-Drop。
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

- **每晚 21:51（北京时间）「每晚提交学习笔记」**：在 Pan-Yan-0 的学习对话里整理当天的问答，提交到 `note` 分支，并按私有 skill 结算学习进度。没有新内容就不提交。
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
