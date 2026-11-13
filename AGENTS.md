# AGENTS.md · 仓库公共规则与工作线入口

Pan-Yan-0 的 `study_note` 仓库公共备忘：要遵守的规则、仓库结构、实验约定与各工作线唯一状态入口。开始工作前先读一遍；历史进度归档在 [agents/AGENTS历史进度_2026-11.md](agents/AGENTS历史进度_2026-11.md) 与 [agents/AGENTS旧版.md](agents/AGENTS旧版.md)，只作历史参考。

## 1. 最重要的几条

1. **称呼和语言**：称呼 Pan-Yan-0（GitHub 用户名），不用中文名；交流和文档都用中文，但**专业名词保留英文原词**（如 fold、seed、OOF、batch、checkpoint、warmup、family、subagent、SMOKE），不硬译成中文（Pan-Yan-0 11-07 定的）。
2. **仓库是公开的，导师也会看**：不写个人画像、个人邮箱、令牌和本机容器路径。教学方法与私有学习状态以 Google Drive `Pan-Yan-0_工作区` 私有目录为主工作区，并同步备份至私有仓库 `Pan-Yan-0/personal_study_image`；公开仓库只进标记为`允许公开发布`并通过检查的成果。
3. **工作分支是 `note`**：`master`、`d2l`、`cuda_study`、`math_modeling` 不要动（`d2l` 已暂停维护）。
4. **任何决策都要有实验支持**（导师原话："扩充数据，参数选取，我也说不清好不好，实验结果说明好不好"）：
   - 改动都和基线对照，一次只改一处。
   - 数据少时跑多个随机种子，看同一种子下的配对差值；差值的波动盖过差值本身时不下结论。
   - 建议改什么时，同时给出验证它的对照实验。
5. **结论之前先抽查真实输出。** SST-2 里 Mistral 的 39.68% 其实是评测解析器太严造成的，改正后是 90.60%。分数异常低或高时，先看几十条原始回答，再下结论。
6. **重要问题直接重写**：发现方案或文档有结构性问题，直接重写，不做修修补补。

### 多端文件交接与对话分工

- 当前研究主线为 **SemEval-2027 Task 2 DiCo-NLI**；IMDB、SST-2 已完成，仅按需引用历史代码和经验。
- **Google Drive 主工作区 + GitHub 同步备份与 Actions 检查**：Google Drive（`Pan-Yan-0_工作区`）作为所有端共同读写的主工作区（当前权威版本）；GitHub 保存同步副本、运行 Actions 自动检查并保留版本历史。Drive 内区分`工作区私有`（默认）和`允许公开发布`两个状态。
- 双方各自整理可独立阅读的交接文件：只有讨论、设计或审阅文字时交付一个 Markdown；涉及代码、测试、配置或多份必要材料时打包一个 ZIP，附入口 README。每份交接记录 6 项必要信息：任务与当前负责人、允许修改范围、来源版本、可见范围、当前状态（待审/退回/通过）与正式结果链接。
- **单点写入与对话分工**（详见 [agents/分工与纠错机制_v2.md](agents/分工与纠错机制_v2.md)）：一个路径在同一时间只有一个写入方；多端协作的工作线，每个具体任务明确一位实现负责人（单点写入）和允许修改的路径范围，另一位做独立审阅（只查改变正确性、结论、可运行性和隐私边界的问题），修改经交接后由该任务负责人提交。
  - **SemEval 实验线**：实验设计、代码、运行结果、实验 README，以及把论文研究交回的想法落成对照实验。唯一待办与进度入口为 [实验状态与算力计划](ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md)；方案达成一致后再更新 [冻结计划](ML_note/实验/SemEval_DiCo-NLI/更大模型与数据扩充计划.md) 的变更记录。交接本身不启动新增 GPU 训练或付费生成。
  - **SemEval 论文研究线**：按错误分类选论文、预读和精读、阅读表、低成本离线检查；只写 `ML_note/实验/SemEval_DiCo-NLI/论文研究/`，要跑的实验写进其中的《交回SemEval对话的实验》，由 SemEval 对话决定。
  - **学习与模型压缩线**：讲解概念、出题批改，收工时整理进 `ML_note/原理笔记/`；模型压缩方向的论文精读笔记写进 `ML_note/论文阅读/`（和 SemEval 的论文研究分开），动手实验在 `ML_note/实验/ShortGPT_BI删层/`。
  - **基建与中转线**：维护根 `AGENTS.md`、`agents/`、`.github/workflows/`、Google Drive 主工作区同步与仓库自动检查脚本（[`agents/check_repo_hygiene.py`](agents/check_repo_hygiene.py)）。

## 2. 背景

- Pan-Yan-0：研究生，方向是 NLP 与大模型，职业目标是 AI Infra / 大模型推理优化。实验由他在 Kaggle、Colab、腾讯 Cloud Studio、阿里云 PAI-DSW 上运行，再把结果带回来；决定由他做。
- 导师：布置实验任务，会看这个仓库，要求 README 记录"结果、问题和下一步计划"。
- 仓库里有一部分代码是 GPT / Codex 写的（例如 SST-2 评测脚本、SemEval A1 训练包），提交时要按第 5 节加它的署名。

## 3. 仓库结构

```text
AGENTS.md / CLAUDE.md      本备忘（CLAUDE.md 只是引用本文件）
README.md                  首页：目录树、实验和笔记的入口
agents/                    多端分工规范、纠错台账、仓库卫生检查脚本及旧版档案
skills/                    教学 skill 的旧版，现行版本在私有仓库里
ML_note/
├── 实验结果汇总.md         全部实验结果速览
├── 实验/IMDB/  实验/SST2/  每个实验一个文件夹：README、code/、notebooks/、metrics/
│   实验/SemEval_DiCo-NLI/  当前研究主线（含 code/、metrics/、论文研究/）
│   实验/ShortGPT_BI删层/   模型压缩方向的动手实验（学习线负责）
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
- **禁用内容**：不写实验过程的流水账、日期和个人私有记录。代码只保留必要的注释。
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
- **推送前**：先 `git fetch`；运行 `python3 agents/check_repo_hygiene.py` 检查提交范围的路径归属、敏感信息与本机容器路径。
- **只用普通推送**：不强推，不为了补署名改写已推送的历史。推送失败就停下来，说明是哪一步、报了什么错。
- **结果文件**：`.gitignore` 忽略了 `*.csv`、`*.json`，要提交的指标文件用 `git add -f`。

## 6. 实验的约定

- **Notebook**：
  - 能断点续跑，带 SMOKE 小规模开关。
  - 默认把实验跑完，不设中途停掉实验的时间预算。
  - 写清它和导师原脚本有哪些区别，并附时间估计。
  - 在 Kaggle、Colab 和普通 Linux（如 Cloud Studio）上都能跑，换平台时能导入上一次的结果包接着跑。
  - 打印的提示写清已经完成了什么、还需要他手动做什么。
  - 结果包（Pan-Yan-0 定的原则）：之后需要、或者可能需要的都放进 zip，用不上的不放。比如 Qwen SFT 选中那一轮的 LoRA 权重（之后蒸馏、换平台都可能要用）单独打一个权重包，不混进发给 AI 分析的结果包；训练中途的检查点、底座模型这类可以重新得到的不放。
- **控制变量（Pan-Yan-0 10-22 定的）**：每次运行都要能和别的运行比，说清改了哪个变量。
  - SemEval 的每次运行都登记进 `ML_note/实验/SemEval_DiCo-NLI/实验登记表.md`，表由 `code/registry.py` 从指纹、运行记录和逐条预测自动生成，不手改；新运行在 `RUNS` 里加一项，新比较在 `COMPARISONS` 里加一项。
  - 脚本自动列出两边记录了的变量差别（没记录的列为未知，不当作相同），比较分两列归类：设计（重复运行 / 单一计划改动 / 系统比较）和执行环境（一致 / 同卡同精度、软件或平台不同 / 卡或精度不同 / 信息不全）。平台名称、torch 和库版本只标注、不要求补跑（Pan-Yan-0 10-22 定的），但结论里要写明环境差别；把差别归因到某个变量，要求是单一计划改动；卡或精度不同时，在同一环境里补跑对照再归因。系统比较可以写进结论，只说在给定设置下哪个更好（GPT 10-22 审阅后的写法）。
  - 新写的 notebook：指纹里写全所有训练变量（包括损失、dropout、正则项的系数、训练数据），运行记录里写平台名称（Kaggle / Colab / 腾讯云 / PAI-DSW）、卡、精度和 torch 版本。
- **实验指纹**：provider、模型、提示词、是否开思考、服务地址、max_tokens、temperature、数据哈希。续跑时这些必须和原来一致。
- **max_tokens 包含思考内容**：开思考时容易写满被截断，没有答案的按错误计，并单独统计截断数。
- **评测解析**：从回答里取标签时，词和数字要一致才算；改了解析规则，要用新规则重算所有已有结果。
- **算力**：
  - Kaggle：GPU 有每周额度，可后台 Commit 运行。
  - Colab：免费 T4，要一直开着页面，单次最多 12 小时。
  - 腾讯 Cloud Studio：T4 可用。`ollama pull` 会被拦，改用 curl 从 `registry.ollama.ai/v2/library/<名字>/manifests|blobs` 下载，核对 SHA-256，manifest 按原字节保存。
  - 阿里云 PAI-DSW：A10 每小时耗 6.991 计算时。**停止实例不能止住扣计算时，要释放实例才行**（Pan-Yan-0 10-22 实测）；闲置 3 小时的自动关机不一定触发。释放前先把结果包和权重包下载下来并逐块核对 SHA-256。每次开实例按同一个流程，由一个 sh 脚本完成：装环境 → SMOKE（Pan-Yan-0 手动跑）→ 全量。连 Hugging Face 要设 `HF_ENDPOINT=https://hf-mirror.com`。pip 要用内网镜像 `http://mirrors.cloud.aliyuncs.com/pypi/simple/`（加 `--trusted-host`）。
  - **本机网络（Pan-Yan-0 10-22 说的）**：和阿里云、腾讯云之间上传下载文件，要把本机网络切到国内线路；用海外服务与 Kaggle 时又要切回去。所以每次开国内实例只安排两次传输：开始时上传一个打包好的 zip，结束时下载结果包和权重包，不中途零散传文件。
  - 本机显卡只有 4 GB；T4 估时按约 10 TFLOPS 算。

## 7. 各工作线唯一状态入口与分工指针

不在本文件重复抄写具体实验进度，直接读取对应工作线的唯一状态入口：

| 工作线 | 唯一状态与进度入口 | 分工与历史参考 |
|---|---|---|
| **SemEval 实验主线** | [实验状态与算力计划](ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md) · [SemEval 总表](ML_note/实验/SemEval_DiCo-NLI/README.md) · [冻结计划](ML_note/实验/SemEval_DiCo-NLI/更大模型与数据扩充计划.md) | Antigravity「SemEval 对话」与 Codex 按任务定单点实现负责人、另一方审阅（详见 [多端分工与纠错机制](agents/分工与纠错机制_v2.md)）；历史详录见 [agents/AGENTS历史进度_2026-11.md](agents/AGENTS历史进度_2026-11.md) |
| **SemEval 论文研究** | `ML_note/实验/SemEval_DiCo-NLI/论文研究/` | 候选实验写进《交回SemEval对话的实验》由 SemEval 线决定 |
| **学习与模型压缩** | 公开成果见 [原理笔记](ML_note/原理笔记/)、[论文阅读](ML_note/论文阅读/README.md)、[ShortGPT_BI删层](ML_note/实验/ShortGPT_BI删层/README.md) | 私有状态在 Google Drive `Pan-Yan-0_工作区/10_交接/学习/` 主工作区维护，同步备份至私有仓库 `Pan-Yan-0/personal_study_image` |
| **多端分工与纠错基建** | [多端分工与纠错机制](agents/分工与纠错机制_v2.md) · [纠错台账](agents/纠错台账.md) | 仓库卫生、任务范围与版本回执检查：[`agents/check_repo_hygiene.py`](agents/check_repo_hygiene.py) |

### 7.1 DiCo-NLI 任务常驻防坑要点（来自官方说明与已验证事实）

- **只用文本**：只有一条记录的 pair 都是 `NEGATIVE_OTHER`，但模型只读文本，不使用 `pair_id` 单双条结构（Pan-Yan-0 11-12 定）。
- **评分脚本与 `SoftCons`**：可对调的短语对只要有一个方向预测成 `NEGATIVE_OTHER`，`SoftCons` 就算这一对不一致，因此对称化不保证 `SoftCons` 满分。
- **`csv.Sniffer` 撇号陷阱**：官方评分脚本用 `csv.Sniffer` 猜金标格式，子集里短语以撇号开头（如 `'s integration`）时会把撇号当成引号导致解析错位，报 `Label must not be empty`。写金标子集时只写 `instance_id, pair_id, reverse_pair_id, label` 四列。
- **家族分组隔离**：内部验证要先把文本相同的 `pair_id` 合成一个家族再分组；训练集里同一有序输入出现过两个标签（`South Africans → South Africa`）。

## 8. 关键技术事实（避免重查）

- **代码验收与测试环境**：模型训练前先通过 CPU 单元测试与独立审阅（重建包、哈希核对、注入错误回归），再在目标 GPU（Kaggle T4 / Colab T4 / 阿里云 A10）上跑通 SMOKE 回执后方可执行正式队列。
- **Qwen 选型**：Qwen3.5 的 0.8B–9B 小模型带视觉编码器、用 Gated DeltaNet；在 T4 上用 transformers 读 token 概率时，用纯文本的 `Qwen/Qwen3-4B-Instruct-2507`（只有非思考模式），或 IMDB 用过的 `Qwen/Qwen3-4B-Base` / `Qwen/Qwen3-8B`。
- **判断运行平台**：Kaggle 的镜像里也有 `google.colab` 和 `COLAB_RELEASE_TAG`，在 Kaggle 上 `drive.mount` 会报 `NotImplementedError`。先用 `/kaggle/working` 认 Kaggle，Colab 用 `/var/colab/hostname` 判断。
- **Kaggle 提交**：用 `sample.copy()` 作模板，并 `assert` id 对齐；`quoting=3` 会让 `test.id` 带上字面引号。
- **gensim 4**：`glove2word2vec` 已删除，要手动给文件加首行"词数 维度"；用 `model.wv.*`；`size` 改名为 `vector_size`。
- **导师的 IMDB 脚本包**：先跑 `imdb_process.py`（GloVe → pickle）；设备硬编码为 `cuda:0`；`imdb_bert_scratch.py` 其实是"手写分类头 + 预训练编码器"，不是从零训练。
