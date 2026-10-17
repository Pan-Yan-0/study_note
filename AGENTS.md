# AGENTS.md · Pan-Yan-0 学习项目协作档案

所有在这个仓库里工作的 AI（Claude、GPT，以及以后加入的）共用这份约定。开始工作前先读一遍；发现内容过时，就在这里改，不要另起一份。2026-10-16 以前的旧档案在 [agents/AGENTS旧版.md](agents/AGENTS旧版.md)，只作历史参考。

## 1. 最重要的几条

1. **称呼和语言**：称呼 Pan-Yan-0（GitHub 用户名），不用中文名；交流和文档都用中文。
2. **仓库是公开的，导师也会看**：不写学习画像、经验值、个人邮箱、令牌和本机路径。
3. **工作分支是 `note`**：`master`、`d2l`、`cuda_study`、`math_modeling` 是别的分支，不要动（`d2l` 由每周的定时任务单独维护）。
4. **任何决策都要有实验支持**（导师原话："扩充数据，参数选取，我也说不清好不好，实验结果说明好不好"）：
   - 改动都和基线对照，一次只改一处。
   - 数据少时跑多个随机种子，差别小于种子间的波动就不下结论。
   - 建议改什么时，同时给出验证它的对照实验。
5. **一个 AI 写的代码，由另一个审过再跑。** 例如 SST-2 里 Mistral 的 39.68% 其实是评测解析器太严造成的，改正后是 90.60%。这种错，写代码的一方最难发现。
6. **重要问题直接重写**：发现方案或文档有结构性问题，直接重写，不做修修补补。

## 2. 参与者

| 谁 | 做什么 |
|---|---|
| Pan-Yan-0 | 研究生，方向是 NLP 与大模型，职业目标是 AI Infra / 大模型推理优化。在 Kaggle、Colab、腾讯 Cloud Studio 上运行实验并带回结果；做最后决定 |
| 导师 | 布置实验任务，会看本仓库，要求 README 记录"结果、问题和下一步计划" |
| Claude | 讲解和出题、整理笔记、写实验代码、每晚自动提交笔记。教学方法和学习画像在 Pan-Yan-0 的私有 skill `cognitive-tutoring` 里，不进本仓库 |
| GPT | 写过 SST-2 评测脚本等代码（GPT6.1sol）；SemEval 由 GPT6astra 参与方案讨论和代码 |

AI 之间不能直接对话，由 Pan-Yan-0 转交文件。大家共同遵守的规则写在这里，新加入的 AI 读这份文件就能接上。

## 3. 仓库结构

```text
AGENTS.md / CLAUDE.md      协作约定（CLAUDE.md 只是引用本文件）
README.md                  首页：目录树、实验和笔记的入口
agents/                    旧版档案
skills/                    教学 skill 的旧版，现行版本在 Pan-Yan-0 的私有 skill 里
ML_note/
├── 实验结果汇总.md         全部实验结果速览
├── 实验/IMDB/  实验/SST2/  每个实验一个文件夹：README、code/、notebooks/、metrics/
├── 原理笔记/               BERT/  DeBERTa/  LLM/  LoRA/  反向传播/
└── 吴恩达机器学习/          课程笔记
Coursera-ML-AndrewNg-Notes-master/   课程资料
```

- 文件放在主题文件夹的叶子节点里，只有汇总类文档放在上层。
- 新增或移动文件时，同步更新根目录 README 和相关 README 里的链接，并检查没有断链。

## 4. 笔记和文档的写法

笔记和汇总是写给导师和别人看的。

- **汇总的排序**：本仓库的成果在前，别人的（如论文）在后；重要的在前；新的在前。
- **实验 README**：分"结果 / 问题 / 下一步计划"三段写（导师的要求）。
- **原理笔记**：讲解按概念组织；"检查题与我的作答"用表格（题目 | 我的作答 | 判定与要点），"我的作答"取 Pan-Yan-0 原话的意思；还没作答的题折叠起来并附参考答案；最后是"易混点"和"下一步"。
- **不写**：实验过程的流水账、日期、画像和经验内容。代码只保留必要的注释。
- **来源一节**：写清谁运行了实验、谁写了哪部分代码，以及引用的论文和链接。

## 5. 提交和推送

- **作者是 Pan-Yan-0，提交者是写代码的 AI**：
  ```bash
  git -c user.name="Claude" -c user.email="noreply@anthropic.com" \
      commit --author="Pan Yan <112330427+Pan-Yan-0@users.noreply.github.com>"
  ```
  其他 AI 把 `user.name`、`user.email` 换成自己的。
- **提交信息末尾的署名**：Claude 写的加 `Co-Authored-By: Claude …` 和会话链接；含 GPT 写的代码、Notebook 或文字时，加 `Co-authored-by: GPT6.1sol <noreply@openai.com>`。换了别的 GPT 模型，就把名字换成实际参与的那个，拿不准时问 Pan-Yan-0。
- **推送前**：先 `git fetch`，确认不会覆盖别人的提交；检查 diff 里没有密钥、令牌和邮箱。
- **只用普通推送**：不强推，不为了补署名改写已推送的历史。推送失败就停下来，说明是哪一步、报了什么错。
- **结果文件**：`.gitignore` 忽略了 `*.csv`、`*.json`，要提交的指标文件用 `git add -f`。

## 6. 实验的约定

- **Notebook**：
  - 能断点续跑，带 SMOKE 小规模开关。
  - 默认把实验跑完，不设中途停掉实验的时间预算。
  - 写清它和导师原脚本有哪些区别。
  - 在 Kaggle、Colab 和普通 Linux（如 Cloud Studio）上都能跑，换平台时能导入上一次的结果包接着跑。
  - 打印的提示写清已经完成了什么、还需要人手动做什么。
- **实验指纹**：provider、模型、提示词、是否开思考、服务地址、max_tokens、temperature、数据哈希。续跑时这些必须和原来一致。
- **max_tokens 包含思考内容**：开思考时容易写满被截断，没有答案的按错误计，并单独统计截断数。
- **评测解析**：从回答里取标签时，词和数字要一致才算；改了解析规则，要用新规则重算所有已有结果。
- **算力**：
  - Kaggle：GPU 有每周额度。
  - Colab：免费 T4，要一直开着页面，单次最多 12 小时。
  - 腾讯 Cloud Studio：T4 可用。`ollama pull` 会被拦，改用 curl 从 `registry.ollama.ai/v2/library/<名字>/manifests|blobs` 下载，核对 SHA-256，manifest 按原字节保存。
  - 阿里云 PAI-DSW：常常抢不到卡。
  - 本机显卡只有 4 GB。T4 估时按约 10 TFLOPS 算。

## 7. 当前进度

- **第一轮 SST-2 指令学习、第二轮 IMDB**：都已完成，结果见[实验结果汇总](ML_note/实验结果汇总.md)和两个实验的 README。
- **第三轮 SemEval-2027 Task 2**：已选定题目，还没开工，开工时间由 Pan-Yan-0 决定，见第 8 节。
- **原理学习**：正在学反向传播，见 `ML_note/原理笔记/反向传播/`。
- **待办**：GitHub 仓库首页默认显示 `master`，`note` 的最新内容在首页看不到。改法等 Pan-Yan-0 决定：把默认分支改成 `note`、在 `master` 的 README 加入口，或者合并（会动别的分支，AI 不自己做）。

## 8. SemEval-2027 Task 2：DiCo-NLI

### 8.1 任务（来自官方 README）

- **输入**：一对短语，前提 A 和假设 B，来自 PhrasIS（图片描述和新闻标题）。(A, B) 和 (B, A) 两个方向都要预测。
- **标签**：EQUIVALENCE、FORWARD_ENTAILMENT、BACKWARD_ENTAILMENT、NEGATIVE_OTHER。例如前提 "Everyone is hungry"、假设 "Someone is hungry" 是 FORWARD，对调后是 BACKWARD；EQUIVALENCE 对调后还是它自己。
- **赛道**：英语、西班牙语、巴斯克语三个单语赛道，训练集各 3,042 条、dev 各 660 条；混合语言赛道训练集 18,252 条、dev 3,960 条。
- **指标**：Weighted F1、SoftCons（两个方向是否自洽，不看对错）、HardCons（两个方向都答对），官方没有指定主指标。用官方评分脚本：`python3 -m evaluation_functions --gold <reference> --predictions <file> --output-dir <dir>`，提交文件只有 `instance_id,label` 两列。
- **基线**：主办方给了 random 和 majority 基线。试点（英语，只含三个正向标签）里 DeBERTa-v3-base 是 WF1 0.75、SoftCons 0.84、HardCons 0.79，设置和正式赛不完全一样，只作参考。
- **规则**：允许外部数据和大模型，但必须在系统论文里写清楚模型、提示词和外部数据。PhrasIS 原数据能不能拿来训练，以官方最终规则为准，确认之前不用。
- **时间**：评测 2027-01-10 至 01-31，2 月交系统论文。
- 官方说明：[SemEval-2027 Task 2 DiCo-NLI README](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/blob/main/README.md)

### 8.2 导师的要求

1. 先把现有模型跑出结果：mBERT、多语言 DeBERTa、大模型 instruct、Qwen 等。BERT 类做 NLI 就是 `[CLS] A [SEP] B [SEP]`。
2. 再试 SCL、R-Drop、MoCo 和 CoT。
3. 数据少，用 DeepSeek 扩充，"越多越好，增加的过程中看性能会不会升"。
4. 不先看综述，动手最重要。
5. 之后导师会讲反事实推论和因果推断。
6. 仓库记录结果、问题和下一步计划。

### 8.3 多个 AI 怎么合作（草案，开工时碰撞后定稿）

1. **独立出方案**：同一份任务简报分别交给每个 AI，各自写方案，先不看别人的。所以本文件只放任务事实和导师要求，不放任何一方的方案。
2. **交换挑错**：Pan-Yan-0 把各方案交换过去，互相找问题。
3. **分歧用实验定**：能用实验定的写成对照实验，定不了的由 Pan-Yan-0 决定。
4. **共识写回这里**：写进本节，以及 SemEval README 的"下一步计划"。

开工后的固定做法：
- 实验放在 `ML_note/实验/SemEval_DiCo-NLI/`，README 分"结果 / 问题 / 下一步计划"。
- 一律用官方 dev 和官方评分脚本，三个指标按赛道分开报。
- 每个实验登记编号、作者（哪个 AI）、commit、配置、种子和三个分数。
- 外部数据、生成数据和提示词都留档，写论文时要用。
- 一个 AI 写的代码由另一个审过再跑。

## 9. 定时任务（Claude 这边）

- **每晚 21:51（北京时间）「每晚提交学习笔记」**：在 Pan-Yan-0 的学习对话里整理当天的问答，提交到 `note` 分支，并结算学习进度。没有新内容就不提交。它只读得到绑定的那个对话，换新对话学习时要重新设置。
- **每周日 10:00「推送 D2L 学习记录」**：把伴学簿导出到 `d2l` 分支。

## 10. 关键技术事实（避免重查）

- **Kaggle 提交**：用 `sample.copy()` 作模板，并 assert id 对齐；`quoting=3` 会让 test.id 带上字面引号。
- **gensim 4**：`glove2word2vec` 已删除，要手动给文件加首行"词数 维度"；用 `model.wv.*`；`size` 改名为 `vector_size`。
- **导师的 IMDB 脚本包**：
  - 先跑 `imdb_process.py`（GloVe → pickle），CNN、LSTM、GRU、Attention、Transformer 系列都依赖这个 pickle；BERT 系列独立。
  - 设备硬编码为 `cuda:0`。
  - 显存不够时依次试：batch_size 减半 → max_len 降到 256 → 换 DistilBERT。
  - `imdb_bert_scratch.py` 其实是"手写分类头 + 预训练编码器"，不是从零训练。
