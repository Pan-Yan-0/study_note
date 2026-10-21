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
- **实验指纹**：provider、模型、提示词、是否开思考、服务地址、max_tokens、temperature、数据哈希。续跑时这些必须和原来一致。
- **max_tokens 包含思考内容**：开思考时容易写满被截断，没有答案的按错误计，并单独统计截断数。
- **评测解析**：从回答里取标签时，词和数字要一致才算；改了解析规则，要用新规则重算所有已有结果。
- **算力**：
  - Kaggle：GPU 有每周额度。
  - Colab：免费 T4，要一直开着页面，单次最多 12 小时。
  - 腾讯 Cloud Studio：T4 可用。`ollama pull` 会被拦，改用 curl 从 `registry.ollama.ai/v2/library/<名字>/manifests|blobs` 下载，核对 SHA-256，manifest 按原字节保存。
  - 阿里云 PAI-DSW：常常抢不到卡。
  - 本机显卡只有 4 GB。
  - T4 估时按约 10 TFLOPS 算。

## 7. 当前进度

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
  - 10-20 起 SemEval 的工作在单独的对话里做；原来的对话只做学习（每晚的笔记任务绑定在那里）。
  - 下一步严格按导师的清单：先大模型 instruct / Qwen（零样本、少样本、CoT，再 LoRA SFT），然后 R-Drop、SCL、MoCo，再 DeepSeek 扩充。清单之外的想法排在后面。
  - 第三组（`notebooks/dico_llm_instruct`，Qwen3-4B-Instruct-2507 的指令提示，不训练）已跑完，结果在 `metrics/llm_instruct/`，Claude 从缓存复算过：
    - dev 对称化 WF1 最高是 cot（英 .618、西 .604、巴 .506、跨语言 .535），zero 英 .584、巴 .361、跨语言 .313，都远低于 mDeBERTa（英 .805）。
    - 原方向常把 BACKWARD 判成 FORWARD（zero 英语 97 / 190）；巴斯克语和跨语言掉得多。
    - 英语训练集 select 子集上，按事先定的规则选出 few，但它和 zero 分不出高低（+.014，[−.009, +.034]）；加计数规则的 zero_rules 降 .132（[−.162, −.102]），把大量预测推向 O。
    - cot 81–96% 写满 320 token 后被强制作答；生成每批 8 条，cot 跑了约 7 小时（运行前没跑 SMOKE）。
    - `model_revision` 没记下来（transformers 5.19 的 config 里没有 `_commit_hash`），下个 notebook 改用 `snapshot_download` 取 commit 号。
  - 第四组（`notebooks/dico_qwen_sft`，Qwen3-4B-Instruct-2507 的 LoRA SFT，zero 模板，只算答案字母的损失）第一版的 SMOKE 已在 Kaggle T4 上跑通（零样本概率和第三组一致；每步 5.7 秒、峰值 8.7 GiB，英语每个种子约 45 分钟；模型 commit `cdbee75f`）。GPT 审过后改成第二版（`dico_qwen_sft-2`：检查点逐个验证、载入权重后逐个核对张量、torchao 只清 site-packages、记录代码哈希、SMOKE 四类都有并实测中断恢复和重新加载），要先跑新的 SMOKE 再正式运行：
    - 数据划分和第二组 fold 0 逐行相同（训练 2,190、内层验证 244、留出 608），英语 fold 0 上和 mDeBERTa 12 轮配方的折外预测按种子配对比较。
    - 事先定好的蒸馏条件：英语 fold 0 上，同种子的 sym WF1 差（Qwen SFT − mDeBERTa）三个种子平均大于 0，或同种子两模型概率 0.5 / 0.5 平均后比 mDeBERTa 高，而且按家族配对重抽样的区间下限大于 0；都不满足就不做蒸馏。这只是探索性门槛，通过了也要和同数据、同预算的 mDeBERTa 对照。
    - 训练部分 E / F / B / O 是 574 / 631 / 631 / 354（O 只占 16%，不是四成）；dev 有 15 行（8 个 pair）的文本在训练集里出现过，选择看 fold 0。
    - 本地用假的 torch / transformers / peft 测过续跑：训练中途、评估后存检查点前、读概率时被打断，都能接着跑，选中的权重能复现记录的验证分数。
  - 再下一步：在 mDeBERTa 上逐个加 R-Drop、SCL、MoCo（英语基线是 12 轮 + 对称化）。
- **原理学习**：第一章（情感分类、反向传播）已通关，笔记在 `ML_note/原理笔记/`；第二章跟着 SemEval 实验学，现在讲句对输入和 A、B 对调。
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
