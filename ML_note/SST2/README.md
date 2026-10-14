# SST-2 指令学习实验

不训练模型，把影评填进一段指令，让大模型直接回答 -1（负面）或 1（正面），在 GLUE SST-2 验证集全部 872 条上统计准确率。任务来自导师的《1-使用大模型进行指令学习》材料。

已完成两部分：

- `basic` 指令、关闭思考时，DeepSeek 与 Qwen3.5 4B 的对比。
- DeepSeek 在材料里三个指令模板（basic / cot / role_cot）× 思考开关下的全部组合。

Qwen3.5 各尺寸的同样组合，以及 DeBERTa 微调对照组，还在等 Kaggle 运行（见“尚缺”）。

## 文件

| 文件 | 内容 |
|---|---|
| [code/sst2_instruction_eval.py](code/sst2_instruction_eval.py) | 评测脚本：DeepSeek API、Ollama 和其他 OpenAI 兼容服务通用；三种指令模板（basic / cot / role_cot），断点续跑，导出逐条 CSV 与汇总 JSON |
| [notebooks/teacher_sst2_ollama.ipynb](notebooks/teacher_sst2_ollama.ipynb) | Kaggle 上安装 Ollama 0.40.3、下载 `qwen3.5:4b` 并调用上面脚本的 Notebook，内置同一份脚本和 872 条数据（这一版在 `apt-get update` 报错时会继续安装 `zstd`，其余与运行的版本相同） |
| DeepSeek basic、关思考：[逐条](metrics/deepseek_deepseek-flash_basic_think-off_all_a4d72f5968bf.csv)、[汇总](metrics/deepseek_deepseek-flash_basic_think-off_all_a4d72f5968bf_summary.json)、[错例](metrics/deepseek_deepseek-flash_basic_think-off_all_a4d72f5968bf_errors.csv) | 872 条逐条结果（标签、预测、解析方式、耗时、token 数、原句），准确率与混淆矩阵，42 条错例 |
| Qwen3.5 4B basic、关思考：[逐条](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd.csv)、[汇总](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_summary.json)、[错例](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_errors.csv) | 同上，54 条错例 |
| DeepSeek 其余 5 种组合：`metrics/deepseek_deepseek-flash_<模板>_think-<off/on>_all_<指纹>` 的 `.csv`、`_summary.json`、`_errors.csv` | 每组 872 条；cot + 思考跑了两次（`b77435b2aaa8` 上限 4096，`cb39e1849fae` 上限 8192） |
| [metrics/deepseek_matrix_logs/](metrics/deepseek_matrix_logs/) | DeepSeek 组合实验的终端输出（开头核对了脚本和数据的 SHA-256）、cot + 思考的单条完整返回、脚本生成的汇总表 |
| [metrics/kaggle_env/](metrics/kaggle_env/) | Qwen3.5 4B 那次 Kaggle 运行的环境：`nvidia-smi`、`ollama ps`、Ollama 服务版本 |

文件名末尾的 12 位编号是实验指纹，由后端、服务地址、模型、模板、思考开关、生成参数和有序数据一起算出；改动任何一项都会存成新文件。

## 设置

| 项目 | 内容 |
|---|---|
| 数据 | GLUE SST-2 验证集 872 条（正面 444、负面 428），标签 0/1 换成 -1/1。test 集标签不公开，算不了准确率。所有实验用同一份文件（SHA-256 `a7a3bfee…`，汇总里的数据指纹同为 `be83d50b…`） |
| 指令 | 三个模板都照抄材料，作为单轮用户消息发送，末尾不加 `<think>`；思考由接口的开关控制 |
| 不思考 | temperature 0；basic 最多输出 64 个 token，cot 和 role_cot 最多 1024 个 |
| 思考 | 请求里写 temperature 0.6，但 DeepSeek 思考模式不支持 temperature（设了不报错也不起作用），top_p 最低也只能到 0.95，所以采样的随机性关不掉；最多输出 4096 个 token（含思考），cot + 思考另用 8192 重跑了一次 |
| 判分 | 从回答中解析 -1 / 1，解析不出按错误计（见下面“解析规则的一个漏洞”） |

`basic` 模板原文（`{影评}` 处填入句子）：

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the given text from an online review and determine the sentiment polarity. Return a single number of either -1 and 1, with 1 being positive and -1 being the negative sentiment.
No further explanation or justification is required.

### Question:
{影评} What's the sentiment polarity of the text?

### Response:
```

`cot` 模板（材料第 5 页循环里的版本）在开头说明之后加了一句要求先逐步推理，任务描述里去掉了“不需要解释”，影评放在 `### Input:` 下：

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
Analyze the given text from an online review and determine the sentiment polarity. Return a single number of either -1 and 1, with -1 being negative and 1 being the positive sentiment.

### Input:
{影评}

### Response:
```

`role_cot` 模板（材料第 4 页）与 cot 相同，只在 `### Instruction:` 下第一句加了 `You are an expert of consumer comment analysis.`。

## 结果一：basic 指令，DeepSeek 与 Qwen3.5 4B

关闭思考，temperature 0，max_tokens 64。

| 模型 | 运行方式 | 正确 / 872 | 准确率 [Wilson 95%] | 负面召回 | 正面召回 | 单条请求耗时 |
|---|---|---:|---|---:|---:|---:|
| `deepseek-flash` | 本机调用 DeepSeek 官方 API，8 路并发 | 830 | 95.18% [93.55, 96.42] | 98.36%（421/428） | 92.12%（409/444） | 1.00 秒 |
| `qwen3.5:4b`（Q4_K_M） | Ollama 0.40.3，Kaggle T4，单路 | 818 | 93.81% [92.01, 95.22] | 95.33%（408/428） | 92.34%（410/444） | 0.19 秒 |

| 真实标签 | DeepSeek 预测 -1 | DeepSeek 预测 1 | Qwen 预测 -1 | Qwen 预测 1 |
|---|---:|---:|---:|---:|
| 负面 -1 | 421 | 7 | 408 | 20 |
| 正面 1 | 35 | 409 | 34 | 410 |

逐条配对（同一 872 条，按 idx 对齐）：

| | Qwen 对 | Qwen 错 |
|---|---:|---:|
| **DeepSeek 对** | 804 | 26 |
| **DeepSeek 错** | 14 | 28 |

- DeepSeek 多对 12 条（1.38 个百分点）。McNemar 精确检验只看两个模型意见不一致的 40 条（26 对 14），双侧 p = 0.081，在 0.05 水平上不显著。
- 两个模型对 832 条（95.4%）给出相同预测，Cohen κ = 0.91。
- 两个模型的错误都以正面判成负面为主（35 条、34 条）；负面判成正面 DeepSeek 7 条、Qwen 20 条。
- token：DeepSeek 输入 104,380、输出 1,328；Qwen 输入 116,559、输出 2,186（每条回答 2–3 个 token）。两者的分词器和对话模板不同，输入 token 数不能直接比较。
- 参照：LoRA 论文表 2 在同一个 SST-2 验证集上报告的微调准确率为 RoBERTa-base 全量 94.8、LoRA 95.1，RoBERTa-large LoRA 96.2，DeBERTa-XXL LoRA 96.9（5 个种子的中位数，取最佳 epoch；见[实验结果汇总](../实验结果汇总.md) 1.3 节）。训练条件和统计方式不同，只能看量级。

### 错例

下面只描述句子本身。关闭思考时模型只返回标签，不能据此断言模型内部的出错原因。

**两个模型都错（28 条，其中 23 条是正面判成负面）**：多是褒贬并列的句子，也有几条单看很难判为正面。

- 667：good film , but very glum .
- 481：funny but perilously slight .
- 143：a solid film ... but more conscientious than it is truly stirring .
- 273：minority report is exactly what the title indicates , a report .
- 171（负面判成正面）：rarely has leukemia looked so shimmering and benign .

**只有 DeepSeek 错（14 条，其中 12 条是正面判成负面）**：正面评论里有 heartbreaking、harrowing、dark、grimly 这类字面偏负面的词。

- 117：moody , heartbreaking , and filmed in a natural , unforced style that makes its characters seem entirely convincing even when its script is not .
- 683：true tale of courage -- and complicity -- at auschwitz is a harrowing drama that tries to tell of the unspeakable .
- 88：fresnadillo 's dark and jolting images have a way of plying into your subconscious like the nightmare you had a week ago that wo n't go away .
- 780：a grimly competent and stolid and earnest military courtroom drama .

**只有 Qwen 错（26 条，其中 15 条是负面判成正面）**：负面评论里没有直接的负面词，贬义靠反讽或比较表达。

- 183：the lower your expectations , the more you 'll enjoy it .
- 312：i 'll bet the video game is a lot more fun than the film .
- 724：it 's everything you do n't go to the movies for .
- 695：sit through this one , and you wo n't need a magic watch to stop time ; your dvd player will do it for you .

## 结果二：DeepSeek 的三个模板 × 思考开关

每组 872 条，8 路并发，每组用时约 3 分钟。“前 200 条”只统计验证集前 200 条（正面 99、负面 101），用来和只跑前 200 条的 Qwen 组对照。

| 模板 | 思考 | 正确 / 872 | 准确率 [Wilson 95%] | 负面召回 | 正面召回 | 无效 | 前 200 条 | 输出 token / 条 | 单条请求耗时 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| basic | 关 | 830 | 95.18% [93.55, 96.42] | 98.36% | 92.12% | 0 | 96.5% | 2 | 1.00 秒 |
| basic | 开 | 838 | 96.10% [94.60, 97.20] | 98.36% | 93.92% | 0 | 98.0% | 80 | 1.43 秒 |
| cot | 关 | 833 | 95.53% [93.94, 96.71] | 97.20% | 93.92% | 5\* | 97.0% | 124 | 1.77 秒 |
| cot | 开（上限 8192） | 836 | 95.87% [94.34, 97.00] | 98.13% | 93.69% | 0 | 97.5% | 125 | 1.76 秒 |
| cot | 开（上限 4096，重复运行） | 841 | 96.44% [95.00, 97.48] | 98.36% | 94.59% | 1 | 98.5% | 122 | 1.71 秒 |
| role_cot | 关 | 833 | 95.53% [93.94, 96.71] | 97.20% | 93.92% | 0 | 96.5% | 132 | 1.89 秒 |
| role_cot | 开 | 838 | 96.10% [94.60, 97.20] | 97.90% | 94.37% | 0 | 97.5% | 118 | 1.75 秒 |

\* 这 5 条其实都给出了正确答案，见下面“解析规则的一个漏洞”；按答案计入后为 838/872 = 96.10%。

开思考时输出 token 含思考部分。cot + 思考两次运行的唯一设置差别是输出上限，4096 那次有 1 条写满上限被截断、没有答案（idx 830，思考写了 17,025 个字符），所以 8192 那次作为这一格的主结果。

- **思考开关**：三个模板都是开思考时略高，basic +0.92、cot +0.34、role_cot +0.57 个百分点，主要是正面判成负面的错误变少（basic 从 35 条降到 27 条）。但没有一个差别显著：basic 关对开不一致的 18 条里是 5 对 13，McNemar p = 0.096；cot 是 6 对 9，p = 0.61；role_cot 是 4 对 9，p = 0.27。
- **模板**：关思考时，cot、role_cot 对 basic 都是 833 对 830，p = 0.68；cot 对 role_cot 不一致的 22 条正好各对 11 条。开思考时三个模板的正确数在 836–838 之间（重复运行那次 841）。所有两两比较里只有关思考的 basic 对 cot + 思考的重复运行 p < 0.05（后者多对 17 条、少对 6 条，p = 0.035）；同一设置的另一次运行只有 p = 0.26，而且一共比较了十几对，不能当作显著。
- **同一设置重跑也会变**：思考模式的采样关不掉，cot + 思考两次运行有 6 条预测不同。除了被截断的 idx 830，另外 5 条是重新采样带来的变化（4096 那次多对 5 条，p = 0.062）。所以单次运行相差几条、0.5 个百分点左右，都在这个量级的噪声之内；要比较这些设置，需要每种设置重复运行几次。
- **思考很短**：思考内容的中位数是 238–385 个字符（大约 60–90 个 token），最长的有 1.5 万–2 万字符；输出 token 的中位数 57–89。
- **共同的错例**：24 条在 6 种设置下全都判错，其中 20 条的标签是正面。例如 112 “hilariously inept and ridiculous .”、230 “reign of fire looks as if it was made without much thought -- and is best watched that way .” 的标签是正面，377 “nothing is sacred in this gut-buster .” 的标签是负面。这类句子单看很难判对，换模板、开思考都没有改变结果。
- **开思考改对和改错的例子**（basic）：改对 13 条，如 667 “good film , but very glum .”、143 “a solid film ... but more conscientious than it is truly stirring .”、683 “true tale of courage ... is a harrowing drama ...”；改错 5 条，如 675 “slick piece of cross-promotion .”（标签正面）、95 “this riveting world war ii moral suspense story deals with the shadow side of american culture ...”（标签负面）。
- token 合计（5 种新组合的全量运行，加上 cot + 思考的重复运行）：输入 765,940，输出 611,392。

### 解析规则的一个漏洞

cot、关思考那组被判为无效的 5 条，回答的最后一行分别是：

| idx | 真实标签 | 最后一行 |
|---:|---:|---|
| 72 | 1 | `The answer is: **1**` |
| 140 | -1 | `The sentiment polarity is: **-1**` |
| 393 | 1 | `The sentiment polarity is: **1**` |
| 723 | 1 | `The sentiment polarity should be **1**.` |
| 867 | -1 | `The answer is: **-1**` |

脚本接受 “answer is 1” 和 “Answer: 1”，但不接受 “is:” 连写和 “should be”，所以这 5 条明确的答案被判成了无效。表里按脚本的原规则报告；Kaggle 上的 Qwen 实验还在用同一版脚本，等结果都回来后，再统一放宽规则，用保存的逐条回答重新判分（不需要重新请求）。

## 尚缺

- Qwen3.5 的 0.8b、2b、4b、9b 在三个模板 × 思考开关下的结果，以及 DeBERTa-v3-base 对照组（只用 16 / 64 / 256 / 1024 条和用全部训练集，各 3 个种子，在同一份 872 条上评测）。两部分已写成一个 Kaggle Notebook，等待运行。
- 放宽解析规则后，对所有运行统一重新判分。

## 来源

- 实验：
  - basic、关思考的两组由潘岩运行：DeepSeek 在本机，Qwen 在 Kaggle。
  - DeepSeek 的三模板 × 思考组合由 GPT6astra 在潘岩的本机运行，用的是同一版脚本（SHA-256 `187d2f88…`）和同一份数据（`a7a3bfee…`），终端记录见 [metrics/deepseek_matrix_logs/](metrics/deepseek_matrix_logs/)。
  - `metrics/` 中的文件由这几次运行导出。
- 代码：评测脚本初版由 Claude 编写，GPT6.1sol 加入了实验指纹、更严格的答案解析和断点修复；Kaggle Notebook 由 GPT6.1sol 编写，Claude 补了 `apt-get update` 报错时的容错。
- 本文：由 Claude 整理，参考了 DeepSeek 实验记录；区间、召回率、配对检验和错例统计由 `metrics/` 中的逐条结果计算。
