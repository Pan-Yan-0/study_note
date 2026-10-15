# SST-2 指令学习实验

不训练模型，把影评填进一段指令，让大模型直接回答 -1（负面）或 1（正面），在 GLUE SST-2 验证集全部 872 条上统计准确率。任务来自导师的《1-使用大模型进行指令学习》材料。为了知道这个准确率算高还是低，另外用同一份验证集做了微调对照组。

已完成四部分：

1. `basic` 指令、关闭思考时，DeepSeek 与 Qwen3.5 4B 的对比。
2. Qwen3.5 的 0.8b、2b、4b、9b 四个尺寸，以及它们在三个指令模板（basic / cot / role_cot）× 思考开关下的全部 24 种组合。除 basic + 关思考外都只跑前 200 条；9b 的 cot、role_cot 开思考没跑完（184、168 条）。
3. DeepSeek 在三个模板 × 思考开关下的全部组合。
4. DeBERTa-v3-base 微调对照组：只用 16 / 64 / 256 / 1024 条训练数据和用全部 67,349 条，各 3 个种子。

主要结论：

- **模型大小**：basic 指令、关思考时，0.8b 78.78%，2b 83.14%，4b 93.81%，9b 93.35%。从 2b 到 4b 提升明显，4b 和 9b 之间没有显著差别（p = 0.64）。DeepSeek 95.18%，比 9b 显著更高（p = 0.014），比 4b 高但不显著（p = 0.081）。
- **指令模板**（关思考）：要求先一步步推理的 cot 模板主要帮了小模型，0.8b 提高 3.5 个百分点、2b 提高 5.5 个百分点（p = 0.36、0.061）；4b、9b 和 DeepSeek 上差别都在 2.5 个百分点以内，都不显著。模型越大越常无视这句要求，9b 有一半多的条目直接只答一个数字。
- **思考开关**：在 4096 个输出 token 的上限内，Qwen 的 basic 模板开思考，整体准确率和关思考差不多（0.8b 除外）；cot、role_cot 模板再开思考，模型要推理两遍，大量条目写不完被截断，所有尺寸都变差，最多掉 39 个百分点。每条还要慢几十到几百倍。只看写完的条目，开思考不比关思考差，但这个比较本身偏向开思考。DeepSeek 开不开思考差别不显著。
- **和微调比**：DeBERTa-v3-base 用全部训练集微调后是 95.60%，和 DeepSeek 打平（p ≈ 0.7）。只用 1024 条训练的 DeBERTa（92.97%）和直接提问的 Qwen 4b（93.81%）差不多（p = 0.35–0.46）。只用 16 条时只有 61.62%，种子之间也差得多。

## 文件

| 文件 | 内容 |
|---|---|
| [code/sst2_instruction_eval.py](code/sst2_instruction_eval.py) | 评测脚本：DeepSeek API、Ollama 和其他 OpenAI 兼容服务通用；三种指令模板，断点续跑，导出逐条 CSV 与汇总 JSON |
| [code/rescore_offline.py](code/rescore_offline.py) | 用当前解析规则，从保存的原始返回离线重新判分，不重新调用模型 |
| [code/sst2_report_stats.py](code/sst2_report_stats.py) | 复算本文的统计数字（区间、召回、配对检验、DeBERTa 均值与标准差），只用 Python 标准库 |
| [code/deberta_sst2.py](code/deberta_sst2.py) | DeBERTa 对照组的训练与评测脚本 |
| [notebooks/teacher_sst2_ollama.ipynb](notebooks/teacher_sst2_ollama.ipynb) | 结果一里 Qwen3.5 4B 那次运行的 Kaggle Notebook |
| [notebooks/sst2_remaining_experiments.ipynb](notebooks/sst2_remaining_experiments.ipynb) | 结果二、三、五的 Kaggle Notebook（Qwen 各组的第一次运行和 DeBERTa） |
| [notebooks/sst2_finish_experiments.ipynb](notebooks/sst2_finish_experiments.ipynb) | 结果三剩余各组的 Kaggle Notebook（从第一次的断点接着跑） |
| [notebooks/sst2_more_models.ipynb](notebooks/sst2_more_models.ipynb) | Mistral、Llama、Gemma、Phi-4 四个系列的 Notebook（Colab 或 Kaggle）：basic 全部 872 条、cot 前 200 条；不设时间限制，跑完的组跳过，没跑完的组从断点接着跑 |
| `metrics/` 根目录 | 结果一和结果四：DeepSeek 7 次全量运行和 Qwen 4B 第一次运行的逐条 `.csv`、`_summary.json`、`_errors.csv` |
| [metrics/qwen_matrix/](metrics/qwen_matrix/) | Qwen 24 组的逐条结果、汇总和错例；没跑完的两组汇总里 `completed` 为 false |
| [metrics/qwen_matrix_logs/](metrics/qwen_matrix_logs/) | Qwen 各组的终端输出、模型信息（量化方式、文件摘要）、执行状态和 `nvidia-smi`；[finish_run/](metrics/qwen_matrix_logs/finish_run/) 是第二次运行的记录，含服务日志摘录 |
| [metrics/deberta_sst2/](metrics/deberta_sst2/) | DeBERTa 15 次运行的汇总（`runs.csv`、`summary.md`）、每次的设置与学习曲线（`summary.json`）、872 条逐条预测和正类概率（`predictions.csv`）、训练日志和库版本 |
| [metrics/raw/](metrics/raw/) | DeepSeek 7 次和 Qwen 24 组的原始返回（含思考内容，gzip 压缩的 jsonl）和运行时导出的原汇总 |
| [metrics/deepseek_matrix_logs/](metrics/deepseek_matrix_logs/) | DeepSeek 组合实验的终端输出、cot + 思考的单条完整返回、汇总表 |
| [metrics/kaggle_env/](metrics/kaggle_env/) | 结果一里 Qwen3.5 4B 那次 Kaggle 运行的环境 |

文件名末尾的 12 位编号是实验指纹，由后端、服务地址、模型、模板、思考开关、生成参数和有序数据一起算出。

## 设置

| 项目 | 内容 |
|---|---|
| 数据 | GLUE SST-2 验证集 872 条（正面 444、负面 428），标签 0/1 换成 -1/1。test 集标签不公开，算不了准确率。所有实验用同一份文件（SHA-256 `a7a3bfee…`，汇总里的数据指纹同为 `be83d50b…`）。前 200 条是正面 99、负面 101 |
| 指令 | 三个模板都照抄材料，作为单轮用户消息发送，末尾不加 `<think>`；思考由接口的开关控制 |
| 不思考 | temperature 0；basic 最多输出 64 个 token，cot 和 role_cot 最多 1024 个 |
| 思考 | 请求里写 temperature 0.6；最多输出 4096 个 token（含思考），DeepSeek 的 cot + 思考另用 8192 重跑了一次。DeepSeek 思考模式不支持 temperature（设了不报错也不起作用），top_p 最低只能到 0.95，所以采样的随机性关不掉 |
| DeepSeek | `deepseek-flash`，官方 API，8 路并发，在本机运行 |
| Qwen | Ollama 0.40.3，Kaggle T4，逐条请求，每个请求的上下文 8192。0.8b、2b 为 Q8_0 量化，4b、9b 为 Q4_K_M。四个尺寸的 basic + 关思考跑全部 872 条，其余组合只跑前 200 条。分两次运行，第二次从第一次的断点接着跑，模型版本（digest）相同 |
| DeBERTa | `microsoft/deberta-v3-base` 加分类头，全部参数参与训练；训练数据取自 GLUE SST-2 训练集（67,349 条）。少样本组按正负各半抽样，batch 8 训练 1000 步；全量组 batch 32 训练 3 轮（6,315 步）。学习率 2e-5，前 10% 步数线性升温后线性下降，weight decay 0.01，fp16，最长 128 个 token。报告最后一步的结果，不按验证集挑选 |
| 判分 | 从回答中解析 -1 / 1，解析不出（包括被截断、没有答案）按错误计；规则见“解析规则” |

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
- 同样设置在另一个 Kaggle 会话里重跑 Qwen 4B（结果二里的 4b），872 条预测完全相同。
- token：DeepSeek 输入 104,380、输出 1,328；Qwen 输入 116,559、输出 2,186（每条回答 2–3 个 token）。两者的分词器和对话模板不同，输入 token 数不能直接比较。

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

## 结果二：Qwen3.5 四个尺寸

basic 指令，关思考，全部 872 条。

| 模型 | 量化 | 正确 / 872 | 准确率 [Wilson 95%] | 负面召回 | 正面召回 | 无效 | 前 200 条 | 单条请求耗时 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `qwen3.5:0.8b` | Q8_0 | 687 | 78.78% [75.95, 81.37] | 86.92% | 70.95% | 2 | 74.5% | 0.21 秒 |
| `qwen3.5:2b` | Q8_0 | 725 | 83.14% [80.51, 85.48] | 96.50% | 70.27% | 12 | 83.5% | 0.12 秒 |
| `qwen3.5:4b` | Q4_K_M | 818 | 93.81% [92.01, 95.22] | 95.33% | 92.34% | 0 | 92.0% | 0.24 秒 |
| `qwen3.5:9b` | Q4_K_M | 814 | 93.35% [91.50, 94.82] | 96.96% | 89.86% | 0 | 94.0% | 0.33 秒 |
| `deepseek-flash`（参照） | — | 830 | 95.18% [93.55, 96.42] | 98.36% | 92.12% | 0 | 96.5% | 1.00 秒 |

- **相邻尺寸配对比较**：0.8b 对 2b，不一致的 122 条里是 42 对 80，p = 0.0007；2b 对 4b 是 13 对 106，p < 10⁻¹⁸；4b 对 9b 是 23 对 19，p = 0.64。所以在这个任务上，提升集中在 2b 到 4b 之间，9b 没有比 4b 更好。
- **小模型偏向判负面**：0.8b 和 2b 的正面召回只有 71% 和 70%，负面召回 87% 和 97%。4b 和 9b 两类召回都在 90% 以上。
- **无效输出**：2b 有 12 条回答 `0`（材料只允许 -1 和 1），0.8b 有 2 条回答 `-2`。
- **和 DeepSeek 比**：Qwen 9b 不一致的 38 条里是 11 对 27，p = 0.014；Qwen 4b 是 14 对 26，p = 0.081。
- 0.8b 每条比 2b 还慢；0.8b 的所有层都在 GPU 上，它的模型包另带一个推测解码用的草稿模型。

## 结果三：Qwen 的模板 × 思考

除了四个尺寸的 basic + 关思考（结果二，全部 872 条），其余组合只跑验证集前 200 条。9b 的 cot 开思考和 role_cot 开思考只跑到 184 条和 168 条。

前 200 条准确率（截断、无效都按错误计）：

| 模型 | basic 关 | basic 开 | cot 关 | cot 开 | role_cot 关 | role_cot 开 |
|---|---:|---:|---:|---:|---:|---:|
| 0.8b | 74.5% | 56.5% | 78.0% | 42.5% | 79.0% | 40.0% |
| 2b | 83.5% | 85.5% | 89.0% | 68.5% | 88.5% | 68.0% |
| 4b | 92.0% | 92.5% | 94.5% | 80.5% | 90.0% | 85.0% |
| 9b | 94.0% | 95.0% | 93.5% | 66.8%（184 条） | 94.0% | 70.8%（168 条） |

basic + 关思考取的是 872 条那次运行的前 200 条。最高的几格（9b basic 开 95.0%、4b cot 关 94.5%、9b basic 关和 role_cot 关 94.0%）彼此之间没有显著差别。

### 关思考时的模板

| 模型 | basic 对 cot | basic 对 role_cot | cot / role_cot 直接只答一个数字的条数 | cot 关的输出 token 中位数 | cot 关每条耗时 |
|---|---|---|---|---:|---:|
| 0.8b | 74.5% → 78.0%（18 对 25，p = 0.36） | → 79.0%（18 对 27，p = 0.23） | 0 / 0 | 208 | 2.5 秒 |
| 2b | 83.5% → 89.0%（9 对 20，p = 0.061） | → 88.5%（11 对 21，p = 0.11） | 1 / 13 | 229 | 3.7 秒 |
| 4b | 92.0% → 94.5%（3 对 8，p = 0.23） | → 90.0%（10 对 6，p = 0.45） | 60 / 74 | 127 | 3.0 秒 |
| 9b | 94.0% → 93.5%（4 对 3，p = 1） | → 94.0%（3 对 3，p = 1） | 105 / 104 | 3 | 4.2 秒 |

括号里是两种模板意见不一致的条目中，前者对、后者对的条数和 McNemar 检验的 p。“直接只答一个数字”指输出不超过 5 个 token，也就是没有按要求先写推理。

- **cot 模板主要帮了小模型**：0.8b 提高 3.5–4.5 个百分点，2b 提高 5–5.5 个百分点（p = 0.061、0.11），4b 有升有降，9b 不变。都没有达到 p < 0.05；只跑了 200 条，这种几个百分点的差别要更多数据才能确定。
- **模型越大，越常无视“先一步步推理”这句要求**：9b 有一半多的条目直接回答了一个数字，4b 有三分之一左右，0.8b 一次也没有。所以对 9b 来说，cot 模板有一半时候其实就是 basic。
- cot 和 role_cot 之间只有 4b 差得多（94.5% 对 90.0%，11 对 2，p = 0.022），其余三个尺寸几乎一样，DeepSeek 上也没有差别（见结果四）。这是 12 次比较里的一次，不能说明加角色设定有害。
- 无效输出大多是回答 `0`（材料只允许 -1 和 1）：9b 的 cot、role_cot 关各有 3 条和 2 条，0.8b role_cot 关有 3 条。

### 开思考：截断，以及怎样公平比较

开思考时，输出上限 4096 个 token 是思考和回答加在一起的，写满了还没给出答案就按错误计。小模型常在思考里反复确认、停不下来。下面是 0.8b 一条被截断的思考的结尾，它已经决定答 1，却一直重复到写满上限，没有输出答案：

````text
Okay, I will output 1.

Wait, one last check. Is there any reason it might be -1?
If the text is negative... no.
Okay, 1.

Final decision: 1.

Wait, I need to make sure I don't add any markdown like ``` or anything.
Just "1".
````

同一模板关思考对开思考：

| 组 | 跑完 | 截断 | 关 → 开（全部条目） | 只看开思考没截断的条目：关 → 开 | 不一致的条目 | p | 被截断的条目关思考时的准确率 | 开思考输出 token 中位数 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| 0.8b basic | 200 | 73（36%） | 74.5% → 56.5% | 82.7% → 89.0%（127 条） | 4 对 12 | 0.077 | 60.3% | 2,571 |
| 0.8b cot | 200 | 107（54%） | 78.0% → 42.5% | 82.8% → 90.3%（93 条） | 1 对 8 | 0.039 | 73.8% | 4,096 |
| 0.8b role_cot | 200 | 110（55%） | 79.0% → 40.0% | 81.1% → 88.9%（90 条） | 2 对 9 | 0.065 | 77.3% | 4,096 |
| 2b basic | 200 | 18（9%） | 83.5% → 85.5% | 86.3% → 94.0%（182 条） | 2 对 16 | 0.001 | 55.6% | 460 |
| 2b cot | 200 | 54（27%） | 89.0% → 68.5% | 92.5% → 93.8%（146 条） | 2 对 4 | 0.69 | 79.6% | 3,050 |
| 2b role_cot | 200 | 56（28%） | 88.5% → 68.0% | 91.0% → 94.4%（144 条） | 1 对 6 | 0.13 | 82.1% | 3,284 |
| 4b basic | 200 | 9（4%） | 92.0% → 92.5% | 95.3% → 96.9%（191 条） | 1 对 4 | 0.38 | 22.2% | 402 |
| 4b cot | 200 | 33（16%） | 94.5% → 80.5% | 95.8% → 96.4%（167 条） | 1 对 2 | 1 | 87.9% | 1,292 |
| 4b role_cot | 200 | 25（12%） | 90.0% → 85.0% | 92.0% → 97.1%（175 条） | 0 对 9 | 0.004 | 76.0% | 981 |
| 9b basic | 200 | 5（2%） | 94.0% → 95.0% | 94.9% → 97.4%（195 条） | 0 对 5 | 0.063 | 60.0% | 368 |
| 9b cot | 184 | 55（30%） | 93.5% → 66.8% | 94.6% → 95.3%（129 条） | 2 对 3 | 1 | 90.9% | 3,286 |
| 9b role_cot | 168 | 48（29%） | 94.0% → 70.8% | 96.7% → 99.2%（120 条） | 0 对 3 | 0.25 | 87.5% | 3,304 |

“不一致的条目”是在没截断的条目上，只有关思考对和只有开思考对的条数。9b 的两组没跑完，只比跑到的条目。

- **basic 模板开思考，整体差别不大**：2b、4b、9b 开不开思考差 0.5–2 个百分点，都不显著；只有 0.8b 截断太多，掉了 18 个百分点。
- **cot、role_cot 模板再开思考，所有尺寸都变差了**：除了 4b role_cot 掉 5 个百分点（p = 0.087），其余掉 14–39 个百分点，都非常显著。差别几乎全部来自截断：模型在思考里推理一遍，正文里按模板要求再推理一遍，0.8b 有一半以上、2b 和 9b 将近三成写不完。
- **被截断的不一定是难句**。basic 模板下被截断的条目，关思考时只对了 22–60%，确实偏难；cot、role_cot 模板下被截断的条目，关思考时对了 74–91%，9b cot 开被截断的 55 条里关思考对了 91%。所以后者主要是推理写得太长，不是句子难。
- **只看没截断的条目，开思考不比关思考差**，12 组里有 3 组显著更准（2b basic、0.8b cot、4b role_cot）。但这个比较也偏向开思考：留下的是开思考自己写完了的条目，写完的往往是它比较有把握的句子。所以只能说“写得完的时候，思考没有坏处”，不能说思考让模型更准。
- **代价**：开思考每条 15–46 秒，9b 的 cot、role_cot 开思考每条约 2 分钟，是关思考的几十到几百倍。

## 结果四：DeepSeek 的三个模板 × 思考开关

每组 872 条，8 路并发，每组用时约 3 分钟。

| 模板 | 思考 | 正确 / 872 | 准确率 [Wilson 95%] | 负面召回 | 正面召回 | 无效 | 前 200 条 | 输出 token 中位数 | 单条请求耗时 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| basic | 关 | 830 | 95.18% [93.55, 96.42] | 98.36% | 92.12% | 0 | 96.5% | 2 | 1.00 秒 |
| basic | 开 | 838 | 96.10% [94.60, 97.20] | 98.36% | 93.92% | 0 | 98.0% | 57 | 1.43 秒 |
| cot | 关 | 838 | 96.10% [94.60, 97.20] | 97.66% | 94.59% | 0 | 98.0% | 122 | 1.77 秒 |
| cot | 开（上限 8192） | 836 | 95.87% [94.34, 97.00] | 98.13% | 93.69% | 0 | 97.5% | 82 | 1.76 秒 |
| cot | 开（上限 4096，重复运行） | 841 | 96.44% [95.00, 97.48] | 98.36% | 94.59% | 1 | 98.5% | 82 | 1.71 秒 |
| role_cot | 关 | 833 | 95.53% [93.94, 96.71] | 97.20% | 93.92% | 0 | 96.5% | 128 | 1.89 秒 |
| role_cot | 开 | 838 | 96.10% [94.60, 97.20] | 97.90% | 94.37% | 0 | 97.5% | 89 | 1.75 秒 |

开思考时输出 token 含思考部分。cot + 思考两次运行的唯一设置差别是输出上限，4096 那次有 1 条写满上限被截断、没有答案（idx 830，思考写了 17,025 个字符），所以 8192 那次作为这一格的主结果。

- **思考开关**：basic 开思考多对 8 条（不一致的 18 条里 13 对 5，p = 0.096）；role_cot 多对 5 条（9 对 4，p = 0.27）；cot 开思考（8192）反而少对 2 条（4 对 6，p = 0.75）。都不显著。
- **模板（关思考）**：cot 比 basic 多对 8 条（13 对 5，p = 0.096），role_cot 比 basic 多对 3 条（13 对 10，p = 0.68），cot 比 role_cot 多对 5 条（11 对 6，p = 0.33）。开思考时三个模板的正确数在 836–838 之间（重复运行那次 841）。
- 7 次运行两两比较共 21 对，只有关思考的 basic 对 cot + 思考的重复运行 p < 0.05（17 对 6，p = 0.035）；同一设置的另一次运行对 basic 不显著。比较了 21 对，出现一个 p < 0.05 并不奇怪，不能当作显著。
- **同一设置重跑也会变**：思考模式的采样关不掉，cot + 思考两次运行有 6 条预测不同。除了被截断的 idx 830，另外 5 条是重新采样带来的变化（4096 那次多对 5 条，p = 0.062）。所以单次运行相差几条、0.5 个百分点左右，都在噪声之内。
- **思考很短**：DeepSeek 思考内容的中位数是 238–385 个字符（大约 60–90 个 token），最长的有 1.5 万–2 万字符。和 Qwen 小模型相比，几乎不会写到上限。
- **共同的错例**：24 条在 6 种设置（不含重复运行）下全都判错，其中 20 条的标签是正面。例如 112 “hilariously inept and ridiculous .”、230 “reign of fire looks as if it was made without much thought -- and is best watched that way .” 的标签是正面，377 “nothing is sacred in this gut-buster .” 的标签是负面。这类句子单看很难判对，换模板、开思考都没有改变结果。
- **开思考改对和改错的例子**（basic）：改对 13 条，如 667 “good film , but very glum .”、143 “a solid film ... but more conscientious than it is truly stirring .”、683 “true tale of courage ... is a harrowing drama ...”；改错 5 条，如 675 “slick piece of cross-promotion .”（标签正面）、95 “this riveting world war ii moral suspense story deals with the shadow side of american culture ...”（标签负面）。
- **和 Qwen 4b 在同样的前 200 条上比**（都用 cot、关思考）：DeepSeek 98.0%，Qwen 94.5%，不一致的 7 条全是 DeepSeek 对（p = 0.016）。
- token 合计（5 种新组合的全量运行，加上 cot + 思考的重复运行）：输入 765,940，输出 611,392。

## 结果五：DeBERTa-v3-base 微调对照组

同一份 872 条验证集，3 个种子（13 / 21 / 42）。少样本组每个种子抽到的训练句子不同，正负各半。

| 训练条数 | 各种子准确率 | 均值 ± 样本标准差 | AUC 均值 | 单次训练用时（T4） |
|---:|---|---:|---:|---:|
| 16 | 62.16% / 64.79% / 57.91% | 61.62% ± 3.47% | 0.823 | 约 2.8 分钟 |
| 64 | 85.67% / 85.32% / 83.37% | 84.79% ± 1.24% | 0.928 | 约 2.6 分钟 |
| 256 | 90.71% / 88.76% / 89.79% | 89.76% ± 0.98% | 0.960 | 约 2.6 分钟 |
| 1,024 | 93.00% / 92.89% / 93.00% | 92.97% ± 0.07% | 0.976 | 约 2.6 分钟 |
| 全部 67,349 | 95.64% / 95.53% / 95.64% | 95.60% ± 0.07% | 0.989 | 约 19 分钟 |

- **训练数据越多越准，也越稳定**：16 条时三个种子相差近 7 个百分点；1024 条以上，种子之间只差 1 条。
- **16 条时几乎没学会**：种子 13 把 872 条里的 736 条判成正面，种子 42 只判了 85 条正面，模型基本是在猜一个方向。少样本组都训练了 1000 步（16 条数据要反复看约 500 遍）。16 条时学习曲线也不稳定：种子 42 在第 600 步还有 74.9%，最后一步掉到 57.9%；本表报告最后一步，没有用验证集挑最好的一步（那样等于偷看了评测数据）。64 条以上的曲线比较平稳。
- **和指令学习比**：
  - 全量微调的 DeBERTa（95.60%）和 DeepSeek basic 关思考（95.18%）没有显著差别（三个种子 p = 0.67–0.77），比 Qwen 4b（93.81%）显著更高（p = 0.03）。
  - 只用 1024 条的 DeBERTa（92.97%）和 Qwen 4b 没有显著差别（p = 0.35–0.46）。粗略地说，在这个任务上，直接问 Qwen 4b 的效果，相当于拿一千条左右的标注数据去微调 DeBERTa-v3-base（主干约 8600 万参数，加上词嵌入约 1.8 亿）。
  - 参照 LoRA 论文表 2 同一验证集上的微调结果：RoBERTa-base 全量 94.8、LoRA 95.1，RoBERTa-large LoRA 96.2，DeBERTa-XXL LoRA 96.9（5 个种子的中位数，取最佳 epoch；见[实验结果汇总](../实验结果汇总.md) 1.3 节）。本组只用最后一步、不挑 epoch，统计方式不同。
- **都判错的句子**：DeepSeek、Qwen 4b、Qwen 9b（basic 关思考）和全量 DeBERTa（种子 13）都错的有 13 条，其中 10 条标签是正面。有几条单看句子更像负面，例如 271 “as unseemly as its title suggests .”、282 “... this jerry bruckheimer production has little else to offer”、699 “... routine , harmless diversion and little else .”，标签可能本身有争议。
- 训练过程没有出现 NaN（fp16 的非有限 loss 计数全为 0），显存峰值 3.5–4.4 GB。15 次运行共 89.5 分钟。

## 解析规则

评测脚本（[code/sst2_instruction_eval.py](code/sst2_instruction_eval.py) 的 `parse_label`、`line_label`）只从回答里取明确的答案：

- 整段只有一个数字、第一行只有一个数字（后面的行不能改成另一个答案），或 `\boxed{1}`。
- 最后一行是明确的作答：答案标记（answer、response、sentiment polarity、value、number、conclusion 等，可带 “of the text”“for this review”）加连接词（is、is:、should be、of、:、=）加 ±1 或 positive / negative，如 “The answer is: **1**”“Response: 1”“the corresponding number is -1”。
- 末行是 “1:”“Return `1`.”“1 (Positive)”“-1 (Negative sentiment)”；括号里的词必须和数字一致，`1: -1` 这样前后矛盾的算无效。单独成行的 `<final_answer>` 这类标签忽略。
- 明确作答后紧跟一句同向的说明也算，如 “The sentiment polarity of the given text is -1, indicating a negative sentiment.”，说明里的词必须和数字一致；`$\boxed{1}$` 按 `\boxed{1}` 算。这两种写法和上一条的 “(Negative sentiment)” 出现在 Llama、Mistral 的回答里，用这一版规则（`c3c96be1…`）重新判分，Qwen 和 DeepSeek 的 31 组结果一条都没有变。
- 不从推理文字中猜答案：回答 `0`、`-2`，被截断没有答案，只是复述规则（“-1 represents negative sentiment.”），或最后一行在答案后面又接着写解释（“… is -1, indicating a negative sentiment. The reviewer …”），都算无效。

本文所有数字都用这套规则，从 `metrics/raw/` 的原始返回重新判分（[code/rescore_offline.py](code/rescore_offline.py)）。和运行时较早版本的规则相比，变化大的几组：

| 组 | 运行时的规则 | 本文的规则 |
|---|---:|---:|
| Qwen 4b cot 关（200 条） | 72.0%（无效 49，多为 “Response: 1”“1:”） | 94.5%（无效 0） |
| Qwen 4b role_cot 关（200 条） | 70.5%（无效 43） | 90.0%（无效 1） |
| DeepSeek cot 关（872 条） | 95.53%（5 条 “The answer is: **1**” 这类） | 96.10% |

重新判分时，原来已经解析出的预测一条都没有改变。

## 缺少的数据

- 9b 的 cot 开思考和 role_cot 开思考缺 16 条和 32 条。
- 材料要求的 Llama、Gemma、Phi-4、Mistral 正在用 [sst2_more_models.ipynb](notebooks/sst2_more_models.ipynb) 运行，结果还没有整理进本文。
- 开思考只用了 4096 个 token 的输出上限，没有更大上限的对照，思考本身的作用和截断没有完全分开。
- 每种设置只运行了一次（DeBERTa 除外）；开思考时采样有随机性，几个百分点以内的差别没有重复运行来确认。

## 来源

- 实验：
  - 结果一的两组由 Pan-Yan-0 运行：DeepSeek 在本机，Qwen 在 Kaggle。
  - 结果二、三、五由 Pan-Yan-0 在 Kaggle 上运行 [sst2_remaining_experiments.ipynb](notebooks/sst2_remaining_experiments.ipynb)（两张 T4，共 9 小时），导出的结果包里核对了脚本（`187d2f88…`、`fb581c22…`）和数据（`a7a3bfee…`）的 SHA-256。
  - 结果三的剩余部分由 Pan-Yan-0 在 Kaggle 上运行 [sst2_finish_experiments.ipynb](notebooks/sst2_finish_experiments.ipynb)（两张 T4，共 11.2 小时），结果包里核对了脚本（`301d0bb3…`、`beb4cc0b…`）和数据的 SHA-256；接着跑的 7 组，第一次已有的条目与仓库里的原始返回逐条一致。
  - 结果四的 DeepSeek 组合由 GPT6astra 在 Pan-Yan-0 的本机运行，用的是同一版评测脚本和同一份数据，终端记录见 [metrics/deepseek_matrix_logs/](metrics/deepseek_matrix_logs/)。
- 代码：
  - 评测脚本初版由 Claude 编写，GPT6.1sol 加入了实验指纹、更严格的答案解析和断点修复；解析规则的修正由 Claude 完成。
  - 结果一的 Kaggle Notebook 由 GPT6.1sol 编写，Claude 补了 `apt-get update` 报错时的容错。
  - `sst2_remaining_experiments.ipynb`、`sst2_finish_experiments.ipynb`、`deberta_sst2.py`、`rescore_offline.py`、`sst2_report_stats.py` 由 Claude 编写。
- DeBERTa-v3-base 的参数量（12 层、隐藏维度 768、主干 8600 万、词嵌入 9800 万）取自模型说明：[microsoft/deberta-v3-base（Azure AI 模型目录）](https://ai.azure.com/catalog/models/microsoft-deberta-v3-base)。
- 本文：由 Claude 整理，参考了 DeepSeek 实验记录；所有准确率、区间、召回率、配对检验和 DeBERTa 统计都可以用 `python code/sst2_report_stats.py` 从 `metrics/` 中的逐条结果复算。
