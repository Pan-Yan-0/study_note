# SST-2 指令学习实验

不训练模型，把影评填进一段指令，让大模型直接回答 -1（负面）或 1（正面），在 GLUE SST-2 验证集全部 872 条上统计准确率。任务来自导师的《1-使用大模型进行指令学习》材料，对照组是在同一验证集上评测的 DeBERTa 微调（尚未运行）。

## 文件

| 文件 | 内容 |
|---|---|
| [code/sst2_instruction_eval.py](code/sst2_instruction_eval.py) | 评测脚本：DeepSeek API、Ollama 和其他 OpenAI 兼容服务通用；三种指令模板（basic / cot / role_cot），断点续跑，导出逐条 CSV 与汇总 JSON |
| [notebooks/teacher_sst2_ollama.ipynb](notebooks/teacher_sst2_ollama.ipynb) | Kaggle 上安装 Ollama 0.40.3、下载 `qwen3.5:4b` 并调用上面脚本的 Notebook，内置同一份脚本和 872 条数据（这一版在 `apt-get update` 报错时会继续安装 `zstd`，其余与运行的版本相同） |
| [metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_summary.json](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_summary.json) | Qwen3.5 4B 全量汇总：准确率、混淆矩阵、token 数、实验指纹 |
| [metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd.csv](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd.csv) | 872 条逐条结果：标签、预测、解析方式、耗时、token 数、原句 |
| [metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_errors.csv](metrics/ollama_qwen3.5-4b_basic_think-off_all_052bf7f44efd_errors.csv) | 其中的 54 条错例 |
| [metrics/kaggle_env/](metrics/kaggle_env/) | Kaggle 运行环境：`nvidia-smi`、`ollama ps`、Ollama 服务版本 |

DeepSeek 的逐条结果在本机实验目录，没有放进仓库；下面的 DeepSeek 数字取自本机的实验记录。

## 设置

| 项目 | 内容 |
|---|---|
| 数据 | GLUE SST-2 验证集 872 条（正面 444、负面 428），标签 0/1 换成 -1/1。test 集标签不公开，算不了准确率。两组实验用同一份文件（SHA-256 `a7a3bfee…`） |
| 指令 | `basic` 模板（材料里发给 ChatGPT 的版本），Alpaca 格式，要求只返回一个数字；作为单轮用户消息发送，末尾不加 `<think>` |
| 生成 | 关闭思考，temperature 0，max_tokens 64 |
| 判分 | 从回答中解析 -1 / 1，解析不出按错误计；两组共 1,744 条回答全部是单个数字 |

指令原文（`{影评}` 处填入句子）：

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the given text from an online review and determine the sentiment polarity. Return a single number of either -1 and 1, with 1 being positive and -1 being the negative sentiment.
No further explanation or justification is required.

### Question:
{影评} What's the sentiment polarity of the text?

### Response:
```

## 结果

| 模型 | 运行方式 | 正确 / 872 | 准确率 [95% 区间] | 负面召回 | 正面召回 | 单条请求耗时 |
|---|---|---:|---|---:|---:|---:|
| `deepseek-flash` | DeepSeek 官方 API，8 路并发 | 830 | 95.18% [93.55, 96.42] | 98.36%（421/428） | 92.12%（409/444） | 1.00 秒 |
| `qwen3.5:4b`（Q4_K_M） | Ollama 0.40.3，Kaggle T4，单路 | 818 | 93.81% [92.01, 95.22] | 95.33%（408/428） | 92.34%（410/444） | 0.19 秒 |

| 真实标签 | DeepSeek 预测 -1 | DeepSeek 预测 1 | Qwen 预测 -1 | Qwen 预测 1 |
|---|---:|---:|---:|---:|
| 负面 -1 | 421 | 7 | 408 | 20 |
| 正面 1 | 35 | 409 | 34 | 410 |

- 区间为 Wilson 95% 区间，两者重叠。DeepSeek 多对 12 条（1.38 个百分点）；差距是否显著要用两边的逐条结果做配对检验（McNemar），需要先补上 DeepSeek 的逐条 CSV。
- 两个模型的错误都以正面判成负面为主（35 条、34 条）。差别在负面判成正面：DeepSeek 7 条，Qwen 20 条。
- token：DeepSeek 输入 104,380、输出 1,328；Qwen 输入 116,559、输出 2,186（每条回答 2–3 个 token）。两者的分词器和对话模板不同，输入 token 数不能直接比较。
- 参照：LoRA 论文表 2 在同一个 SST-2 验证集上报告的微调准确率为 RoBERTa-base 全量 94.8、LoRA 95.1，RoBERTa-large LoRA 96.2，DeBERTa-XXL LoRA 96.9（5 个种子的中位数，取最佳 epoch；见[实验结果汇总](../实验结果汇总.md) 1.3 节）。训练条件和统计方式不同，只能看量级。

## 错例

下面只描述句子本身。关闭思考时模型只返回标签，不能据此断言模型内部的出错原因。

DeepSeek 实验记录里列出的 5 条错例，Qwen 错了其中 3 条：

| idx | 真实 | DeepSeek | Qwen | 句子 |
|---:|---:|---:|---:|---|
| 667 | 1 | -1 | -1 | good film , but very glum . |
| 112 | 1 | -1 | -1 | hilariously inept and ridiculous . |
| 791 | -1 | 1 | 1 | it 's somewhat clumsy and too lethargically paced -- but its story … offers a solid build-up , a terrific climax , and some nice chills along the way . |
| 706 | -1 | 1 | -1 | how do you spell cliché ? |
| 741 | 1 | -1 | 1 | but it still jingles in the pocket . |

Qwen 把负面判成正面的 20 条里，多数句子没有直接的负面词，贬义靠反讽或比较表达，例如：

- 171：rarely has leukemia looked so shimmering and benign .
- 183：the lower your expectations , the more you 'll enjoy it .
- 312：i 'll bet the video game is a lot more fun than the film .
- 724：it 's everything you do n't go to the movies for .
- 695：sit through this one , and you wo n't need a magic watch to stop time ; your dvd player will do it for you .

把正面判成负面的 34 条多是褒贬并列或很短的片段，其中有些句子单看很难判断为正面：

- 481：funny but perilously slight .
- 143：a solid film ... but more conscientious than it is truly stirring .
- 289：cool ?
- 195：my thoughts were focused on the characters .
- 273：minority report is exactly what the title indicates , a report .

## Kaggle 运行环境

- 2 张 Tesla T4（驱动 580.178.04，CUDA 13.0）。模型只放在第一张卡上，33/33 层全部在 GPU，占显存 3.1 GB，上下文长度 4096。
- Ollama 0.40.3 启动了一个 llama.cpp 的 `llama-server` 子进程来运行模型：`ollama ps` 的 RUNNER 列为 `llamacpp`，`nvidia-smi` 里的进程是 `.../lib/ollama/llama-server`。
- 模型 `qwen3.5:4b`：digest `d8b0f5e9760c…`，Q4_K_M 量化，4,205,751,296 个参数，32 层，隐藏维度 2560，原生上下文 262,144。
- 第一次请求需要把模型载入显存，用了 108.9 秒；之后每条 0.16–0.24 秒（平均 0.19 秒），872 条约 2.8 分钟。

## 尚缺

- DeepSeek 的逐条 CSV、原始响应 JSONL 与汇总 JSON（本机实验目录中文件名带 `a4d72f5968bf` 的几份）。
- DeBERTa-base 对照组：全量训练集，以及只用 16 / 64 / 256 / 1024 条，各 3 个种子，在同一份 872 条上评测。
- 材料要求的其他对比：开启思考、cot / role_cot 模板、其他尺寸（`qwen3.5:0.8b` / `2b` / `9b`）。

## 来源

`metrics/` 中的文件由 Kaggle Notebook 导出；DeepSeek 的数字、错例与 token 数取自本机实验记录。区间、召回率由这些数字计算。本文由 Claude 整理。
