# Qwen3-8B v5 重建队列

## 结果

本包是原 v4 源码遗失后的新实现，供 Claude 审阅和 A10 验收。上游 `upstream_qwen.py` 保持逐字节不变，SHA-256 为 `502e638814ba2b7a840f33aa3b481de1f1e0bf2887ff7d976dd86131d3b603b0`。这是已审阅的 Qwen methods v2 核心，不是声称找回了 v4。

已完成 CPU 单元测试、Python 编译检查和 shell 语法检查；尚未运行真实模型、GPU SMOKE、4B兼容性检查或正式15项。测试报告见 `validation.txt`。

冻结设计：英语 NLI，Qwen/Qwen3-8B 快照 b968826d9c46dd6066d109eabc6255188de91218，关闭思考；A10 bf16；CE；LoRA r16/alpha32/七投影/dropout0；微批8×累积2；最多3轮，原方向内层验证WF1选轮；五折×seed42–44，先同种子的五折。原方向和对称化指标均报告，dev不选轮。上游 warmup 参数原样保留，没有借重建改变训练配方。

## 问题与审阅重点

- 原 v4 指纹 ce340cba… 的通过记录不适用于本版。上传的旧结果包实际为 v3 fb0875af…，在模拟中断后的再次加载处 OOM。v5 中断退出码75，恢复由新子进程执行，不在仍持有异常栈的进程里重建模型。
- `compat.py` 比较的是相同上游默认4B配置与显式设置4B参数的适配入口，并非联合队列 `joint_worker.py` 的端到端复现。需已有4B LoRA目录以及历史零样本 `cache.jsonl`；没有这些输入就不能完成该检查，不能伪造通过。
- 兼容性检查固定同一初始adapter、16条输入、三次更新，math SDPA确定性模式；逐次保存参数并要求逐位相同，同时保存编码输入、标签、目标token和ForwardCheck。严格比较可能因环境/算子而失败，失败时检查原因，不放宽门槛冒充通过。
- v5 使用固定历史官方数据的16个文件哈希。官方数据ZIP须完整解压；不自动从GitHub拉最新版。
- 模型只经 ModelScope 下载并按固定HF快照核验；不回退下载模型到HF。获取快照元数据仍需HF_ENDPOINT可访问。包版本由setup固定；实际运行环境完整绑定。同一root不允许改变环境、代码、微批、数据。
- 最长输入探测独立运行最长16条fit样本、完整内层验证；另做四类覆盖SMOKE及损坏检查点/恢复检查。8×2失败时先检查是否真实OOM；只有OOM才另开root尝试4×4。本版不自动回退，防止其他错误被当作显存不足。
- CPU检查不能代替真实Trainer兼容性、恢复等价、显存、速度验证。当前10小时/70计算时只来自旧v4估时。
- 兼容性中途失败的旧目录保留；重新验收使用新root，避免混用半成品。普通训练/SMOKE可同root续跑。SMOKE最终回执还会复查文件哈希。
- OOF输出当前供后续配对统计与registry登记使用，本包不生成8B优于4B的结论；家族bootstrap、逐类分析沿用仓库后续分析流程。

## 下一步计划与运行入口

先让 Claude 审查本包，再运行GPU验收。以下命令均在解压后的代码目录执行；DATA是用户解压的官方仓库目录，ROOT是新的工作目录。不要把v3/v4结果覆盖进ROOT。

```bash
bash run_a10.sh setup
export ROOT=/mnt/workspace/dico_qwen8b_v5
export DATA=/mnt/workspace/SemEval-2027-Task-2-DiCo-NLI-main
bash run_a10.sh plan --root "$ROOT" --data "$DATA"
bash run_a10.sh download --root "$ROOT" --data "$DATA"
bash run_a10.sh smoke --root "$ROOT" --data "$DATA"
bash run_a10.sh compatibility --root "$ROOT" --data "$DATA" \
  --adapter /path/to/existing_4b_adapter \
  --baseline-cache /path/to/historical_zero_cache.jsonl
```

`--adapter` 目录需包括 `adapter_model.safetensors`；使用已有4B LoRA，不用8B权重。历史缓存需覆盖原英语dev全部所需有序输入，不能拿训练后概率充当零样本。两项路径为占位符，上传输入后替换。

所有验收通过后，用户手动执行：

```bash
bash run_a10.sh formal --root "$ROOT" --data "$DATA" --confirm-formal
# 单独重做汇总或打包：
bash run_a10.sh collect --root "$ROOT" --data "$DATA"
bash run_a10.sh pack --root "$ROOT" --data "$DATA"
```

正式入口会复查SMOKE产物、兼容性回执、环境和代码身份；每项新进程，完成任务核对全部记录与权重哈希后跳过；未完成任务利用Trainer检查点续跑；缺失选中权重明确失败，不悄悄重训。`pred_fold0_*.csv`是上游固定文件名，在v5各fold独立目录中实际表示该折holdout；done中统一改名holdout。

结果在ROOT/download：

- `a1_qwen8b_v5_results.zip`：日志、预测、指标、身份及源码快照，发给Claude/GPT。
- `a1_qwen8b_v5_weights.zip`：LoRA，自己保存；结果与权重须共同恢复。
- `a1_qwen8b_v5_resume.zip`：仅未完成任务的检查点；跨实例续跑未完成任务时额外保存。已完成训练的临时检查点上游会清除。

三个包都带源码和逐文件哈希，避免再次丢失运行版本。恢复时合并解压到新ROOT，重新执行download核验模型；若原运行环境不同，不会静默继续。不要上传底座模型；从已核验源重新下载。

## 文件与来源

`common.py`：身份与完整性；`worker.py`：单任务及跨进程恢复；`compat.py`：4B一致性；`a1_queue.py`：队列、门槛、OOF、打包；`run_a10.sh`：环境入口；`test_queue.py`：CPU负例；`expected_data.json`：原验收记录的数据哈希。

重建实现由GPT编写；冻结方案来自仓库《更大模型与数据扩充计划》。Claude此前审过v1/v2，不代表已审过本v5。旧SMOKE由Pan-Yan-0运行，本版尚待实测。

### 一次上传准备

本交接ZIP不含官方数据和私人4B权重。可在本机先合成一个上传包：

```bash
python prepare_upload.py --official-zip /path/to/SemEval-2027-Task-2-DiCo-NLI-main.zip --adapter /path/to/existing_4b_adapter --baseline-cache /path/to/zero_cache.jsonl --output a10_upload_v5.zip
```

上传后解压 `a10_upload_v5.zip`，再解压其中 `official.zip`，DATA指向含final_data的目录；adapter和cache使用inputs里的路径。该上传包含4B LoRA，只用于自己的实验实例，不提交公开仓库。
