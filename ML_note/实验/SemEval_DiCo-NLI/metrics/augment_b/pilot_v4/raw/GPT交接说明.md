# GPT → Claude：AugB v4 15 批试点原文交接

日期：2026-11-05。冻结输入为 augb-pilot-v4，依据 Claude_Go.md 放行生成。

## 执行结果

- pilot_b01–pilot_b14：首次格式合格。
- pilot_b15：首次 E2/F8/O4，check_format.py 判 format_fail；全新 Luna 同一提示词重试一次后 E3/F7/O4，合格。
- 共 16 次尝试：15 ok、1 format_fail、0 infra_error。全部原文保留，没有覆盖、改写、语义筛选或因语义重试。
- 合格批次共 210 对（E45/F105/O60）；这不是过滤后的数据量，也不是审核通过的数据量。

## 调度与隔离

每次尝试调用一个全新子 agent，fork_turns=none，requested_model=gpt-6-luna；message 为相应批次 prompt 全文，不附加评价、重试反馈或其他批次内容。生成器禁止读取文件/联网。
逐次发起调用，允许尚未返回的批次与下一批重叠；按父 agent 收到最终回复的记录，同时在途最多 2 个。工具返回规范任务名，已记入 agent_id。工具没有回显实际模型标识，visible_model 如实填“工具未返回”；不能将请求模型等同于经独立验证的实际模型。

## 日志时间限制（须保留）

start 是调用前已记录的排队时间，end 是收到回复后写入原文时的时间，不是服务端精确开始/完成时间。b02/b03 没有单独采样调用前时刻，使用 b01 调度前的公共下界；b04–b15 使用提前逐项记录的排队时刻。b15 重试有单独调用前时刻。日志 timing_note 明确这一点。它们可证明记录区间的顺序和边界，但不得据此估算单批模型耗时、吞吐或并发峰值。原文按最终消息可见文本直接保存为 UTF-8，无附加末尾换行，未重新序列化 JSON。

## 检查范围

已核对源 MANIFEST 中的全部文件哈希、15 个 prompt 哈希、16 份原文哈希、格式/配额和日志对应关系。本轮未重复运行上一轮已通过的 CPU 测试，未调用 process_outputs.py build，未做主审核、过滤或 GPU 训练。

## 请 Claude 接手

使用冻结 v4 包与本目录调用 process_outputs.py build，检查全部停止条件；随后按 P1 准备 60 对盲审。不要将本轮格式通过视为 ready_for_training。
按放行交接预先确定的口径，报告 number_pair（按生成标签）及单向对 text2 词为 text1 子集/只删一词的比例；仅报告，不新增过滤或停止条件。数量规则来源声明保持不变。
为避免影响 Pan-Yan-0 的独立盲审，本交接不附逐条语义裁决。原始失败尝试仅留档，不可作为候选补充。

## 文件

outputs/<batch>/attempt_N.txt：原始最终回复；check_N.json：冻结检查器输出。
dispatch_log.jsonl：完整 16 条尝试日志。
batches/：冻结批次原文件；SOURCE_MANIFEST.json：源包清单；SHA256SUMS.json：本交接完整性清单。
