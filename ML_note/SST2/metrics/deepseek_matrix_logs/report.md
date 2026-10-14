| 后端 | 模型 | 模板 | 思考 | 数据 | 生成设置 | 样本数 | 准确率 | 无效 | 平均秒/条 | 输出 tokens | 解析统计 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| deepseek | deepseek-flash | basic | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=64 | 20 | 0.9500 | 0 | 1.35 | 31 | exact=20 |
| deepseek | deepseek-flash | basic | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=64 | 872 | 0.9518 | 0 | 1.00 | 1,328 | exact=872 |
| deepseek | deepseek-flash | basic | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 20 | 1.0000 | 0 | 1.39 | 1,090 | exact=20 |
| deepseek | deepseek-flash | basic | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 872 | 0.9610 | 0 | 1.43 | 69,993 | exact=872 |
| deepseek | deepseek-flash | cot | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=1024 | 20 | 1.0000 | 0 | 1.77 | 2,255 | last_number=20 |
| deepseek | deepseek-flash | cot | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=1024 | 872 | 0.9553 | 5 | 1.77 | 108,182 | exact=2, invalid=5, last_number=865 |
| deepseek | deepseek-flash | cot | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 20 | 0.9500 | 0 | 1.71 | 1,800 | exact=20 |
| deepseek | deepseek-flash | cot | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 872 | 0.9644 | 1 | 1.71 | 106,350 | empty=1, exact=871 |
| deepseek | deepseek-flash | cot | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=8192 | 872 | 0.9587 | 0 | 1.76 | 108,975 | exact=872 |
| deepseek | deepseek-flash | role_cot | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=1024 | 20 | 0.9500 | 0 | 1.82 | 2,269 | exact=1, last_number=19 |
| deepseek | deepseek-flash | role_cot | off | data\sst2_validation.tsv | temperature=0.0, max_tokens=1024 | 872 | 0.9553 | 0 | 1.89 | 114,690 | exact=6, last_number=866 |
| deepseek | deepseek-flash | role_cot | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 20 | 1.0000 | 0 | 1.79 | 1,963 | exact=20 |
| deepseek | deepseek-flash | role_cot | on | data\sst2_validation.tsv | temperature=0.6, max_tokens=4096 | 872 | 0.9610 | 0 | 1.75 | 103,202 | exact=872 |
