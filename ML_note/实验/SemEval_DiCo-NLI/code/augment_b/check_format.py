"""判断一次生成尝试是否"格式合格"（GPT 调度时用它决定要不要重试；Claude 处理时用同一函数）。
用法：python check_format.py <batches/xxx.json> <outputs/xxx/attempt_N.txt>
退出码 0 = 合格，1 = 格式不合格；标准输出是一行 JSON：{"status", "reasons", "fence_stripped", "raw_sha256"}。

格式不合格（只有这些情况可以用同一提示词、全新子 agent 重试一次）：
  1. 去掉首尾空白后不是一个完整的 JSON 对象（允许外面恰好包一层 ``` 代码块，记录 fence_stripped）；
  2. batch_id 和批次文件不同；没有 pairs 列表；
  3. 某一项缺 text1 / text2 / label，或不是字符串，或文本去空白后为空；
  4. label 不在本批允许的标签里；
  5. 各标签的对数和配额不完全相等。
重复、过长、和官方数据或父样本相同等问题不算格式错误，不触发重试，交给 process_outputs.py 确定性过滤。
空回复、超时、工具报错属于基础设施错误，由调度方另记 status=infra_error，不经过本脚本。
"""
import hashlib, json, sys
from collections import Counter


def parse(raw):
    s = raw.strip()
    fenced = False
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            s, fenced = "\n".join(lines[1:-1]).strip(), True
    return s, fenced


def check(batch, raw):
    reasons = []
    s, fenced = parse(raw)
    out = {"status": "ok", "reasons": reasons, "fence_stripped": fenced,
           "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(), "pairs": None}
    try:
        obj = json.loads(s)
    except Exception as e:
        reasons.append(f"不是完整的 JSON：{type(e).__name__}")
        obj = None
    if obj is not None:
        if not isinstance(obj, dict):
            reasons.append("顶层不是 JSON 对象")
        else:
            if obj.get("batch_id") != batch["batch_id"]:
                reasons.append("batch_id 不一致")
            pairs = obj.get("pairs")
            if not isinstance(pairs, list):
                reasons.append("缺少 pairs 列表")
            else:
                for i, p in enumerate(pairs):
                    if not isinstance(p, dict) or any(not isinstance(p.get(k), str) for k in ("text1", "text2", "label")):
                        reasons.append(f"第 {i} 项缺字段或字段不是字符串"); continue
                    if not p["text1"].strip() or not p["text2"].strip():
                        reasons.append(f"第 {i} 项有空文本")
                    if p["label"] not in batch["allowed_labels"]:
                        reasons.append(f"第 {i} 项标签不合法：{p['label']}")
                if not reasons:
                    got = Counter(p["label"] for p in pairs)
                    if dict(got) != {k: v for k, v in batch["quota_pairs"].items() if v}:
                        reasons.append(f"配额不符：要求 {batch['quota_pairs']}，实际 {dict(got)}")
                    else:
                        out["pairs"] = [{"text1": p["text1"], "text2": p["text2"], "label": p["label"]} for p in pairs]
    if reasons:
        out["status"] = "format_fail"
    return out


if __name__ == "__main__":
    batch = json.load(open(sys.argv[1], encoding="utf-8"))
    raw = open(sys.argv[2], encoding="utf-8").read()
    r = check(batch, raw)
    print(json.dumps({k: v for k, v in r.items() if k != "pairs"}, ensure_ascii=False))
    sys.exit(0 if r["status"] == "ok" else 1)
