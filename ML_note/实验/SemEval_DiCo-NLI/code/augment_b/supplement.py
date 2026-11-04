"""全量阶段的补充批次规划（确定性）。过滤后某折某类可用行数不够 +1 需求时，用它决定补哪些签名、补多少批。

规则（按组 E → 单向 → O 依次处理）：
  1. 缺口 D[k] = 需求行数 − 该折该类已保留的可用行数（≤ 0 表示够了）。
  2. 每次选一个签名 s：覆盖的缺口折数最多者优先；相同时家族数多者优先；再相同按签名字符串升序。
  3. 给 s 下一个只含该组的补充批次，每批 PAIRS_PER_BATCH 对；预计保留行数 = 对数 × 每对行数（E、单向 2，O 1）
     × keep_rate（该组在已完成批次里的实际保留率，向下取到 0.05 的倍数，事先记录）。s 覆盖的每一折缺口都减去这个数。
  4. 重复到所有缺口 ≤ 0。补充总行数超过 +1 计划行数的 BUDGET 比例就停下，报告"预算不足"，等人决定。
  5. 父样本：在该签名里按固定种子打乱家族，先用没用过的，用完再从头循环；每批 PARENTS_PER_BATCH 个。
  补充批次用新的批次号，生成和处理规则不变；最多补两轮，第二轮后仍有缺口就停下报告。
"""
import math, random

PAIRS_PER_BATCH, PARENTS_PER_BATCH, BUDGET, SEED = 14, 13, 0.20, 20261104
ROWS_PER_PAIR = {"E": 2, "U": 2, "O": 1}
LABEL_OF = {"E": "EQUIVALENCE", "U": "FORWARD_ENTAILMENT", "O": "NEGATIVE_OTHER"}


def plan(deficit, sig_families, keep_rate, planned_rows, used=None, round_no=1):
    """deficit: {(fold, group): 行数}；sig_families: {"0134": [家族, ...]}；keep_rate: {group: 比例}；
    planned_rows: +1 计划总行数；used: {签名: 已用家族集合}。返回 (批次列表, 状态)。"""
    D = dict(deficit); used = {s: set(v) for s, v in (used or {}).items()}
    batches, total = [], 0
    for g in "EUO":
        kr = math.floor(keep_rate[g] * 20) / 20
        if kr <= 0: return batches, f"{g} 组保留率为 0，无法规划"
        per = PAIRS_PER_BATCH * ROWS_PER_PAIR[g] * kr
        while any(D.get((k, g), 0) > 0 for k in range(5)):
            need = {k for k in range(5) if D.get((k, g), 0) > 0}
            s = min(sig_families, key=lambda s: (-len(need & {int(c) for c in s}), -len(sig_families[s]), s))
            if not need & {int(c) for c in s}: return batches, f"{g} 组的缺口折 {sorted(need)} 没有可用签名"
            for k in (int(c) for c in s): D[(k, g)] = D.get((k, g), 0) - per
            total += PAIRS_PER_BATCH * ROWS_PER_PAIR[g]
            if total > BUDGET * planned_rows: return batches, "预算不足：补充行数超过计划的 20%，停下等人决定"
            fams = sorted(sig_families[s]); random.Random(f"{SEED}-{s}").shuffle(fams)
            u = used.setdefault(s, set())
            fresh = [f for f in fams if f not in u]
            pick = (fresh + [f for f in fams if f in u])[:PARENTS_PER_BATCH]
            u.update(pick)
            batches.append({"batch_id": f"supp{round_no}_b{len(batches) + 1:03d}", "signature": s, "group": g,
                            "quota_pairs": {LABEL_OF[g]: PAIRS_PER_BATCH}, "parent_families": pick})
    return batches, "ok"
