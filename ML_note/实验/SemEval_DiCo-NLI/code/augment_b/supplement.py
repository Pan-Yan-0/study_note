"""全量阶段的补充批次规划（确定性）。过滤后某折某类可用行数不够 +1 需求时，用它决定补哪些签名、补多少批。

规则（按组 E → 单向 → O 依次处理）：
  1. 缺口 D[k] = 需求行数 − 该折该类已保留的可用行数（≤ 0 表示够了）。单向组的需求是 F + B 的合计行数，
     最终仍要逐折逐类（E / F / B / O 四类）核对，不只核对三组总数。
  2. 每次选一个签名 s：覆盖的缺口折数最多者优先；相同时该组的家族数多者优先；再相同按签名字符串升序。
  3. 给 s 下一个只含该组的补充批次，每批 PAIRS_PER_BATCH 对；预计保留行数 = 对数 × 每对行数（E、单向 2，O 1）
     × keep_rate（该组在已完成批次里的实际保留率，向下取到 0.05 的倍数，事先记录）。s 覆盖的每一折缺口都减去这个数。
  4. 重复到所有缺口 ≤ 0。两轮合计下达的补充行数（台账里累计）超过 +1 计划行数的 BUDGET 比例就停下，报告"预算不足"。
  5. 只允许第 1、2 轮；第二轮后仍有缺口就停下报告。
  6. 父样本（v6 起按父样本仿写，一个父样本对应一对）：每个签名、每个组的家族按固定种子打乱一次，
     台账里按"签名-组"保存游标，每批从游标起取 PAIRS_PER_BATCH 个、循环取用；该组家族不够一批时少取，对数随之减少。
  台账（ledger）在两轮之间持久保存：累计行数、各签名游标、已下达的批次号。
"""
import json, math, random

PAIRS_PER_BATCH, BUDGET, SEED = 14, 0.20, 20261104
ROWS_PER_PAIR = {"E": 2, "U": 2, "O": 1}
LABEL_OF = {"E": "EQUIVALENCE", "U": "FORWARD_ENTAILMENT", "O": "NEGATIVE_OTHER"}


def new_ledger():
    return {"rows_issued": 0, "cursor": {}, "batches": [], "rounds_done": []}


def plan(deficit, sig_families, keep_rate, planned_rows, ledger, round_no):
    """deficit: {(fold, group): 行数}；sig_families: {"0134": {"E": [家族, ...], "U": [...], "O": [...]}}（家族按父样本的组分开，
    不含只差数量的父样本）；keep_rate: {group: 比例}；
    planned_rows: +1 计划总行数；ledger: 上一轮返回的台账（第 1 轮用 new_ledger()）。
    返回 (本轮批次列表, 状态, 新台账)。状态不是 "ok" 时不要下达本轮批次。"""
    if round_no not in (1, 2): raise ValueError("补充最多两轮")
    if round_no in ledger["rounds_done"] or (round_no == 2 and 1 not in ledger["rounds_done"]):
        raise ValueError("轮次顺序不对")
    led = json.loads(json.dumps(ledger))
    D, batches = dict(deficit), []
    for g in "EUO":
        kr = math.floor(keep_rate[g] * 20) / 20
        if kr <= 0: return [], f"{g} 组保留率为 0，无法规划", ledger
        cands = [s for s in sig_families if sig_families[s].get(g)]
        while any(D.get((k, g), 0) > 0 for k in range(5)):
            need = {k for k in range(5) if D.get((k, g), 0) > 0}
            if not cands: return [], f"{g} 组没有可用的父样本", ledger
            s = min(cands, key=lambda s: (-len(need & {int(c) for c in s}), -len(sig_families[s][g]), s))
            if not need & {int(c) for c in s}: return [], f"{g} 组的缺口折 {sorted(need)} 没有可用签名", ledger
            fams = sorted(sig_families[s][g]); random.Random(f"{SEED}-{s}-{g}").shuffle(fams)
            key = f"{s}-{g}"; c = led["cursor"].get(key, 0)
            pick = [fams[(c + i) % len(fams)] for i in range(min(PAIRS_PER_BATCH, len(fams)))]
            led["cursor"][key] = (c + len(pick)) % len(fams)
            for k in (int(c) for c in s): D[(k, g)] = D.get((k, g), 0) - len(pick) * ROWS_PER_PAIR[g] * kr
            led["rows_issued"] += len(pick) * ROWS_PER_PAIR[g]
            if led["rows_issued"] > BUDGET * planned_rows:
                return [], f"预算不足：两轮累计补充 {led['rows_issued']} 行，超过计划的 {BUDGET:.0%}，停下等人决定", ledger
            bid = f"supp{round_no}_b{len(batches) + 1:03d}"
            batches.append({"batch_id": bid, "signature": s, "group": g, "quota_pairs": {LABEL_OF[g]: len(pick)},
                            "parent_families": pick})
            led["batches"].append(bid)
    led["rounds_done"].append(round_no)
    return batches, "ok", led
