"""第四组西班牙语、巴斯克语（Track 2、3）的复算：从缓存重建预测、和保存的预测逐条比较、用官方脚本重新评分，
并列出每个种子的分数、逐轮内层验证 WF1 和 fold 0 对称化后的每类 F1。

用法：python3 qwen_sft_tracks23.py <结果目录 dico_qwen_sft_results> <官方仓库> <第二组 metrics/diagnostics>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_sft_analysis import rows, jsonl, predict, official, METRICS, class_f1
res, dico, diag = sys.argv[1:4]
fam = json.load(open(os.path.join(diag, "families.json")))
out = {}
mism = 0
for t in (2, 3):
    tr = rows(os.path.join(dico, f"final_data/train/dico_nli_train_track{t}_reference.csv"))
    dev = rows(os.path.join(dico, f"final_data/dev/dico_nli_dev_track{t}_reference.csv"))
    fold0 = [r for r in tr if fam["fold_of_family"][fam["family_of_pair"][r["pair_id"]]] == 0]
    parts = {"fold0": fold0, "dev": dev}
    print(f"track{t}: fold0 {len(fold0)} 行, dev {len(dev)} 行")
    for run in ["zero_shot", "seed42", "seed43", "seed44"]:
        d = os.path.join(res, f"track{t}", run)
        P = jsonl(os.path.join(d, "cache.jsonl"))
        done = json.load(open(os.path.join(d, "done.json")))
        rec = {}
        for name, rs in parts.items():
            for mode in ("plain", "sym"):
                lab = predict(P, rs, mode)
                fp = {r["instance_id"]: r["label"] for r in rows(os.path.join(d, f"pred_{name}_{mode}.csv"))}
                mism += sum(fp[r["instance_id"]] != l for r, l in zip(rs, lab))
                sc = official(dico, rs, lab)
                assert all(abs(sc[k] - done["scores"][name][mode][k]) < 1e-12 for k in METRICS), (t, run, name, mode)
                rec[f"{name}_{mode}"] = [round(sc[k], 4) for k in METRICS]
                if mode == "sym" and name == "fold0" and run != "zero_shot":
                    rec["fold0_sym_classf1"] = {k[:3]: round(v, 3) for k, v in class_f1([r["label"] for r in rs], lab).items()}
        if run != "zero_shot":
            rec["history"] = [round(h["val_weighted_f1"], 4) for h in done["history"]]
            rec["best"] = done["best_epoch"]; rec["sec_per_step"] = done.get("seconds_per_step"); rec["min"] = done.get("minutes_last_session")
            rec["train_loss"] = [round(h["train_loss"], 3) for h in done["history"]]
        out[f"track{t}/{run}"] = rec
print("重建预测与保存的不同：", mism, "；官方脚本重算全部一致")
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False))
