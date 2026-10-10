# -*- coding: utf-8 -*-
"""
由问题一三种方案的 .npz 特征文件生成附录中的 100 条样本汇总表（results/q1_summary.csv）。

用法（在论文文件夹里运行）：
    python tools/make_q1_table.py --pkg <q1_code_package 路径> --data-root <E题数据根目录>

--data-root 用来读取原始视频时长；不给或找不到视频时，时长一列留空。
运行结束会打印各列的平均值，可以和论文表 q1eval 的数字对一下。
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

METHOD_DIRS = {
    "H": "step1_extract/results/q1_features",
    "T50": "step1_time50/results/q1_time50_features",
    "V50": "step1_vad50/results/q1_vad50_features",
}


def nonzero_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return int((np.linalg.norm(x.reshape(x.shape[0], -1), axis=1) > 1e-8).sum())


def scalar(d, key):
    if key not in d.files:
        return None
    v = np.asarray(d[key])
    return v.item() if v.size == 1 else None


def video_index(data_root):
    idx = {}
    if not data_root:
        return idx
    for p in Path(data_root).rglob("*.mp4"):
        vid, clip = p.parent.name, p.stem
        for key in (f"{vid}_{clip}", f"{vid}$_${clip}", f"{vid}__{clip}", f"{vid}-{clip}", clip):
            idx.setdefault(key, p)
    return idx


def video_duration(path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0 and n > 0:
            return n / fps
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=".", help="q1_code_package 所在路径")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    ap.add_argument("--out", default="results/q1_summary.csv")
    args = ap.parse_args()

    pkg = Path(args.pkg)
    files = {m: {p.stem: p for p in (pkg / d).glob("*.npz")} for m, d in METHOD_DIRS.items()}
    for m, fs in files.items():
        print(f"方法 {m}: 找到 {len(fs)} 个 npz（{pkg / METHOD_DIRS[m]}）")
    ids = sorted(set().union(*[set(fs) for fs in files.values()]))
    vids = video_index(args.data_root)

    header = ["sample", "dur", "L_txt", "eff_s", "eff_ratio", "n_seg",
              "nzA_H", "nzV_H", "nzA_T50", "nzV_T50", "nzA_V50", "nzV_V50"]
    rows = []
    for sid in ids:
        row = {"sample": sid}
        dur = video_duration(vids[sid]) if sid in vids else None
        row["dur"] = f"{dur:.3f}" if dur else ""
        for m in ("H", "T50", "V50"):
            p = files[m].get(sid)
            if p is None:
                row[f"nzA_{m}"] = row[f"nzV_{m}"] = ""
                continue
            d = np.load(p, allow_pickle=True)
            row[f"nzA_{m}"] = nonzero_rows(d["audio"])
            row[f"nzV_{m}"] = nonzero_rows(d["vision"])
            if "L_txt" not in row and scalar(d, "L_txt") is not None:
                row["L_txt"] = int(scalar(d, "L_txt"))
            if m == "V50":
                eff = scalar(d, "effective_secs")
                nseg = scalar(d, "n_segments")
                row["eff_s"] = f"{eff:.2f}" if eff is not None else ""
                row["n_seg"] = int(nseg) if nseg is not None else ""
                row["eff_ratio"] = f"{eff / dur:.2f}" if (eff is not None and dur) else ""
        rows.append({k: row.get(k, "") for k in header})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {out}，共 {len(rows)} 行")

    for k in header[1:]:
        vals = [float(r[k]) for r in rows if r[k] not in ("", None)]
        if vals:
            print(f"  {k:>10s} 平均 {np.mean(vals):.2f}（{len(vals)} 条）")


if __name__ == "__main__":
    main()
