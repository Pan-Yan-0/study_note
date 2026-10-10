# -*- coding: utf-8 -*-
"""由 data/ 下的 CSV 生成附录 A、B、C 的 LaTeX 表格（数值照抄原稿附录）。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUT = ROOT / "sections"


def rows(name):
    with open(D / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(k):
    return r"\texttt{" + k.replace("_", r"\_").replace("$", r"\$") + "}"


def num(s):
    return "--" if s in ("–", "-", "") else s


# 附录 A：附件1台账
a = rows("att1_ledger.csv")
lines = [r"{\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.02}",
         r"\begin{longtable}{lrrrrrcc}",
         r"\caption{附件1全部100条样本的特征与对齐台账}\label{tab:appA}\\",
         r"\toprule",
         r"样本键（video\_id\$\_\$clip\_id） & $D_i$/s & $L_i^T$ & $L_i^A$ & $L_i^V$ & $W_i$ & 状态 & $C_i$ \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{8}{l}{表\ref{tab:appA}（续）}\\\toprule",
         r"样本键 & $D_i$/s & $L_i^T$ & $L_i^A$ & $L_i^V$ & $W_i$ & 状态 & $C_i$ \\\midrule\endhead",
         r"\bottomrule\multicolumn{8}{r}{续下页}\endfoot",
         r"\bottomrule\endlastfoot"]
for r in a:
    lines.append(" & ".join([key(r["key"]), r["D"], r["LT"], num(r["LA"]), num(r["LV"]), r["W"], r["state"], r["C"]]) + r" \\")
lines += [r"\end{longtable}}"]
(OUT / "app_A_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 附录 B：附件3，两栏排
b = rows("att3.csv")
half = (len(b) + 1) // 2
lines = [r"\begin{table}[H]", r"\centering",
         r"\caption{附件3的30条全量预测（强度为 $[-3,3]$ 尺度，$p_{\max}$ 为三类概率的最大值）}\label{tab:appB}",
         r"\begin{tabular}{crrr@{\hspace{2.2em}}crrr}", r"\toprule",
         r"编号 & 极性 & 强度 & $p_{\max}$ & 编号 & 极性 & 强度 & $p_{\max}$ \\", r"\midrule"]
fmt = lambda v: ("$" + ("-" if float(v) < 0 else "") + f"{abs(float(v)):.3f}" + "$")
for i in range(half):
    L = b[i]; R = b[i + half] if i + half < len(b) else None
    cells = [L["id"], L["pol"], fmt(L["y"]), L["pmax"]]
    cells += [R["id"], R["pol"], fmt(R["y"]), R["pmax"]] if R else ["", "", "", ""]
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "app_B_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 附录 C：附件4
c = rows("att4.csv")
lines = [r"\begin{table}[H]", r"\centering",
         r"\caption{附件4全部20条的预测与时间核验状态（T、A、V 分别为文本、语音、视觉）}\label{tab:appC}",
         r"\begin{tabular}{cccccccc}", r"\toprule",
         r"样本 & 预测类别 & 强度 & 类别概率 & 分类主导 & 强度主导 & 时间状态 \\", r"\midrule"]
lines[3] = r"\begin{tabular}{ccrcccc}"
sgn = lambda v: ("$" + ("+" if float(v) >= 0 else "-") + f"{abs(float(v)):.3f}" + "$")
for r in c:
    lines.append(" & ".join([r["id"], r["pol"], sgn(r["y"]), r["p"], r["dom_cls"], r["dom_reg"], r["time"]]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "app_C_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("ok", len(a), len(b), len(c))
