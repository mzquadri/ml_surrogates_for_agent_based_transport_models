"""Replace [htbp] / [ht] / [tbp] / [t] with [H] in Appendix A only."""
import re
p = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document/chapters/appendix_a_master_table.tex"
s = open(p, encoding="utf-8").read()
n_before = sum(s.count(f"\\begin{{table}}[{x}]") for x in ["htbp", "ht", "tbp", "t", "h"])
s = re.sub(r"\\begin\{table\}\[(htbp|ht|tbp|t|h)\]", r"\\begin{table}[H]", s)
n_after = s.count("\\begin{table}[H]")
open(p, "w", encoding="utf-8").write(s)
print(f"Converted {n_after} table floats to [H] placement")
