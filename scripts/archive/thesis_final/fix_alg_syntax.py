"""Convert algpseudocode -> algorithmic syntax in chapters 3 and 4."""
import os, re

CHAPTERS = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document/chapters"

def fix(text):
    # algorithmic uses uppercase keywords and no [1] in begin
    text = text.replace(r"\begin{algorithmic}[1]", r"\begin{algorithmic}")
    text = text.replace(r"\Require ", r"\REQUIRE ")
    text = text.replace(r"\Ensure ",  r"\ENSURE ")
    text = text.replace(r"\State ",   r"\STATE ")
    text = text.replace(r"\Return ",  r"\RETURN ")
    text = re.sub(r"\\For\{",   r"\\FOR{",   text)
    text = text.replace(r"\EndFor", r"\ENDFOR")
    text = re.sub(r"\\If\{",    r"\\IF{",    text)
    text = text.replace(r"\EndIf", r"\ENDIF")
    text = re.sub(r"\\While\{", r"\\WHILE{", text)
    text = text.replace(r"\EndWhile", r"\ENDWHILE")
    text = re.sub(r"\\Comment\{", r"\\COMMENT{", text)
    return text

for name in ["03_methodology.tex", "04_experiments.tex"]:
    p = os.path.join(CHAPTERS, name)
    s = open(p, encoding="utf-8").read()
    s2 = fix(s)
    open(p, "w", encoding="utf-8").write(s2)
    print(f"  fixed: {name}  ({len(s)}->{len(s2)} chars)")
print("done")
