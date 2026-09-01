"""Verify every cited key in the thesis exists in bibliography.bib."""
import re, glob, os

ROOT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document"

cited = set()
for f in glob.glob(f"{ROOT}/chapters/*.tex") + glob.glob(f"{ROOT}/pages/*.tex"):
    txt = open(f, encoding="utf-8").read()
    for m in re.finditer(r"\\(?:cite|textcite)\{([^}]+)\}", txt):
        for k in m.group(1).split(","):
            cited.add(k.strip())

defined = set()
for m in re.finditer(r"^@\w+\{(\S+?),", open(f"{ROOT}/bibliography.bib", encoding="utf-8").read(), re.MULTILINE):
    defined.add(m.group(1))

missing = cited - defined
unused = defined - cited

print(f"Citation keys used in thesis : {len(cited)}")
print(f"Bibliography entries defined : {len(defined)}")
print()
print(f"Missing (cited but undefined) : {len(missing)}")
for k in sorted(missing):
    print(f"  - {k}")
print()
print(f"Unused (defined but uncited)  : {len(unused)}")
for k in sorted(unused):
    print(f"  - {k}")
