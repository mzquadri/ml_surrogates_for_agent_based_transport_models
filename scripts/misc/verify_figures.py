"""Verify every \\includegraphics reference resolves to a real file."""
import re, glob, os

DOC = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document"

refs = set()
pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
for f in glob.glob(f"{DOC}/chapters/*.tex") + glob.glob(f"{DOC}/pages/*.tex"):
    s = open(f, encoding="utf-8").read()
    for m in pat.finditer(s):
        refs.add(m.group(1))

missing = []
for r in sorted(refs):
    candidates = [
        os.path.join(DOC, r),
        os.path.join(DOC, r + ".pdf"),
        os.path.join(DOC, r + ".png"),
    ]
    if not any(os.path.exists(c) for c in candidates):
        missing.append(r)

print(f"Figure references: {len(refs)}")
print(f"Missing: {len(missing)}")
for r in missing:
    print(f"  - {r}")
print()
print("All references:")
for r in sorted(refs):
    print(f"  {r}")
