"""Verify every \\includegraphics reference resolves to a real file."""
import re, glob, os
import os
from pathlib import Path

DOC = os.environ.get(
    "THESIS_DOC_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "thesis" / "latex_tum_official"),
)

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
