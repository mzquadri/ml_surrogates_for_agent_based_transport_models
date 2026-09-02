import re,os

from pathlib import Path

# Resolved from this file's location rather than hardcoded, so the script runs from any
# checkout. They previously carried an absolute path from one developer machine.
REPO_ROOT = Path(__file__).resolve().parents[3]
TEX_DIR = REPO_ROOT / "thesis" / "latex_tum_official"

tex_dir = str(TEX_DIR)
bib_file=os.path.join(tex_dir,"bibliography.bib")
with open(bib_file,"r",encoding="utf-8") as f:
    bib=f.read()

# Parse entries
entries={}
texts={}
for m in re.finditer(r"@(\w+)\{(\w+)\s*,",bib):
    et=m.group(1).lower();ek=m.group(2)
    entries[ek]=et
    s=m.start();bc=0;e=s
    for i in range(s,len(bib)):
        if bib[i]=="{":bc+=1
        elif bib[i]=="}":
            bc-=1
            if bc==0:e=i+1;break
    texts[ek]=bib[s:e]

def gf(t):
    fs=set()
    for fm in re.finditer(r"^\s*(\w+)\s*=",t,re.MULTILINE):fs.add(fm.group(1).lower())
    return fs

req={"article":["author","title","journal","year"],"inproceedings":["author","title","booktitle","year"],"book":[["author","editor"],"title","publisher","year"],"phdthesis":["author","title","school","year"]}
print("=== CHECK 5 ===")
ok=True
for k in sorted(texts):
    et=entries[k]
    if et not in req:continue
    fs=gf(texts[k])
    for r in req[et]:
        if isinstance(r,list):
            if not any(x in fs for x in r):print(f"  FAIL:{k} missing {r}");ok=False
        else:
            if r not in fs:print(f"  FAIL:{k} missing {r}");ok=False
if ok:print("  All PASS")
