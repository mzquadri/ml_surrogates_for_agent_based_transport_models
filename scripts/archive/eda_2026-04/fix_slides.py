"""Fix corrupted make_slides.py and apply TUM-style footer with slide numbers."""

import re

path = r"C:\Users\zamin\Downloads\Nazim\make_slides.py"
content = open(path, "r", encoding="utf-8").read()

# 1. Remove injected PowerShell blocks (they replaced the footer(s) calls)
bad_block = (
    "\n    param($match)\n"
    '    $result = "footer(s, slide_num=$slideNum)"\n'
    "    $script:slideNum++\n"
    "    $result\n"
)
count = content.count(bad_block)
print(f"Bad blocks found: {count}")
content = content.replace(bad_block, "\nfooter(s)\n")

# Verify footer(s) calls are back
calls = len(re.findall(r"\bfooter\(s\)", content))
print(f"footer(s) calls after restore: {calls}")

open(path, "w", encoding="utf-8").write(content)
print("Step 1 done (PS blocks removed).")

# 2. Now replace each footer(s) with footer(s, slide_num=N), slides 2..29
content = open(path, "r", encoding="utf-8").read()
slide_num = [2]


def replacer(m):
    result = f"footer(s, slide_num={slide_num[0]})"
    slide_num[0] += 1
    return result


content = re.sub(r"\bfooter\(s\)", replacer, content)
print(f"footer calls replaced, final slide_num={slide_num[0]}")
open(path, "w", encoding="utf-8").write(content)
print("Step 2 done.")
