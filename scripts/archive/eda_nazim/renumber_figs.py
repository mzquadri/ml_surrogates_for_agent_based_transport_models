"""
Cleanly renumber all Fig: N. prefixes in caption_box calls sequentially.
Strategy: strip any existing 'Fig: N. ' prefix, then add new ones in order.
"""

import re

path = r"C:\Users\zamin\Downloads\Nazim\make_slides.py"
content = open(path, "r", encoding="utf-8").read()

# Step 1: strip all existing "Fig: N. " prefixes from string literals in caption_box calls
# Pattern: "Fig: \d+. " at the start of a string after a quote char
content = re.sub(r'(?<=["\'])Fig: \d+\. ', "", content)

# Step 2: find all caption_box( ... ) blocks and number the text argument
# Each caption_box call: caption_box(\n  s,\n  l,\n  t,\n  w,\n  "TEXT"\n)
# The text is always the 5th argument (index 4), on its own line starting with a quote

fig_num = 1


def number_caption(m):
    global fig_num
    block = m.group(0)
    # Find the text arg: first quoted string in the block (skip "s,")
    # The text arg starts with indentation + quote on a line AFTER 4 preceding args
    lines = block.split("\n")
    # Find the line with the actual text (has a " or ' but isn't "s,")
    text_line_idx = None
    arg_count = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if idx == 0:
            continue  # "caption_box(" line
        # Count args: s, l, t, w (4 before text)
        if s in ("s,", "s"):
            arg_count += 1
            continue
        if re.match(r"^[0-9]", s):
            arg_count += 1
            continue
        if s.startswith('"') or s.startswith("'"):
            text_line_idx = idx
            break

    if text_line_idx is None:
        return block

    line = lines[text_line_idx]
    q = '"' if '"' in line else "'"
    # Find position of quote
    qpos = line.index(q)
    # Insert Fig: N. after the opening quote
    new_line = line[: qpos + 1] + f"Fig: {fig_num}. " + line[qpos + 1 :]
    lines[text_line_idx] = new_line
    fig_num += 1
    return "\n".join(lines)


# Match caption_box calls (non-greedy across lines)
pattern = re.compile(r"caption_box\(.*?\)", re.DOTALL)


def replacer(m):
    block = m.group(0)
    if "def caption_box" in block:
        return block
    return number_caption(m)


content = pattern.sub(replacer, content)
open(path, "w", encoding="utf-8").write(content)
print(f"Done. Numbered {fig_num - 1} captions.")
