"""
Add 'Fig: N. ' prefix to every caption_box text argument.
Handles both single-string and adjacent-string-literal cases.
"""

import re

path = r"C:\Users\zamin\Downloads\Nazim\make_slides.py"
lines = open(path, "r", encoding="utf-8").read().splitlines()

fig_num = 1
in_caption = False
caption_args_seen = 0  # count of comma-separated args before text

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # Detect start of a caption_box call (not the def)
    if re.match(r"^caption_box\($", stripped):
        in_caption = True
        caption_args_seen = 0
        new_lines.append(line)
        i += 1
        continue

    if in_caption:
        # Count args: s, l, t, w are 4 args before the text arg
        # Each arg ends with a comma on its own line
        if caption_args_seen < 4:
            if (
                stripped.rstrip(",")
                .replace(".", "")
                .replace("-", "")
                .lstrip("-")
                .isdigit()
                or stripped == "s,"
                or stripped.startswith("0.")
                or stripped.startswith("1")
                or stripped.startswith("2")
                or stripped.startswith("3")
                or stripped.startswith("4")
                or stripped.startswith("5")
                or stripped.startswith("6")
                or stripped.startswith("7")
                or stripped.startswith("s,")
                or re.match(r"^[0-9s]", stripped)
            ):
                caption_args_seen += 1
                new_lines.append(line)
                i += 1
                continue

        # We are at the text argument line(s)
        # Check if it already has Fig: prefix
        if re.search(r'"Fig:\s*\d', line) or re.search(r"'Fig:\s*\d", line):
            in_caption = False
            new_lines.append(line)
            i += 1
            continue

        # Find the opening quote in the line
        m = re.search(r'([ \t]*)(["\'])(.*)', line)
        if m:
            indent = m.group(1)
            q = m.group(2)
            rest = m.group(3)
            # Prepend Fig: N.
            new_line = f"{indent}{q}Fig: {fig_num}. {rest}"
            fig_num += 1
            new_lines.append(new_line)
            in_caption = False
            i += 1
            continue

    new_lines.append(line)
    if stripped == ")":
        in_caption = False
    i += 1

open(path, "w", encoding="utf-8").write("\n".join(new_lines) + "\n")
print(f"Done. Figures numbered up to Fig: {fig_num - 1}")
