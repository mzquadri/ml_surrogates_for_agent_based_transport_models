"""Add Fig: N. prefix to every caption_box text, and trim long bullets."""

import re

path = r"C:\Users\zamin\Downloads\Nazim\make_slides.py"
content = open(path, "r", encoding="utf-8").read()

# ── 1. Number each caption_box text ──────────────────────────────────────────
# caption_box calls look like:
#   caption_box(
#       s,
#       0.3,
#       6.25,
#       8.2,
#       "Some text here",   <-- or a multi-line string joined
#   )
# We'll find each occurrence of caption_box and inject "Fig: N. " into the string.

fig_num = [1]


def add_fig_num(m):
    full = m.group(0)

    # Find the text argument: the first quoted string after the 4 numeric args
    def add_prefix(sm):
        q = sm.group(1)  # quote char
        text = sm.group(2)
        if text.startswith("Fig:"):
            return sm.group(0)  # already numbered
        return f"{q}Fig: {fig_num[0]}. {text}{q}"

    result = re.sub(r'(["\'])(?!\s*Fig:)((?:(?!\1).)+)\1', add_prefix, full, count=1)
    if result != full:
        fig_num[0] += 1
    return result


# Match a full caption_box(...) call (non-greedy, single-call)
pattern = re.compile(r"caption_box\(.*?\)", re.DOTALL)


# But we need to skip the function DEFINITION line
def replacer(m):
    # Skip the def line
    if "def caption_box" in m.group(0):
        return m.group(0)
    return add_fig_num(m)


content = pattern.sub(replacer, content)
print(f"Figures numbered up to Fig: {fig_num[0] - 1}")

open(path, "w", encoding="utf-8").write(content)
print("Done.")
