import re, os, glob

from pathlib import Path

# Resolved from this file's location rather than hardcoded, so the script runs from any
# checkout. They previously carried an absolute path from one developer machine.
REPO_ROOT = Path(__file__).resolve().parents[3]
TEX_DIR = REPO_ROOT / "thesis" / "latex_tum_official"


tex_dir = str(TEX_DIR)
cite_keys = set()

cite_pattern = re.compile(r'\\cite[tp]?\{([a-zA-Z0-9_,\s]+)\}')

for tex_file in glob.glob(os.path.join(tex_dir, '**', '*.tex'), recursive=True):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    for m in cite_pattern.finditer(content):
        keys_str = m.group(1)
        for key in keys_str.split(','):
            k = key.strip()
            if k:
                cite_keys.add(k)

bib_file = os.path.join(tex_dir, 'bibliography.bib')
with open(bib_file, 'r', encoding='utf-8') as f:
    bib_content = f.read()

bib_entries = {}
entry_texts = {}
for m in re.finditer(r'@(\w+)\{(\w+)\s*,', bib_content):
    entry_type = m.group(1).lower()
    entry_key = m.group(2)
    bib_entries[entry_key] = entry_type
    start = m.start()
    brace_count = 0
    end = start
    for i in range(start, len(bib_content)):
        if bib_content[i] == '{':
            brace_count += 1
        elif bib_content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    entry_texts[entry_key] = bib_content[start:end]

print('=== CITATION KEYS FROM .TEX FILES ===')
for k in sorted(cite_keys):
    print('  ' + k)
print('Total unique citation keys: ' + str(len(cite_keys)))
print()

print('=== BIB ENTRIES ===')
for k in sorted(bib_entries):
    print('  ' + k + ' -> @' + bib_entries[k])
print('Total bib entries: ' + str(len(bib_entries)))
