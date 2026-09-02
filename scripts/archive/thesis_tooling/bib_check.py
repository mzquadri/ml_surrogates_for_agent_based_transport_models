import re, os, glob

from pathlib import Path

# Resolved from this file's location rather than hardcoded, so the script runs from any
# checkout. They previously carried an absolute path from one developer machine.
REPO_ROOT = Path(__file__).resolve().parents[3]
TEX_DIR = REPO_ROOT / "thesis" / "latex_tum_official"


tex_dir = str(TEX_DIR)
cite_keys = set()
cite_pattern = re.compile(r'[\]cite[tp]?\{([^}]+)\}')

for tex_file in glob.glob(os.path.join(tex_dir, '**', '*.tex'), recursive=True):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    for m in cite_pattern.finditer(content):
        keys_str = m.group(1)
        for key in keys_str.split(','):
            cite_keys.add(key.strip())

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
print()

from collections import Counter
type_counts = Counter(bib_entries.values())
print('=== ENTRY COUNTS BY TYPE ===')
for t in sorted(type_counts):
    print('  @' + t + ': ' + str(type_counts[t]))
total_target = type_counts.get('article',0) + type_counts.get('inproceedings',0) + type_counts.get('book',0) + type_counts.get('phdthesis',0)
print('  Total (article+inproceedings+book+phdthesis): ' + str(total_target))
print()

missing = cite_keys - set(bib_entries.keys())
print('=== CITED BUT NOT IN BIB ===')
if missing:
    for k in sorted(missing):
        print('  MISSING: ' + k)
else:
    print('  None - PASS')
print()

orphaned = set(bib_entries.keys()) - cite_keys
print('=== IN BIB BUT NOT CITED ===')
if orphaned:
    for k in sorted(orphaned):
        print('  ORPHANED: ' + k)
else:
    print('  None - PASS')
print()

def get_fields(entry_text):
    fields = set()
    for fm in re.finditer(r'^\s*(\w+)\s*=', entry_text, re.MULTILINE):
        fields.add(fm.group(1).lower())
    return fields

required_fields = {
    'article': ['author', 'title', 'journal', 'year'],
    'inproceedings': ['author', 'title', 'booktitle', 'year'],
    'book': [['author', 'editor'], 'title', 'publisher', 'year'],
    'phdthesis': ['author', 'title', 'school', 'year'],
}

print('=== CHECK 5: REQUIRED FIELD CHECKS ===')
all_fields_ok = True
for key in sorted(entry_texts):
    etype = bib_entries[key]
    if etype not in required_fields:
        continue
    fields = get_fields(entry_texts[key])
    reqs = required_fields[etype]
    for req in reqs:
        if isinstance(req, list):
            if not any(r in fields for r in req):
                print('  FAIL: ' + key + ' (@' + etype + ') missing one of ' + str(req))
                all_fields_ok = False
        else:
            if req not in fields:
                print('  FAIL: ' + key + ' (@' + etype + ') missing field: ' + req)
                all_fields_ok = False
if all_fields_ok:
    print('  All entries have required fields - PASS')
print()

print('=== CHECK 6: wang2023uncertainty details ===')
wang = entry_texts.get('wang2023uncertainty', '')
wang_author = re.search(r'author\s*=\s*\{([^}]+)\}', wang)
if wang_author:
    authors = wang_author.group(1)
    print('  Authors: ' + authors)
    expected = 'Wang, Qingyi and Wang, Shenhao and Zhuang, Dingyi and Koutsopoulos, Haris and Zhao, Jinhua'
    if expected in authors:
        print('  Author (5 names): PASS')
    else:
        print('  Author: FAIL')
else:
    print('  Author: FAIL - not found')

wang_journal = re.search(r'journal\s*=\s*\{([^}]+)\}', wang)
if wang_journal:
    j = wang_journal.group(1)
    print('  Journal: ' + j)
    ok = 'IEEE Transactions on Intelligent Transportation Systems' in j
    print('  Journal check: ' + ('PASS' if ok else 'FAIL'))
else:
    print('  Journal: FAIL - not found')

wang_vol = re.search(r'volume\s*=\s*\{?(\d+)', wang)
if wang_vol:
    v = wang_vol.group(1)
    print('  Volume: ' + v + ' -> ' + ('PASS' if v=='25' else 'FAIL'))
else:
    print('  Volume: FAIL - not found')

wang_pages = re.search(r'pages\s*=\s*\{([^}]+)\}', wang)
if wang_pages:
    p = wang_pages.group(1)
    print('  Pages: ' + p + ' -> ' + ('PASS' if '8770--8781' in p else 'FAIL'))
else:
    print('  Pages: FAIL - not found')

wang_year = re.search(r'year\s*=\s*\{?(\d+)', wang)
if wang_year:
    y = wang_year.group(1)
    print('  Year: ' + y + ' -> ' + ('PASS' if y=='2024' else 'FAIL'))
else:
    print('  Year: FAIL - not found')
print()

print('=== CHECK 7: {B}ayesian in titles ===')
for key in ['hasanzadeh2020bayesian', 'zhang2019bayesian']:
    text = entry_texts.get(key, '')
    title_m = re.search(r'title\s*=\s*\{(.+?)\}', text)
    if title_m:
        title = title_m.group(1)
        if '{B}ayesian' in title:
            print('  ' + key + ': PASS')
        else:
            print('  ' + key + ': FAIL - title="' + title + '"')
    else:
        print('  ' + key + ': FAIL - no title found')
print()

print('=== CHECK 8: Proceedings of in booktitle ===')
for key in ['kingma2015adam', 'li2018diffusion']:
    text = entry_texts.get(key, '')
    bt_m = re.search(r'booktitle\s*=\s*\{(.+?)\}', text)
    if bt_m:
        bt = bt_m.group(1)
        if 'Proceedings of' in bt:
            print('  ' + key + ': PASS (booktitle=' + bt + ')')
        else:
            print('  ' + key + ': FAIL (booktitle=' + bt + ')')
    else:
        print('  ' + key + ': FAIL - no booktitle')
print()

print('=== CHECK 9: fuchsgruber2024energy no 37 in booktitle ===')
text = entry_texts.get('fuchsgruber2024energy', '')
bt_m = re.search(r'booktitle\s*=\s*\{(.+?)\}', text)
if bt_m:
    bt = bt_m.group(1)
    if '37' not in bt:
        print('  PASS (booktitle=' + bt + ')')
    else:
        print('  FAIL (booktitle=' + bt + ' contains "37")')
else:
    print('  FAIL - no booktitle found')
