"""Extract physics & world model code from Structured Baseline."""
import json
import os

NB_PATH = r"C:\code\[kaggle]\Orbit Wars\other's_work\orbit-wars-structured-baseline.ipynb"
OUT_DIR = r"C:\code\[kaggle]\Orbit Wars\src"

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

sections = {}
current = "unknown"

for cell in nb['cells']:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    if cell['cell_type'] == 'markdown':
        if 'Shared Setup' in src or 'Shared Configuration' in src:
            current = "config_and_types"
        elif 'Physics' in src:
            current = "physics"
        elif 'World Model' in src:
            current = "world_model"
        continue

    if cell['cell_type'] == 'code' and src.strip():
        sections.setdefault(current, []).append(src)

# Write each section
for section_name, codes in sections.items():
    full = '\n\n'.join(codes)
    fname = os.path.join(OUT_DIR, f'_extracted_{section_name}.py')
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"Wrote {fname}: {len(full)} chars, {len(codes)} cells")
