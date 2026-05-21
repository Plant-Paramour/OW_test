"""Extract physics & world model code from Structured Baseline v11 notebook."""
import json
import os

NB_PATH = r"C:\code\[kaggle]\Orbit Wars\other's_work\orbit-wars-structured-baseline.ipynb"
OUT_DIR = r"C:\code\[kaggle]\Orbit Wars\src"

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Concatenate all code cells into one big string with section markers
all_code = []
current_section = "unknown"

for cell in nb['cells']:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    if cell['cell_type'] == 'markdown':
        if 'Shared Setup' in src or 'Shared Configuration' in src:
            current_section = "config_and_types"
        elif 'Physics' in src and '🧱' in src:
            current_section = "physics"
        elif 'World Model' in src and '🛡️' in src:
            current_section = "world_model"
        elif 'Strategy' in src and '🤝' in src:
            current_section = "strategy"
        elif 'Agent Entry Point' in src and '🛰️' in src:
            current_section = "agent"
        continue

    if cell['cell_type'] == 'code' and src.strip():
        all_code.append(f"# === SECTION: {current_section} ===\n{src}")

full_code = '\n\n'.join(all_code)

# Print section boundaries to understand structure
for line in full_code.split('\n'):
    if line.startswith('# === SECTION:'):
        print(line)

print(f"\nTotal code length: {len(full_code)} chars")
print(f"Total lines: {len(full_code.split(chr(10)))}")
