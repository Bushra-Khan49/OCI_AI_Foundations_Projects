import json

notebook_path = '/Users/bushrakhan/Desktop/OCI_AI_Foundations_Projects/01_Synthetic_Nonlinear_Classification/01_mlp_make_circles.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'markdown':
        print(f"--- Cell {i} ---")
        print("".join(cell['source']))
