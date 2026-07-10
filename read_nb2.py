import json

notebook_path = '/Users/bushrakhan/Desktop/OCI_AI_Foundations_Projects/02_gene_expression_classification/notebook.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'markdown':
        print(f"--- Cell {i} ---")
        print("".join(cell['source']))
