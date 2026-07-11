import sys

def md_to_py(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```python'):
            in_code_block = True
            continue
        elif line.strip().startswith('```') and in_code_block:
            in_code_block = False
            continue
            
        if in_code_block:
            out_lines.append(line)
        else:
            if line.strip() == '':
                out_lines.append(line)
            else:
                out_lines.append(f'# {line}')
                
    py_filepath = filepath.rsplit('.md', 1)[0] + '.py'
    with open(py_filepath, 'w') as f:
        f.writelines(out_lines)

md_to_py('01_Synthetic_Nonlinear_Classification/train_classifier.md')
md_to_py('02_gene_expression_classification/train_eval.md')
