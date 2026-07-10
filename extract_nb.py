import json

nb = json.load(open('01_Synthetic_Nonlinear_Classification/01_mlp_make_circles.ipynb'))
code = []
for c in nb['cells']:
    if c['cell_type'] == 'code':
        code.append(''.join(c['source']))

full_code = '\n'.join(code)

# Add Agg backend so we can test without X11
full_code = "import matplotlib\nmatplotlib.use('Agg')\n" + full_code

# Replace the interactive widget display with a manual call to the function to test if it crashes
full_code = full_code.replace("display(interactive_plot)", "update_plot(10)")

with open('test_nb.py', 'w') as f:
    f.write(full_code)
