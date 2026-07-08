import json

with open('only_mode/tabular-only-01-07.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def run_cv' in source and 'pickle.dump(encoders, f)' in source:
            new_source = source.replace(
'''        if cat_cols and fold == 0:
            with open('tabm_ablation_weights/cat_encoders.pkl', 'wb') as f:
                pickle.dump(encoders, f)''',
'''        # Removed cat_encoders.pkl dump since FoldTabularPreprocessor handles it internally'''
            )
            
            lines = new_source.splitlines(True)
            cell['source'] = lines

with open('only_mode/tabular-only-01-07.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Fixed encoders NameError in notebook')
