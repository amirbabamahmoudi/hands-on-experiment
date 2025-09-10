#!/usr/bin/env python3
"""
Quick fix for the notebook to replace silent=True with verbose=False
"""

import json
import sys

def fix_notebook():
    # Read the notebook
    with open('pycaret_analysis_notebook.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find and fix the cell with silent=True
    fixed = False
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            # Convert source to string if it's a list
            if isinstance(cell['source'], list):
                source_str = ''.join(cell['source'])
            else:
                source_str = cell['source']
            
            if 'silent=True' in source_str:
                print(f"Found cell {i} with silent=True, fixing...")
                
                # Fix the source
                if isinstance(cell['source'], list):
                    # Fix each line in the list
                    for j, line in enumerate(cell['source']):
                        if 'silent=True' in line:
                            cell['source'][j] = line.replace('silent=True', 'verbose=False  # Fixed: replaced silent=True with verbose=False')
                            print(f"Fixed line: {cell['source'][j].strip()}")
                            fixed = True
                else:
                    # Fix the string
                    cell['source'] = source_str.replace('silent=True', 'verbose=False  # Fixed: replaced silent=True with verbose=False')
                    fixed = True
                
                # Clear any existing error output
                if 'outputs' in cell:
                    cell['outputs'] = []
                if 'execution_count' in cell:
                    cell['execution_count'] = None
                
                break
    
    if fixed:
        # Write the fixed notebook back
        with open('pycaret_analysis_notebook.ipynb', 'w') as f:
            json.dump(notebook, f, indent=1)
        print("✅ Notebook fixed successfully!")
        print("🔄 The 'silent=True' parameter has been replaced with 'verbose=False'")
        print("🚀 You can now run the notebook without errors!")
    else:
        print("❌ Could not find 'silent=True' in any cell")

if __name__ == "__main__":
    fix_notebook()
