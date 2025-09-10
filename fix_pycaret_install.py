#!/usr/bin/env python3
"""
Fix PyCaret Installation Script

This script fixes the common scipy compatibility issue with PyCaret:
"ImportError: cannot import name 'interp' from 'scipy'"

The issue occurs because PyCaret 3.3.2 is not compatible with scipy >= 1.12.0
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and print the result"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"⚠️  {description} had some issues:")
            if result.stderr:
                print(result.stderr.strip())
    except Exception as e:
        print(f"❌ Error during {description}: {e}")

def test_pycaret_import():
    """Test if PyCaret can be imported successfully"""
    print("🧪 Testing PyCaret import...")
    try:
        from pycaret.regression import setup
        print("✅ PyCaret imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ PyCaret import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Other error: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 PYCARET INSTALLATION FIX")
    print("=" * 60)
    print("This script fixes the scipy compatibility issue with PyCaret")
    print("Error: 'cannot import name interp from scipy'")
    print()
    
    # Step 1: Uninstall conflicting packages
    run_command("pip uninstall -y pycaret scipy", "Uninstalling conflicting packages")
    
    # Step 2: Install compatible scipy version
    run_command("pip install scipy==1.11.4", "Installing compatible SciPy version")
    
    # Step 3: Install PyCaret
    run_command("pip install pycaret==3.3.2", "Installing PyCaret")
    
    # Step 4: Install other requirements
    if os.path.exists("requirements.txt"):
        run_command("pip install -r requirements.txt", "Installing other requirements")
    
    print("\n" + "=" * 60)
    print("🧪 TESTING INSTALLATION")
    print("=" * 60)
    
    # Test the installation
    if test_pycaret_import():
        print("\n🎉 SUCCESS! PyCaret is now installed and working!")
        print("🚀 You can now run your PyCaret analysis:")
        print("   - python pycaret_analysis.py")
        print("   - Or open pycaret_analysis_notebook.ipynb in Jupyter")
    else:
        print("\n❌ Installation still has issues.")
        print("💡 Manual fix:")
        print("   pip install scipy==1.11.4")
        print("   pip install pycaret==3.3.2")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
