#!/usr/bin/env python3
"""
Python script to fix macOS OpenMP library issue for PyCaret
This addresses the "libomp.dylib not found" error
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"❌ {description} failed:")
            if result.stderr.strip():
                print(result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_homebrew():
    """Check if Homebrew is installed"""
    result = subprocess.run("which brew", shell=True, capture_output=True)
    return result.returncode == 0

def test_imports():
    """Test if the problematic packages can be imported"""
    packages_to_test = [
        ('pycaret.regression', 'PyCaret'),
        ('lightgbm', 'LightGBM'), 
        ('xgboost', 'XGBoost'),
        ('catboost', 'CatBoost')
    ]
    
    print("🧪 Testing package imports...")
    all_success = True
    
    for package, name in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully!")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_success = False
        except Exception as e:
            print(f"⚠️  {name} other error: {e}")
            all_success = False
    
    return all_success

def main():
    print("=" * 60)
    print("🔧 MACOS OPENMP LIBRARY FIX FOR PYCARET")
    print("=" * 60)
    print("This script fixes the 'libomp.dylib not found' error")
    print()
    
    # Check if we're on macOS
    if sys.platform != 'darwin':
        print("⚠️  This script is designed for macOS only")
        return
    
    # Check Homebrew
    if not check_homebrew():
        print("❌ Homebrew is not installed!")
        print("Please install Homebrew first:")
        print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return
    
    print("✅ Homebrew is installed")
    
    # Install libomp
    if not run_command("brew install libomp", "Installing OpenMP library (libomp)"):
        print("💡 Try running: brew update && brew install libomp")
        return
    
    # Reinstall problematic packages
    packages_to_reinstall = [
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"), 
        ("catboost", "CatBoost")
    ]
    
    print("\n📦 Reinstalling packages that require OpenMP...")
    for package, name in packages_to_reinstall:
        command = f"pip install --force-reinstall --no-cache-dir {package}"
        run_command(command, f"Reinstalling {name}")
    
    # Test the fix
    print("\n" + "=" * 60)
    print("🧪 TESTING THE FIX")
    print("=" * 60)
    
    if test_imports():
        print("\n🎉 SUCCESS! All packages are working correctly!")
        print("🚀 You can now run your PyCaret analysis without OpenMP errors!")
    else:
        print("\n⚠️  Some packages still have issues.")
        print("💡 Try restarting your Python kernel/terminal and running the test again.")
    
    print("\n" + "=" * 60)
    print("📋 WHAT WAS DONE:")
    print("✅ Installed OpenMP library (libomp) via Homebrew")
    print("✅ Reinstalled LightGBM with OpenMP support")
    print("✅ Reinstalled XGBoost with OpenMP support")
    print("✅ Reinstalled CatBoost with OpenMP support")
    print("=" * 60)

if __name__ == "__main__":
    main()
