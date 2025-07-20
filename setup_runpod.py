#!/usr/bin/env python3
"""
Automated RunPod Setup Script

This script automatically sets up the environment for RunPod deployment:
1. Checks if cross-validation fix is needed
2. Applies the fix if necessary
3. Validates the setup
4. Provides deployment status

Run this before training on RunPod to ensure proper cross-validation.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Detect and validate the current environment"""
    print("🔍 CHECKING ENVIRONMENT")
    print("=" * 50)
    
    # Import config after adding to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import BASE_PATH, FOLD_PATH
    from utils.helpers import get_base_path, is_runpod
    
    environment = "local"
    if is_runpod():
        environment = "runpod"
    elif os.path.exists('/kaggle'):
        environment = "kaggle"
    
    print(f"📊 Environment: {environment}")
    print(f"📂 Base path: {get_base_path()}")
    print(f"🎯 Dataset path: {BASE_PATH}")
    print(f"📄 Folds file: {FOLD_PATH}")
    
    return environment, BASE_PATH, FOLD_PATH

def check_dataset_availability(base_path):
    """Check if the BreakHis dataset is available"""
    print(f"\n🔍 CHECKING DATASET AVAILABILITY")
    print("=" * 50)
    
    # Check for main dataset directory
    dataset_path = base_path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    # Check for original Folds.csv
    original_folds = os.path.join(base_path, 'Folds.csv')
    if not os.path.exists(original_folds):
        print(f"❌ Original Folds.csv not found: {original_folds}")
        print("   Please ensure BreakHis dataset is properly uploaded to RunPod")
        return False
    
    # Check for images directory
    images_path = os.path.join(base_path, 'BreaKHis_v1')
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found: {images_path}")
        return False
    
    print(f"✅ Dataset found at: {dataset_path}")
    print(f"✅ Original folds: {original_folds}")
    print(f"✅ Images directory: {images_path}")
    
    return True

def check_cross_validation_fix(fold_path):
    """Check if cross-validation fix has been applied"""
    print(f"\n🔍 CHECKING CROSS-VALIDATION FIX")
    print("=" * 50)
    
    if os.path.exists(fold_path):
        # Check file size (fixed file should be much larger)
        file_size = os.path.getsize(fold_path)
        print(f"✅ Fixed folds file exists: {fold_path}")
        print(f"📊 File size: {file_size:,} bytes")
        
        if file_size > 20_000_000:  # Should be ~25MB
            print("✅ File size indicates proper cross-validation fix")
            return True
        else:
            print("⚠️  File size seems too small, may need regeneration")
            return False
    else:
        print(f"❌ Fixed folds file not found: {fold_path}")
        return False

def apply_cross_validation_fix():
    """Apply the cross-validation fix"""
    print(f"\n🔧 APPLYING CROSS-VALIDATION FIX")
    print("=" * 50)
    
    try:
        # Run the fix script
        result = subprocess.run([
            sys.executable, 
            'datasets/fix_cross_validation.py'
        ], capture_output=True, text=True, input='y\n')
        
        if result.returncode == 0:
            print("✅ Cross-validation fix applied successfully!")
            print("📄 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to apply cross-validation fix")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running fix script: {e}")
        return False

def validate_fix():
    """Validate that the cross-validation fix worked correctly"""
    print(f"\n🧪 VALIDATING CROSS-VALIDATION FIX")
    print("=" * 50)
    
    try:
        # Run the validation script
        result = subprocess.run([
            sys.executable, 
            'test_fixes.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and "ALL VALIDATIONS PASSED" in result.stdout:
            print("✅ Cross-validation fix validation PASSED!")
            return True
        else:
            print("❌ Cross-validation fix validation FAILED")
            print("Output:", result.stdout)
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

def main():
    """Main setup routine"""
    print("🚀 RUNPOD SETUP FOR BREAST CANCER CLASSIFICATION")
    print("=" * 60)
    
    # Step 1: Check environment
    try:
        environment, base_path, fold_path = check_environment()
    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False
    
    # Step 2: Check dataset availability
    if not check_dataset_availability(base_path):
        print(f"\n❌ SETUP FAILED: Dataset not available")
        print(f"📋 TODO for RunPod:")
        print(f"   1. Upload BreakHis dataset to {base_path}")
        print(f"   2. Ensure Folds.csv and BreaKHis_v1/ directory exist")
        print(f"   3. Re-run this setup script")
        return False
    
    # Step 3: Check if cross-validation fix is needed
    if check_cross_validation_fix(fold_path):
        print(f"\n✅ Cross-validation fix already applied!")
    else:
        print(f"\n🔧 Cross-validation fix needed, applying now...")
        if not apply_cross_validation_fix():
            print(f"\n❌ SETUP FAILED: Could not apply cross-validation fix")
            return False
    
    # Step 4: Validate the fix
    if not validate_fix():
        print(f"\n❌ SETUP FAILED: Cross-validation fix validation failed")
        return False
    
    # Final success message
    print(f"\n🎉 RUNPOD SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Environment: {environment}")
    print(f"✅ Dataset: Available and verified")
    print(f"✅ Cross-validation: Fixed and validated")
    print(f"🚀 Ready to run: python main.py")
    print(f"🎯 Expected results: Realistic 90-96% ensemble accuracy (not 100%)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print(f"\n✅ Setup successful. You can now run the training!")
        sys.exit(0)