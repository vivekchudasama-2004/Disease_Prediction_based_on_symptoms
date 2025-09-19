#!/usr/bin/env python3
"""
Quick deployment test script
This script checks if the Disease Prediction app is deployment-ready
"""

import os
import sys
import subprocess

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_directory_structure():
    """Check if all required files and directories exist"""
    print("🔍 Checking project structure...")
    
    checks = [
        ("app.py", "Root app entry point"),
        ("requirements.txt", "Dependencies file"),
        ("Procfile", "Heroku deployment config"),
        ("runtime.txt", "Python version specification"),
        (".streamlit/config.toml", "Streamlit configuration"),
        ("Project_main/app.py", "Main application"),
        ("Project_main/model_RFC.sav", "ML model file"),
        ("Project_main/Dataset/Symptom-severity.csv", "Symptom severity data"),
        ("Project_main/Dataset/symptom_precaution.csv", "Precaution data"),
        ("Project_main/Dataset/symptom_Description.csv", "Disease descriptions"),
        ("Project_main/Dataset/dataset.csv", "Training dataset"),
    ]
    
    all_good = True
    for file_path, description in checks:
        if not check_file_exists(file_path, description):
            all_good = False
    
    return all_good

def check_python_syntax():
    """Check Python syntax of main files"""
    print("\n🐍 Checking Python syntax...")
    
    files_to_check = [
        "app.py",
        "Project_main/app.py"
    ]
    
    all_good = True
    for file_path in files_to_check:
        try:
            subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                         check=True, capture_output=True)
            print(f"✅ Syntax OK: {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Syntax Error: {file_path}")
            print(f"   Error: {e.stderr.decode()}")
            all_good = False
    
    return all_good

def main():
    """Main deployment check function"""
    print("🚀 Disease Prediction App - Deployment Readiness Check")
    print("=" * 60)
    
    # Change to project directory
    project_dir = "/home/runner/work/Disease_Prediction_based_on_symptoms/Disease_Prediction_based_on_symptoms"
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Run checks
    structure_ok = check_directory_structure()
    syntax_ok = check_python_syntax()
    
    print("\n📊 DEPLOYMENT READINESS SUMMARY")
    print("-" * 40)
    
    if structure_ok and syntax_ok:
        print("✅ READY FOR DEPLOYMENT!")
        print("\n🎯 Next steps:")
        print("1. Push to GitHub repository")
        print("2. Deploy to Streamlit Cloud, Heroku, or Railway")
        print("3. Configure environment variables if using HuggingFace features")
        print("\n📖 See DEPLOYMENT.md for detailed instructions")
        return 0
    else:
        print("❌ NOT READY FOR DEPLOYMENT")
        print("Please fix the issues above before deploying.")
        return 1

if __name__ == "__main__":
    exit(main())