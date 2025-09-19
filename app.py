"""
Disease Prediction System - Main Entry Point
This file serves as the main entry point for deployment platforms.
It redirects to the actual app located in Project_main/app.py
"""

import streamlit as st
import sys
import os

# Add the Project_main directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_main_dir = os.path.join(current_dir, 'Project_main')
sys.path.insert(0, project_main_dir)

# Change working directory to Project_main so relative paths work correctly
os.chdir(project_main_dir)

# Import and run the main app
try:
    # Import the main app module
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(project_main_dir, "app.py"))
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
except Exception as e:
    st.error(f"Failed to load the application: {e}")
    st.info("Please ensure all required files are present in the Project_main directory.")