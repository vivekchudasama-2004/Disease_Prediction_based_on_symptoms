#!/bin/bash
# Streamlit app startup script with dependency checking

echo "🚀 Starting Disease Prediction System..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if streamlit is available
if ! python -c "import streamlit" &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Verify model file exists
if [ ! -f "Project_main/model_RFC.sav" ]; then
    echo "❌ Model file not found: Project_main/model_RFC.sav"
    echo "Please ensure the trained model is available"
    exit 1
fi

# Set default port if not specified
if [ -z "$PORT" ]; then
    PORT=8501
fi

echo "✅ Starting Streamlit app on port $PORT..."

# Start the application
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0