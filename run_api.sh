#!/bin/bash

# Marine Accident Classification API Runner
# This script starts the Flask API server

echo "🚢 Starting Marine Accident Classification API..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Please activate it first:"
    echo "   source marine_accident_env/bin/activate"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import flask, joblib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements_api.txt
fi

# Start the API
echo "🚀 Starting API server on http://210.125.100.136:5000"
echo "📊 API Documentation: http://210.125.100.136:5000/"
echo "🏥 Health Check: http://210.125.100.136:5000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py 