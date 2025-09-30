@echo off
echo Starting SignBridge Streamlit Demo...
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit not found. Installing requirements...
    pip install -r streamlit_requirements.txt
)

:: Launch the Streamlit app
echo Launching SignBridge Demo...
echo Open your browser to: http://localhost:8501
echo Press Ctrl+C to stop the demo
echo.

streamlit run streamlit_demo.py --server.port 8501 --server.address localhost

pause