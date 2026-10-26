@echo off
rem Starts the tyre tread-depth app on this PC and opens it in your browser.
rem Open it at http://localhost:8501 on THIS computer to get the folder browser.
rem To let colleagues use it, give them  http://<this-PC-IP>:8501  (they won't see any folders on your PC).
cd /d "%~dp0"

python -c "import timm, streamlit, torch, cv2" 2>nul
if errorlevel 1 (
    echo.
    echo A required package is missing for the Python that is running now:
    python -c "import sys; print(sys.executable)"
    echo.
    echo Install everything with:
    echo     python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

python -m streamlit run streamlit_app.py --server.port 8501
pause
