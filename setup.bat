@echo off
REM Install PRISM dependencies (run once, from D:\ICLR)
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m pip install -e .
echo.
echo Setup complete. Next: run_tests.bat
