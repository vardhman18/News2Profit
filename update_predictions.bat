@echo off
echo Generating latest predictions for NEWS2PROFIT...
echo.

REM Change to the project directory
cd /d "D:\NEWS2PROFIT"

REM Activate virtual environment and run the prediction script
call .venv\Scripts\activate.bat
python generate_latest_predictions.py

echo.
echo Predictions updated! Check data/processed/latest_predictions.csv
pause