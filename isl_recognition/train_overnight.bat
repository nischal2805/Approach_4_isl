@echo off
setlocal enabledelayedexpansion

:: ============================================
:: ISL Training Pipeline - With Full Notifications
:: ============================================

:: Configuration
set DATASET_DIR=E:\dataset\INCLUDE50
set PROJECT_DIR=E:\5thsem el\Approach_4\isl_recognition
set APP_ASSETS=E:\5thsem el\Approach_4\isl_app\assets
set VENV_DIR=E:\5thsem el\kortex_5th_sem\kortex
set WORKERS=14

:: Telegram Configuration
set TELEGRAM_BOT_TOKEN=8495584439:AAGZ3dwbjGKTsq8yiEZ_pSwybEsmLDeA7aE
set TELEGRAM_CHAT_ID=6600711973

echo ============================================
echo ISL Training Pipeline Starting...
echo Time: %date% %time%
echo ============================================

call :send_telegram "[START] ISL Training Pipeline started at %time%"

:: Step 1: Activate virtual environment
echo [1/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    call :send_telegram "[ERROR] Failed to activate venv"
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
call :send_telegram "[1/4] Venv activated. Starting extraction..."

:: Step 2: Run extraction
echo.
echo [2/4] Extracting landmarks from %DATASET_DIR%...
call :send_telegram "[2/4] Starting landmark extraction with %WORKERS% cores. This takes 1-2 hours..."

python extract_all.py --input "%DATASET_DIR%" --output-training data/processed --output-signs "%APP_ASSETS%\signs\words" --workers %WORKERS% 2>&1

if %ERRORLEVEL% NEQ 0 (
    call :send_telegram "[ERROR] Extraction FAILED! Check console."
    pause
    exit /b 1
)

call :send_telegram "[2/4] Extraction complete! Starting training..."

:: Step 3: Run training
echo.
echo [3/4] Training RandomForest model...
call :send_telegram "[3/4] Training RandomForest classifier..."

python run_training.py --processed data/processed --output models 2>&1

if %ERRORLEVEL% NEQ 0 (
    call :send_telegram "[ERROR] Training FAILED! Check console."
    pause
    exit /b 1
)

:: Get metrics if possible
for /f "tokens=*" %%a in ('findstr /c:"Test Accuracy" models\metrics.txt 2^>nul') do set ACCURACY=%%a

call :send_telegram "[3/4] Training complete! %ACCURACY%"

:: Step 4: Copy files to Flutter app
echo.
echo [4/4] Copying files to Flutter app...

if not exist "%APP_ASSETS%\signs\words" mkdir "%APP_ASSETS%\signs\words"
if not exist "%APP_ASSETS%\signs\letters" mkdir "%APP_ASSETS%\signs\letters"
if not exist "%APP_ASSETS%\signs\numbers" mkdir "%APP_ASSETS%\signs\numbers"

copy /Y "models\model.pkl" "%APP_ASSETS%\" 2>nul
copy /Y "models\labels.txt" "%APP_ASSETS%\" 2>nul

python export_scaler.py --input data/processed --output "%APP_ASSETS%" 2>&1

:: Done!
echo.
echo ============================================
echo TRAINING COMPLETE!
echo Time: %date% %time%
echo ============================================

call :send_telegram "[DONE] ISL Training COMPLETE at %time%! Model ready. Run TFLite conversion manually if needed."

echo.
echo NOTE: TFLite export skipped (no tensorflow in venv).
echo Run manually in global env:
echo   python convert_to_mobile.py --model models/model.pkl --data data/processed --output models
echo.

pause
exit /b 0

:: ============================================
:: Telegram function
:: ============================================
:send_telegram
set MSG=%~1
echo [Telegram] %MSG%
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d "chat_id=%TELEGRAM_CHAT_ID%" -d "text=%MSG%" >nul 2>&1
goto :eof
