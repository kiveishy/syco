@echo off
REM ---------------------------------------------------------------------------
REM  Syco RAG prototype launcher
REM
REM    run.cmd                 start the chat UI (default)
REM    run.cmd ui              start the chat UI
REM    run.cmd ingest [args]   build/update the RAG from training\*.pdf  (e.g. run.cmd ingest --force)
REM                            run.cmd ingest --file "C:\books\x.pdf"   add one book    run.cmd ingest --list
REM    run.cmd query "text"    test retrieval from the command line (add --answer for a Claude answer)
REM    run.cmd setup           only create the venv / install dependencies
REM ---------------------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed"

REM ---- 1. virtual environment ------------------------------------------------
if not exist "%PY%" (
    echo [setup] Creating virtual environment in .venv ...
    where py >nul 2>&1 && (py -3 -m venv "%VENV%") || (python -m venv "%VENV%")
    if not exist "%PY%" (
        echo [error] Could not create the virtual environment. Install Python 3.10+ from python.org.
        exit /b 1
    )
)

REM ---- 2. dependencies: install when requirements.txt changed or imports fail -
set NEED_INSTALL=0
if not exist "%STAMP%" (
    set NEED_INSTALL=1
) else (
    fc /b "requirements.txt" "%STAMP%" >nul 2>&1 || set NEED_INSTALL=1
)
if "%NEED_INSTALL%"=="0" (
    "%PY%" -c "import pymupdf, torch, transformers, huggingface_hub, qdrant_client, streamlit, anthropic, dotenv" >nul 2>&1 || set NEED_INSTALL=1
)
if "%NEED_INSTALL%"=="1" (
    echo [setup] Installing dependencies - the first time this downloads torch, may take a while ...
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [error] Dependency installation failed.
        exit /b 1
    )
    copy /y "requirements.txt" "%STAMP%" >nul
    echo [setup] Dependencies OK.
) else (
    echo [check] Dependencies OK.
)

REM ---- 3. command ------------------------------------------------------------
set "CMD=%~1"
if "%CMD%"=="" set "CMD=ui"

if /i "%CMD%"=="setup" exit /b 0

REM (labels instead of if-blocks: %REST% inside a parenthesised block would be expanded before it is set)
if /i "%CMD%"=="ingest" goto :run_ingest
if /i "%CMD%"=="query" goto :run_query

if /i "%CMD%"=="ui" (
    if not exist "data\manifest.json" (
        echo [check] The RAG index is empty - building it from training\*.pdf first ...
        "%PY%" scripts\ingest.py
        if errorlevel 1 exit /b 1
    )
    echo [run] Starting chat UI at http://localhost:8501   ^(Ctrl+C to stop^)
    REM headless skips Streamlit's first-run e-mail prompt; open the browser ourselves once the server is up
    if not defined SYCO_NO_BROWSER start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
    "%PY%" -m streamlit run ui\app.py --server.headless true --browser.gatherUsageStats false
    exit /b %errorlevel%
)

echo Unknown command: %CMD%
echo Usage: run.cmd [ui ^| ingest [--force] [--file x.pdf] [--list] ^| query "question" [--answer] ^| setup]
exit /b 1

:run_ingest
call :shift_args %*
"%PY%" scripts\ingest.py %REST%
exit /b %errorlevel%

:run_query
call :shift_args %*
"%PY%" scripts\query.py %REST%
exit /b %errorlevel%

REM ---- helper: everything after the first argument -> %REST% -----------------
:shift_args
set "REST="
shift
:shift_loop
if "%~1"=="" goto :eof
set REST=%REST% %1
shift
goto shift_loop
