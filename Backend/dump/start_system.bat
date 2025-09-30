@echo off
echo Starting Sign Language Detection System...

echo.
echo Starting Backend API Server...
start cmd /k "cd /d %~dp0 && python start_api_server.py"

echo.
echo Starting Frontend Development Server...
start cmd /k "cd /d %~dp0\sign-talk-pal && npm run dev"

echo.
echo Sign Language Detection System is starting up!
echo Backend API: http://localhost:5000
echo Frontend UI: http://localhost:5173
echo.
echo Press any key to exit this window (servers will continue running)...
pause > nul