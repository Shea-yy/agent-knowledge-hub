@echo off
setlocal
title AgentKnowledgeHub

where docker >nul 2>&1
if errorlevel 1 (
    echo Docker CLI is not available. Start Docker Desktop and try again.
    exit /b 1
)

if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY is not set. Set it in this shell before starting the stack.
    exit /b 1
)

cd /d "%~dp0.."
echo Starting AgentKnowledgeHub with Docker Compose...
docker compose up --build
endlocal
