@echo off
setlocal enabledelayedexpansion

REM Start langfuse container
echo Starting Langfuse container...
cd /d C:\Users\vadim\Documents\Vadym\GitRep\langfuse
docker compose up -d

REM Navigate to project
cd /d C:\Users\vadim\Documents\Vadym\GitRep\discord-assistant

REM Start Ollama server in a new window
echo Starting Ollama server in a new window...
start "Ollama Server" ollama serve

REM Start Discord bot
echo Starting Discord bot...
title Discord Bot
python main.py

REM Stop langfuse container when bot exits
echo Stopping Langfuse container...
cd /d C:\Users\vadim\Documents\Vadym\GitRep\langfuse
docker compose stop
