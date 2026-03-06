@echo off
echo Starting Ollama server in a new window...
start "Ollama Server" ollama serve

echo Starting Discord bot...
title Discord Bot
python main.py
