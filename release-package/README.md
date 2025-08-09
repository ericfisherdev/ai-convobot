# AI Companion - Windows Release

AI Companion is a full-stack local chatbot application that runs entirely on your Windows machine without requiring internet connectivity for conversations.

## System Requirements

- Windows 10 or later (x64)
- At least 4GB of RAM (8GB+ recommended for larger models)
- 500MB+ free disk space

## Quick Start

1. Double-click `start.bat` to launch AI Companion
2. Open your web browser and go to http://localhost:3000
3. The application will automatically open in your default browser

## Manual Launch

If the batch launcher doesn't work, you can run the application manually:

1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run: `ai-companion.exe`
4. Open your browser to http://localhost:3000

## Features

- Local AI chatbot with personality tracking
- Complete offline operation - no internet required for conversations
- Advanced attitude and emotion modeling
- Long-term memory with conversation search
- Character card import support (.json/.png)
- Responsive web interface optimized for desktop and mobile
- Enhanced console output with attitude changes and third-party mentions
- Improved message layout with persistent scrollbar
- Optimized UI layout with input above attitude summary

## Model Requirements

AI Companion requires GGUF format models. Popular options include:
- Llama 2/3 models
- Mistral models
- Code Llama models

Place your GGUF model files in a `models/` directory or configure the path in the application settings.

## Configuration

The application stores its data in a local SQLite database and creates configuration files on first run. All data remains on your machine.

## Support

For technical issues or questions, visit: https://github.com/ericfisherdev/ai-convobot

## Version Information

This Windows build includes both the React frontend and Rust backend in a single executable for easy deployment and use.

Build Date: 2025-08-09
Includes fixes: ACB-63, ACB-64, ACB-65