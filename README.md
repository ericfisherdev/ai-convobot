# AI Companion - Enhanced Fork

A fork of [https://github.com/Hukasx0/ai-companion](https://github.com/Hukasx0/ai-companion) with significant enhancements and improvements.

**Release builds coming soon!**

## About This Fork

This enhanced version of AI Companion includes numerous improvements, bug fixes, and new features designed to provide a better user experience and more robust functionality. All core features from the original project are preserved while adding significant quality-of-life improvements.

## What's AI Companion?

AI Companion is a full-stack local chatbot application that runs entirely on your computer without requiring internet connectivity for conversations. Built with Rust (backend) and React/TypeScript (frontend), it provides a single binary with embedded web interface for easy deployment.

## ✨ New Features & Improvements

### 🎯 Easy LLM Model Selection (ACB-70)
- **Automatic Model Discovery**: Automatically scans `./llms` and `../llms` directories for GGUF files
- **Visual Model Browser**: Browse all available models with file size and metadata
- **Directory Management**: Add unlimited custom directories to scan for models
- **Smart Model Selector**: Dropdown selection with real-time updates
- **Cross-Platform Compatible**: Works seamlessly on Windows, Linux, and macOS

### 🚀 Performance Optimizations (ACB-68, ACB-67)
- **Response Time Improvements**: Optimized inference pipeline reducing 300+ second response times
- **Intelligent ETA Estimation**: Accurate response time predictions in console output
- **Memory Management**: Better resource allocation and cleanup
- **GPU Acceleration**: Enhanced CUDA, OpenCL, and Metal support

### 🖥️ Enhanced User Interface (ACB-64, ACB-65)
- **Improved Message Layout**: Messages no longer cut off at bottom
- **Persistent Scrollbar**: Always-visible scrollbar for better navigation
- **Repositioned Controls**: Chat input moved above attitude summary for better UX
- **Responsive Design**: Better handling of different screen sizes

### 🔧 Console & Debugging (ACB-63)
- **Clean Console Output**: Removed verbose tensor loading messages
- **Attitude Change Display**: Real-time attitude changes shown in console (e.g., "Love +2 | Trust +5 | Fear -10")
- **Third-Party Mentions**: Tracking and display of mentioned individuals (e.g., "Alicia mentioned for the 3rd time")
- **Response ETA**: Estimated response times displayed during generation

### 🎭 Better Third-Party Detection (ACB-66)
- **Improved Person Recognition**: Objects and activities no longer incorrectly recognized as people
- **Smarter Filtering**: Enhanced algorithms to distinguish between people and other entities
- **Cleaner Memory**: Reduced false positives in relationship tracking

### 🐛 Bug Fixes & Stability (ACB-69)
- **Date Display Fixed**: Resolved "Invalid Date, NaN @ invalid date" errors
- **Memory Leak Prevention**: Better resource cleanup and management
- **Cross-Platform Path Handling**: Improved file path resolution for all operating systems

## Core Features (Preserved from Original)

- **Complete Privacy**: All data stored locally in SQLite database
- **No Internet Required**: Fully offline operation after initial setup  
- **GPU Acceleration**: CUDA, OpenCL, and Metal support
- **Advanced Memory System**: Both short-term and long-term memory with associative recall
- **Character Cards**: Import .json and .png character card formats
- **REST API**: Use as backend for other projects
- **Roleplay Support**: Actions between asterisks (*waves hello*)
- **Real-time Learning**: AI learns about users through conversation
- **Time Awareness**: AI can access current date/time and remember when conversations occurred

## Quick Start

1. **Download**: Get the appropriate binary for your OS (coming soon)
2. **Setup Models**: Place GGUF model files in a `llms` folder next to the executable
3. **Launch**: Double-click the binary or run from command line
4. **Configure**: Open http://localhost:3000 and select your model from the dropdown
5. **Chat**: Start conversing with your AI companion!

## Model Management Made Easy

The new model selection system makes managing multiple LLM models effortless:

- **Automatic Discovery**: Just drop GGUF files in the `llms` folder
- **Multiple Directories**: Add as many model directories as needed
- **Visual Selection**: See all models with sizes and metadata
- **Hot Swapping**: Change models without restarting the application

## System Requirements

- **Windows**: Windows 10+ (x64)
- **Linux**: Any modern distribution
- **macOS**: macOS 10.14+
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 500MB+ free space (plus space for models)
- **GPU**: Optional but recommended for better performance

## Supported Models

Works with any GGUF format models including:
- Llama 2/3/3.1/3.2 series
- Mistral 7B/8x7B series  
- Code Llama variants
- Zephyr models
- Phi-3 models
- And many more!

## Development & Building

### Prerequisites
- [Node.js and npm](https://nodejs.org/)
- [Rust and cargo](https://www.rust-lang.org/)
- For GPU support: Follow [CUDA/OpenCL/Metal setup guide](https://github.com/rustformers/llm/blob/main/doc/acceleration-support.md)

### Build Commands
```bash
# Clone this repository
git clone https://github.com/ericfisherdev/ai-convobot
cd ai-convobot

# Install dependencies
npm install

# Build frontend and backend (CPU only)
npm run build-full

# GPU-accelerated builds
npm run build-full-cuda    # NVIDIA CUDA
npm run build-full-opencl  # OpenCL (AMD/Intel)
npm run build-full-metal   # Apple Metal (macOS)
```

Binary will be available in `backend/target/release/`

## API Documentation

Complete REST API documentation available at [/docs/api_docs.md](/docs/api_docs.md)

## Contributing

This fork welcomes contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request with detailed description

## Roadmap

- [ ] Streaming responses for real-time generation
- [ ] Plugin system for extensibility
- [ ] Multi-language UI support
- [ ] Enhanced voice synthesis integration
- [ ] Docker containerization
- [ ] Model quantization tools

## Acknowledgments

- Original project by [Hukasx0](https://github.com/Hukasx0/ai-companion)
- Built on [rustformers/llm](https://github.com/rustformers/llm)
- UI components from [shadcn/ui](https://ui.shadcn.com/)

## License

This project maintains the same license as the original AI Companion project.

---

**Note**: This is an independent fork focused on improvements and bug fixes. For the original project, visit [Hukasx0/ai-companion](https://github.com/Hukasx0/ai-companion).