# AI Companion

A full-stack local AI chatbot, maintained as a standalone project.

**Release builds coming soon!**

## About This Project

This codebase began as a fork of Hubert "Hukasx0" Kasperek's AI Companion. That upstream repository is no longer publicly accessible, and this repository is no longer part of its fork network, so development continues here independently. The original work remains credited under its MIT license (see [Acknowledgments](#acknowledgments) and [License](#license)); the core features it established are preserved alongside substantial later changes.

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

### Quality Checks

`make check` runs formatting, lint, typecheck, frontend tests, frontend build,
clippy, and backend tests, in that order. These targets are the single source
of truth for what "passing" means; the git hooks below call them, and CI (#53)
is being migrated to do the same.

| Target | What it does |
| --- | --- |
| `make fmt` | Auto-formats Rust code with `cargo fmt` |
| `make fmt-check` | Checks Rust formatting without modifying files |
| `make lint` | Runs ESLint on the frontend |
| `make typecheck` | Runs the TypeScript compiler in `--noEmit` mode |
| `make test-fe` | Runs the frontend test suite (Vitest) |
| `make build-fe` | Builds the frontend (required before backend checks compile) |
| `make clippy` | Runs `cargo clippy` on the backend |
| `make test-be` | Runs the backend test suite (`cargo test`) |
| `make check` | Runs all of the above, in order |

On a fresh clone, run `npm ci` first. The `llama-cpp-2` build also requires
`cmake` and `pkg-config` on your `PATH`.

### Git hooks

```bash
# Install git hooks (pre-commit: fmt/eslint/tsc/whitespace; pre-push: make check)
pre-commit install
```

`pre-commit install` wires up both hook types in one command. The pre-commit
hook runs fast checks (`cargo fmt --check`, ESLint, `tsc --noEmit`, and
whitespace fixers) against staged files; the pre-push hook runs `make check`
in full, so a push rarely fails CI on something local checks could have
caught. To install only the pre-push hook explicitly, run
`pre-commit install --hook-type pre-push`.

## API Documentation

Complete REST API documentation available at [/docs/api_docs.md](/docs/api_docs.md)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request with detailed description

## Roadmap

- [x] Streaming responses for real-time generation
- [ ] Plugin system for extensibility
- [ ] Multi-language UI support
- [ ] Enhanced voice synthesis integration
- [ ] Docker containerization
- [ ] Model quantization tools

## Acknowledgments

- Original AI Companion by Hubert "Hukasx0" Kasperek. The upstream repository and its author's GitHub account are no longer reachable, so no link is given.
- Inference via [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs), bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp). Earlier versions used rustformers/llm, which was archived in June 2024.
- UI components from [shadcn/ui](https://ui.shadcn.com/)

## License

MIT, Copyright (c) 2025 Hubert Kasperek. See [LICENSE](LICENSE). The original
copyright notice is retained as the license requires; later changes are
released under the same terms.
