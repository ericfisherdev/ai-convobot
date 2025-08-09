# AI Companion Deployment Guide

This guide covers various deployment options for AI Companion, including native binaries, Docker containers, and development setups.

## Quick Start

### Option 1: Download Pre-built Releases (Recommended)

1. Visit the [Releases page](https://github.com/ericfisherdev/ai-convobot/releases)
2. Download the appropriate build for your system:
   - **Windows**: `ai-companion-windows-x64-[cpu|cuda|opencl].zip`
   - **Linux**: `ai-companion-linux-x64-[cpu|cuda|opencl].tar.gz`
   - **macOS**: `ai-companion-macos-[x64|arm64]-[cpu|metal].tar.gz`
3. Extract and run the executable
4. Open `http://localhost:3000` in your browser

### Option 2: Docker (CPU-only)

```bash
# Create directories for models and data
mkdir -p models data

# Run CPU-only version
docker run -p 3000:3000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  ericfisherdev/ai-companion:latest-cpu
```

### Option 3: Docker with CUDA

```bash
# Requires NVIDIA Docker runtime
mkdir -p models data

docker run --gpus all -p 3000:3000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  ericfisherdev/ai-companion:latest-cuda
```

## Detailed Deployment Options

### Native Binary Deployment

#### System Requirements

**Minimum Requirements:**
- RAM: 4GB (8GB recommended)
- Storage: 500MB + model storage
- CPU: x64 architecture

**GPU Requirements (optional):**
- **CUDA**: NVIDIA GPU with Compute Capability 3.5+, CUDA 12.0+
- **OpenCL**: Any OpenCL 1.2+ compatible GPU
- **Metal**: macOS 10.15+ with Metal-compatible GPU

#### Installation Steps

1. **Download the appropriate build**:
   ```bash
   # Example for Linux CUDA build
   wget https://github.com/ericfisherdev/ai-convobot/releases/latest/download/ai-companion-linux-x64-cuda.tar.gz
   tar -xzf ai-companion-linux-x64-cuda.tar.gz
   cd ai-companion-linux-x64-cuda
   ```

2. **Run the application**:
   ```bash
   # Linux/macOS
   ./ai-companion
   
   # Or use the launcher script
   ./start.sh
   
   # Windows
   ai-companion.exe
   # Or double-click start.bat
   ```

3. **Configure your setup**:
   - Open `http://localhost:3000`
   - Go to Settings → Config
   - Set your GGUF model path
   - Configure GPU settings if applicable

#### Directory Structure
```
ai-companion/
├── ai-companion[.exe]          # Main executable
├── start.sh / start.bat        # Launcher script
├── README.md                   # Build-specific documentation
├── VERSION                     # Version information
├── companion_database.db       # Created on first run
└── models/                     # Your GGUF models (create manually)
```

### Docker Deployment

#### Using Docker Compose (Recommended)

1. **Clone the repository or download docker-compose.yml**:
   ```bash
   git clone https://github.com/ericfisherdev/ai-convobot.git
   cd ai-convobot
   ```

2. **Create required directories**:
   ```bash
   mkdir -p models data
   ```

3. **Place your GGUF models in the models directory**:
   ```bash
   cp your-model.gguf models/
   ```

4. **Start the service**:
   ```bash
   # CPU-only version
   docker-compose --profile cpu up -d
   
   # CUDA version (requires NVIDIA Docker)
   docker-compose --profile cuda up -d
   
   # Using pre-built images
   docker-compose --profile prebuilt-cpu up -d
   docker-compose --profile prebuilt-cuda up -d
   ```

#### Manual Docker Commands

**CPU Version:**
```bash
docker build -f Dockerfile.cpu -t ai-companion:cpu .
docker run -d --name ai-companion-cpu \
  -p 3000:3000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  ai-companion:cpu
```

**CUDA Version:**
```bash
docker build -f Dockerfile.cuda -t ai-companion:cuda .
docker run -d --name ai-companion-cuda \
  --gpus all \
  -p 3000:3000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  ai-companion:cuda
```

### Development Deployment

#### Prerequisites
- Node.js 18+
- Rust 1.75+
- Platform-specific build tools

#### Build from Source

1. **Clone and setup**:
   ```bash
   git clone https://github.com/ericfisherdev/ai-convobot.git
   cd ai-convobot
   npm install
   ```

2. **Build options**:
   ```bash
   # CPU-only build
   npm run build-full
   
   # CUDA build (requires CUDA toolkit)
   npm run build-full-cuda
   
   # OpenCL build (requires OpenCL headers)
   npm run build-full-opencl
   
   # Metal build (macOS only)
   npm run build-full-metal
   ```

3. **Run development server**:
   ```bash
   # Frontend + backend with auto-reload
   npm run dev-rs
   
   # Frontend only
   npm run dev
   ```

## Configuration

### GPU Memory Management

AI Companion includes intelligent GPU memory management:

1. **Enable Dynamic GPU Allocation**:
   - Go to Settings → Config
   - Toggle "Dynamic GPU Layer Allocation"
   - Configure safety margins and minimum free VRAM

2. **Manual Configuration**:
   - Set GPU Layers manually if needed
   - Adjust VRAM limit based on your system
   - Monitor GPU memory usage in real-time

### Environment Variables

Set these environment variables to customize behavior:

```bash
# Server configuration
COMPANION_HOST=0.0.0.0          # Bind address (default: 0.0.0.0)
COMPANION_PORT=3000             # Port (default: 3000)
RUST_LOG=info                   # Logging level (debug, info, warn, error)

# Database
DATABASE_PATH=./companion_database.db

# Docker-specific
NVIDIA_VISIBLE_DEVICES=all      # GPU visibility for CUDA
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Model Configuration

1. **Supported Formats**: GGUF models only
2. **Model Location**: 
   - Native: Any accessible path
   - Docker: Place in mounted `/app/models` directory
3. **Recommended Models**: 
   - 7B models: 4-8GB VRAM
   - 13B models: 8-16GB VRAM
   - 30B+ models: 24GB+ VRAM

## Production Deployment

### Security Considerations

1. **Firewall Configuration**:
   ```bash
   # Allow only specific IPs if needed
   ufw allow from YOUR_IP to any port 3000
   ```

2. **Reverse Proxy Setup** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

3. **SSL/TLS**: Use Let's Encrypt or similar for HTTPS

### Performance Optimization

1. **System Resources**:
   - Allocate sufficient RAM for models
   - Use SSD storage for faster model loading
   - Ensure adequate CPU/GPU resources

2. **GPU Optimization**:
   - Monitor VRAM usage
   - Use appropriate safety margins
   - Consider model quantization for memory constraints

3. **Monitoring**:
   - Check application logs regularly
   - Monitor system resources
   - Set up health checks for production deployments

### Scaling and High Availability

1. **Load Balancing**: Multiple instances behind load balancer
2. **Health Checks**: Implement endpoint monitoring
3. **Backup Strategy**: Regular database backups
4. **Container Orchestration**: Use Docker Swarm or Kubernetes for larger deployments

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Find process using port 3000
   lsof -i :3000
   # Kill process or change port
   ```

2. **GPU Not Detected**:
   - Verify GPU drivers installed
   - Check CUDA/OpenCL runtime
   - Review application logs for errors

3. **Model Loading Errors**:
   - Verify model path is correct
   - Check file permissions
   - Ensure model is in GGUF format

4. **Memory Issues**:
   - Reduce GPU layers
   - Increase safety margins
   - Use CPU fallback mode

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/ericfisherdev/ai-convobot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ericfisherdev/ai-convobot/discussions)
- **Documentation**: This repository's README and documentation files

## License

This project is licensed under the terms specified in the LICENSE file.