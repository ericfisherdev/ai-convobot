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
   mkdir -p models
   ```
   `docker-compose.yml` stores the database, long-term memory index, and
   assets in a Docker-managed named volume (`data`), not a `./data` bind
   mount: a bind mount that Docker has to create fresh is root-owned, which
   the container's non-root `appuser` cannot write to, so the container
   would enter a restart loop on `init_storage()`'s first write. The manual
   `docker run` commands below still bind-mount `./data`, so create it
   first if you use those instead of Compose: `mkdir -p data`.

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

**Startup failures fail fast:** the process opens `companion_database.db`,
`longterm_memory/`, and `assets/` under `COMPANION_DATA_DIR`, which the
images above set to `/app/data` (the mounted `data` volume). The container
exits with a non-zero status if that directory is not writable by
`appuser`, instead of logging a warning and serving 500s indefinitely. With
`restart: unless-stopped` this shows up as a visible restart loop rather
than a silently broken container; `docker logs` names the exact path (e.g.
`/app/data/companion_database.db` or `/app/data/longterm_memory`) that
could not be opened.

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
COMPANION_DATA_DIR=.            # Directory for the database, long-term memory index, and
                                 # assets (default: the working directory)
AI_COMPANION_WORKERS=4          # actix worker thread count (default: available_parallelism)
RUST_LOG=info                   # Logging level (debug, info, warn, error)

# Docker-specific
NVIDIA_VISIBLE_DEVICES=all      # GPU visibility for CUDA
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Running two instances on one machine:** point each instance at its own
port and data directory so they do not contend for the same database or
long-term memory index:

```bash
COMPANION_PORT=3100 COMPANION_DATA_DIR=./instance-b ./ai-companion
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

## Multi-instance chat (host and joiners)

AI Companion can run as several instances that share one conversation: one
instance is the **host**, the others **join** it. This is a distinct feature
from the "two instances on one machine" scenario above (two unrelated,
unconnected chats) — here every instance is part of the same chat.

### How it works

One instance is the host and owns the chat: every message and reply is
generated and persisted there. Each joiner runs its own model and its own
character card, and connects outbound to the host over a WebSocket at
`/api/multiplayer/ws`; when it is a joiner's bot's turn to speak, the host
asks it over that connection and the joiner generates with its own model. A
user talks to the host's UI as normal; a joiner's UI shows a read-only
mirror of the same conversation, with its own settings dialog reporting its
connection state. Every instance — host and every joiner — still needs its
own GGUF model loaded; a joiner's bot never runs on the host's model.

### Host setup

1. Open the host instance's Settings dialog → Multiplayer, set the mode to
   `host`, and set a password. This is the shared secret joiners use to
   authenticate; it is never sent in the clear (see "Security model" below).
2. Note the host's listening port (`COMPANION_PORT`, default `3000`) and
   make sure it is reachable from every joiner's machine. On a firewalled
   host:
   ```bash
   ufw allow from <joiner-ip> to any port 3000
   ```
3. **Switching `multiplayer_mode` takes effect after a restart**, not
   immediately: `PUT /api/config` saves the new mode, but the running
   process keeps its old role (and, for a joiner, its old identity) until
   it is restarted. Restart the instance after changing the mode.

### Joiner setup

1. Open the joiner instance's Settings dialog → Multiplayer, set the mode
   to `joiner`, and fill in:
   - **Host address**: `host-ip:3000` — a bare `host:port`, no `ws://` or
     `http://` scheme; the client builds the WebSocket URL itself.
   - **Participant ID**: this instance's id in the chat, matching
     `^[a-z][a-z0-9_]{0,15}$` (1-16 lowercase letters, digits or `_`,
     starting with a letter), e.g. `bot1`. Must be unique among everyone
     already connected.
   - **Password**: the same password the host set.
2. The joiner's own companion card supplies its display name and avatar,
   sent to the host as part of the join handshake — there is nothing
   separate to configure for that.
3. As with the host, a mode change only takes effect after a restart.
4. Once restarted, the joiner connects automatically. Its connection state
   (`disconnected`, `connecting`, `connected`, or `rejected`) is shown in
   its own settings dialog and available at `GET /api/multiplayer/status`.
   If the connection drops for any reason other than a rejected handshake,
   the joiner reconnects on its own, backing off from 1 second up to 30
   seconds between attempts.
5. A joiner answers `GenerateRequest`s with its own model, its own card,
   its own dialogue tuning and its own long-term memory — only the
   transcript and the roster of who else is in the chat come from the host.

### Two instances on one machine

The "Running two instances on one machine" snippet under Environment
Variables above (`COMPANION_PORT`/`COMPANION_DATA_DIR`) works for a
host/joiner pair too — each still needs its own port and data directory so
they do not contend for the same database or long-term memory index. Point
the joiner's host address at `127.0.0.1:3000` (the first instance's port).

Running a host and a joiner this way keeps two full models resident at
once: budget RAM/VRAM for both. The host can free its own model with
`POST /api/llm/unload` when it is not generating, but a joiner needs its
model loaded whenever it might be asked to speak, so unloading it defeats
the point.

### Docker Compose

```bash
docker compose --profile cpu --profile multiplayer up -d
```

This starts the CPU host (`ai-companion-cpu`, `http://localhost:3000`) and
a joiner (`ai-companion-joiner`, `http://localhost:3001`) from the same
compose file. Inside the compose network, the joiner's host address is the
host's **service name and container port**, not the published port:
`ai-companion-cpu:3000`. Configure both instances' Settings dialogs exactly
as in "Host setup" and "Joiner setup" above — see `docker-compose.yml`'s
`multiplayer` profile for the service definition.

### Security model

**The password protects only the join handshake.** It authenticates a
joiner to the host with an HMAC-SHA256 challenge/response; the password
itself never crosses the wire, only proof that both sides know it. It does
**not** protect anything else:

- Every REST route on every instance (host or joiner) is unauthenticated —
  anyone who can reach the port can read and write the chat, its
  configuration, and its character card, with or without multiplayer
  enabled.
- The multiplayer WebSocket is plaintext `ws://`. The transcript, every
  generated token, and the join handshake's nonce and proof are all
  readable to anyone on the network path between host and joiner.
- The `nginx` example under Production Deployment below already forwards
  the `Upgrade` headers a WebSocket needs, but putting TLS in front of a
  joiner's *outbound* `ws://` connection to the host is out of scope here
  and not configured by anything in this repository.

Assume a trusted LAN or VPN for multi-instance chat. Never expose a host's
port, or a joiner's outbound connection, to a public, untrusted network.

### Troubleshooting

- **`Rejected` at join**: the password is wrong, the participant ID is
  already connected or reserved (`user`/`char`), or the two instances are
  running different protocol versions.
- **`429 Too Many Requests` on `/api/multiplayer/ws`**: the host throttles
  an address after 5 failed join attempts within a 10-minute window.
- **`404` on `/api/multiplayer/ws`**: the instance you connected to is not
  currently in `host` mode (including a `host` mode saved but not yet
  applied by a restart).
- **A bot is listed but "did not respond"**: the round timed out waiting
  for that joiner (`remote_generation_timeout_secs`, default 120 seconds),
  or the joiner had no model loaded when the host asked it to speak.

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
