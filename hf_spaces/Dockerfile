# ==========================================
# Stage 1: Build Frontend (React/TS/Vite)
# ==========================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

# Copy frontend definitions
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy source and build
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Build Backend (Golang)
# ==========================================
FROM golang:1.23 AS backend-builder
WORKDIR /app/backend

# Copy Go source
COPY main.go .

# Build static binary
RUN go mod init vchat-server && \
    go mod tidy && \
    CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o vchat-server main.go

# ==========================================
# Stage 3: Final Runtime (Hugging Face Space)
# ==========================================
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Default to LITE_MODE=true for HF Spaces (API Only)
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LITE_MODE=true \
    PATH="/home/user/.local/bin:$PATH" \
    PIP_NO_CACHE_DIR=1

# Create a non-root user (Required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# 1. Fix FFmpeg Conflict (Critical Step)
RUN conda uninstall -y ffmpeg || true

# 2. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python Dependencies
RUN pip install uv
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt
# Explicitly force latest yt-dlp to handle Twitter/X API changes
RUN uv pip install --system --upgrade "yt-dlp[default]"

# 4. Copy Python Application Code
COPY --chown=user src/ ./src/

# 5. Install Built Artifacts
COPY --from=backend-builder --chown=user /app/backend/vchat-server /app/vchat-server
RUN mkdir -p /app/static
COPY --from=frontend-builder --chown=user /app/frontend/dist /app/static

# 6. Setup Directories and Permissions
RUN mkdir -p /app/data /app/data/videos /app/data/labels /app/data/prompts /app/data/responses /app/metadata \
    && chown -R user:user /app/data /app/metadata

# 7. Setup Entrypoint
COPY --chown=user start.sh /app/start.sh
RUN sed -i 's/\r$//' /app/start.sh && \
    chmod +x /app/start.sh

# Switch to non-root user
USER user

# Expose the HF Space port
EXPOSE 7860

# Run the Orchestrator
CMD ["/app/start.sh"]