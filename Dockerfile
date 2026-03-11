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
# Stage 3: Final Runtime (PyTorch Base)
# ==========================================
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LITE_MODE=false \
    PATH="/usr/lib/google-cloud-sdk/bin:$PATH" \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    gnupg \
    ca-certificates \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
RUN pip install uv
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# 3. Copy Python Application Code
COPY . .

# 4. Install Built Artifacts
COPY --from=backend-builder /app/backend/vchat-server /usr/local/bin/vchat-server
RUN mkdir -p /usr/share/vchat/static
COPY --from=frontend-builder /app/frontend/dist /usr/share/vchat/static

# 5. Setup Entrypoint (Fix Windows Line Endings Here)
COPY start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /usr/local/bin/start.sh && \
    chmod +x /usr/local/bin/start.sh

# Expose the Go Server port
EXPOSE 8080

# Run the Orchestrator
CMD ["/usr/local/bin/start.sh"]