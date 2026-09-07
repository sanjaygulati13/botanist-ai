#!/bin/bash

# --- Configuration ---
APP_NAME="botanist-ai-dashboard"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$APP_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
# whoami returns "root" when run under sudo; SUDO_USER is the real user
# the service should run as.
USER_NAME=${SUDO_USER:-$(whoami)}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- 1. Check for Root (Required for systemd) ---
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root (use sudo) to configure systemd."
fi

# --- 2. Setup Backend Environment ---
log "Setting up backend environment..."
cd "$BACKEND_DIR" || error "Could not enter backend directory: $BACKEND_DIR"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || error "Failed to create virtual environment."
fi

log "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt" || error "Failed to install dependencies."

# Port comes from .env when set there, otherwise the default.
PORT="$(grep -E '^PORT=' "$APP_DIR/.env" 2>/dev/null | head -n1 | cut -d= -f2)"
PORT="${PORT:-4004}"

# --- 3. Create Systemd Service File ---
log "Configuring systemd service..."
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Botanist AI Dashboard (FastAPI backend)
After=network.target

[Service]
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$BACKEND_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
StandardOutput=append:$BACKEND_DIR/server.log
StandardError=append:$BACKEND_DIR/server.log

[Install]
WantedBy=multi-user.target
EOF

# --- 4. Start/Restart Service ---
log "Reloading systemd and starting service..."
systemctl daemon-reload
systemctl enable "$APP_NAME"
systemctl restart "$APP_NAME"

# --- 5. Verification ---
sleep 2
if systemctl is-active --quiet "$APP_NAME"; then
    log "Deployment successful! Service is running."
    log "Check logs with: journalctl -u $APP_NAME -f"
    log "Check app logs at: $BACKEND_DIR/server.log"
else
    error "Service failed to start. Check 'journalctl -u $APP_NAME' for details."
fi
