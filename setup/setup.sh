#!/bin/bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Calculate Monorepo Root (Assuming setup is in apps/elo-server/setup)
MONOREPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}      Elo Server Setup (Docker)         ${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. Check Dependencies
echo -e "\n${YELLOW}[1/5] Checking dependencies...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: 'docker' is not installed.${NC}"
    echo "Please install Docker Desktop or Docker Engine first."
    exit 1
fi

DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}Error: 'docker compose' is not installed.${NC}"
    exit 1
fi

echo "Docker is available: $(${DOCKER_COMPOSE_CMD} version)"

# 2. Security Check (.gitignore)
if ! grep -q "^\.env" ../.gitignore 2>/dev/null; then
    echo -e "\n${YELLOW}[2/5] Hardening security (.gitignore)...${NC}"
    echo -e ".env\n.env.*" >> ../.gitignore
    echo "  - Added .env to .gitignore."
fi

# 3. Environment Configuration
echo -e "\n${YELLOW}[3/5] Configuring environment...${NC}"

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    touch "$ENV_FILE"
fi

prompt_var() {
    local var_name=$1
    local prompt_text=$2
    local default_value=$3
    local is_sensitive=${4:-false}
    local current_value
    local input_value

    current_value=$(grep "^${var_name}=" "$ENV_FILE" | cut -d '=' -f2-)

    if [ -n "$current_value" ]; then
        if [ "$is_sensitive" = "true" ]; then
            echo -ne "  $prompt_text (already set, enter to keep hidden value, or type new): "
        else
            echo -ne "  $prompt_text [$current_value]: "
        fi
    else
        echo -ne "  $prompt_text"
        if [ -n "$default_value" ] && [ "$is_sensitive" = "false" ]; then
            echo -ne " [$default_value]"
        fi
        echo -ne ": "
    fi

    if [ "$is_sensitive" = "true" ]; then
        read -rs input_value
        echo "" # New line after hidden input
    else
        read -r input_value
    fi

    if [ -z "$input_value" ]; then
        if [ -n "$current_value" ]; then
            input_value="$current_value"
        elif [ -n "$default_value" ]; then
            input_value="$default_value"
        fi
    fi

    if [ -n "$input_value" ]; then
        if grep -q "^${var_name}=" "$ENV_FILE"; then
            sed -i.bak "s|^${var_name}=.*|${var_name}=${input_value}|" "$ENV_FILE" && rm "${ENV_FILE}.bak"
        else
            echo "$var_name=$input_value" >> "$ENV_FILE"
        fi
    else
         echo -e "${RED}  ! Warning: $var_name remains empty.${NC}"
    fi
}

prompt_var "GOOGLE_API_KEY" "Enter Google API Key" "" "true"
prompt_var "OBSIDIAN_API_KEY" "Enter Obsidian API Key" "obsidian-secret-key" "true"
prompt_var "OBSIDIAN_URL" "Enter Obsidian URL" "http://host.docker.internal:27124"
prompt_var "NGROK_AUTHTOKEN" "Enter Ngrok Authtoken" "" "true"

# Workspace Handling
prompt_var "ELO_WORKSPACE_PATH" "Enter absolute path to ELO Workspace" "${MONOREPO_ROOT}/elo-workspace"

WS_PATH=$(grep "^ELO_WORKSPACE_PATH=" "$ENV_FILE" | cut -d '=' -f2-)
if [ -z "$WS_PATH" ]; then WS_PATH="${MONOREPO_ROOT}/elo-workspace"; fi

# Migration: If setup/workspace exists, move it to elo-workspace
if [ -d "workspace" ] && [ ! -d "$WS_PATH" ]; then
    echo -e "\n${YELLOW}Migrating: Moving existing 'workspace/' to '$WS_PATH'...${NC}"
    mkdir -p "$(dirname "$WS_PATH")"
    mv workspace "$WS_PATH"
fi

prompt_var "VAULT_PATH" "Enter absolute path to your Obsidian Vault" ""

# 4. Environment Selection
echo -e "\n${YELLOW}[4/5] Select environment...${NC}"
echo "  1) Development (with hot-reload, code volume mount)"
echo "  2) Production (stable execution)"
echo -ne "Choose [1/2]: "
read -r env_choice

COMPOSE_FILES="-f docker-compose.yml"
if [ "$env_choice" == "1" ]; then
    echo "Setting up for DEVELOPMENT..."
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.dev.yml"
else
    echo "Setting up for PRODUCTION..."
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
fi

# Prepare folders in the workspace
echo "Preparing folders in $WS_PATH..."
mkdir -p "$WS_PATH/logs/caddy" "$WS_PATH/logs/elo-server" 
mkdir -p "$WS_PATH/n8n/workflows" "$WS_PATH/n8n/data"
mkdir -p "$WS_PATH/caddy_data" "$WS_PATH/caddy_config"

# 5. Start Services
echo -e "\n${YELLOW}[5/5] Starting services...${NC}"
export ELO_WORKSPACE_PATH="$WS_PATH"

# Modified entrypoint logic for n8n webhooks through Caddy
# We need to ensure WEBHOOK_URL ends with /n8n/
${DOCKER_COMPOSE_CMD} ${COMPOSE_FILES} up -d --build

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Success! Elo Server is running via Caddy.${NC}"
    if [ "$env_choice" == "1" ]; then 
        echo -e "${YELLOW}Mode: Development${NC}"
    else
        echo -e "${YELLOW}Mode: Production${NC}"
    fi
    echo "  - Access UI/API: http://localhost/"
    echo "  - n8n Automation: http://localhost/n8n/"
    echo "  - Workspace: $WS_PATH"
    echo "  - Logs: tail -f $WS_PATH/logs/elo-server.log"
else
    echo -e "${RED}Failed to start services.${NC}"
    exit 1
fi
