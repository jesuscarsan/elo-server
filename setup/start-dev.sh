#!/bin/bash

# Change to the directory of the script to ensure paths are correct
cd "$(dirname "$0")"

# Auto-trigger setup if .env is missing
if [ ! -f ".env" ]; then
    echo "⚠️  Configuration missing (.env not found). Running setup.sh first..."
    ./setup.sh
    # setup.sh already starts the services, so we can exit here
    exit 0
fi

echo "🛠️  Starting Elo-Server in DEVELOPMENT mode (hot-reload enabled)..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

echo "✅ Development environment started."
docker compose ps
