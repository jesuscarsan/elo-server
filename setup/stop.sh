#!/bin/bash

# Change to the directory of the script
cd "$(dirname "$0")"

echo "🛑 Stopping Elo-Server containers..."
docker compose down

echo "✅ All services stopped."
