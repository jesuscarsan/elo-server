#!/bin/bash

# Load environment variables
if [ -f .env.n8n ]; then
    export $(grep -v '^#' .env.n8n | xargs)
fi

echo "Starting n8n with flat-file storage in workspace/n8n..."
./node_modules/.bin/n8n start
