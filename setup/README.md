# Elo Server Distribution

This folder contains everything you need to run **Elo Server** easily on Linux, macOS, and Windows (via WSL2).

## Prerequisites

- **Docker Desktop** or **Docker Engine** installed and running.
- **Git** (recommended to clone the repository).

### Windows Users

Please ensure you are running these commands inside **WSL2** (Windows Subsystem for Linux).

1.  Install WSL2: `wsl --install` in PowerShell.
2.  Install Docker Desktop and enable WSL2 integration.
3.  Open your WSL terminal (e.g., Ubuntu).

## Quick Start

1.  Open your terminal in this directory (`install/`).
2.  Run the setup script:

    ```bash
    ./elo-setup.sh
    ```

3.  Follow the interactive prompts to configure your API keys and Vault path.

The script will:

- Check for Docker.
- Create/Update your `.env` configuration file.
- Build the local Docker images.
- Start the server and automation engine.

## Configuration

Your configuration is stored in `.env` inside this directory. You can edit it manually if needed.
To add new tools or modify agent settings, edit `elo.config.json` or add tool scripts to the `workspace/` folder (created after first run).

## Updating

To update the server (if you pulled new code):

1.  Run `./elo-setup.sh` again. It will rebuild the images.

## Troubleshooting

- **Permissions**: If you get permission errors on Linux, ensure your user is in the `docker` group (`sudo usermod -aG docker $USER`) or run with `sudo`.
- **Ports**: The server uses ports `8001` (API) and `5678` (n8n). Ensure they are free.
