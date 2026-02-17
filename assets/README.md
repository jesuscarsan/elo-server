# Assets Directory

This directory contains version-controlled resources that are used by the applications in this repository.

Unlike the `/workspace` folder, which is for local, dynamic, and user-specific data (and is ignored by Git), the `/assets` folder is the **Source of Truth** for:

- **LangChain Tools**: Base and "official" Python tools used by the AI.
- **n8n Workflows**: Standard automation workflows exported as JSON.
- **Templates & Static Configs**: Any other configuration that should be shared across the team/deployments.

## Relationship with `/workspace`

To use these assets at runtime, you can:

1.  **Symlink**: Create a symbolic link from `assets/path/to/extra` to `workspace/path/to/extra`.
2.  **Copy**: Create a setup script that copies files from `assets/` to `workspace/`.
3.  **Application Logic**: Configure the apps (like `elo-server`) to read from both `assets/` and `workspace/`.

---

_Note: This folder is tracked by Git. Do not store sensitive data (API keys, personal databases) here._
