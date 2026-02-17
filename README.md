# Elo Server (Python AI Backend)

This is the Python-based backend for Elocuency, responsible for handling AI interactions via **LangChain** and **FastAPI**.

It follows **Hexagonal Architecture** and uses **snake_case** per PEP 8 standards.

## Prerequisites

- Python 3.10+
- `pip`

## Installation

1.  **Navigate to the server directory**:

    ```bash
    cd apps/elo-server
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

2.  Open `.env` and set your Google Gemini API Key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    PORT=8000
    ```

## Running the Web Server

The web server exposes a **FastAPI** application with REST endpoints and a **LangServe Playground** for interactive chat.

### Option 1: Via `python`

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/main.py
```

Esto arranca uvicorn internamente usando la configuración de `config.yaml` (host, port, reload).

### Option 2: Via `uvicorn` directamente

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn src.main:app --reload
```

El servidor estará disponible en `http://localhost:8000` (puerto por defecto).

---

## Running the CLI (Chat Interactivo)

El CLI permite interactuar con el agente de IA directamente desde el terminal, sin necesidad de levantar el servidor web.

```bash
./cli.sh
```

O manualmente:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
./venv/bin/python src/scripts/cli.py
```

Al iniciar, el CLI:

1. Carga la configuración desde `config.yaml`.
2. Inicializa el **MCP Manager** y carga las herramientas (MCP + local).
3. Solicita un **User ID** (o genera uno aleatorio).
4. Abre un bucle de chat interactivo. Escribe `exit` o `quit` para salir.

---

## LangServe Playground

El proyecto incluye un **playground web** proporcionado por [LangServe](https://github.com/langchain-ai/langserve) que permite probar el agente de IA desde el navegador.

### Cómo acceder

1. **Levanta el servidor web** (ver sección anterior).
2. Abre en el navegador una de las siguientes URLs:

| Playground         | URL                                              | Descripción                                  |
| ------------------ | ------------------------------------------------ | -------------------------------------------- |
| **Agent completo** | `http://localhost:8000/agent/playground/`        | Agente LangGraph con tools, memoria y MCP    |
| **Simple Agent**   | `http://localhost:8000/simple-agent/playground/` | Modelo Gemini directo (sin tools ni memoria) |

> **Nota:** La barra final `/` en la URL es obligatoria.

El playground ofrece una interfaz de chat donde puedes enviar mensajes y ver las respuestas del agente en tiempo real, incluyendo las llamadas a herramientas (tool calls) cuando el agente las utiliza.

---

## API Endpoints

- **Health Check**: `GET /health`
- **Ask AI**: `POST /ask`
  - Body: `{"prompt": "Your question here", "user_id": "optional_id"}`
