# Cogitate
A Secure, Collaborative, Multi-Task, Multi-Agent AI System

Copyright (c) 2025 Moonlight Technologies Inc. All rights reserved.

# Overview
A scalable AI system featuring orchestration by LangGraph, a FastAPI backend, a React frontend, Redis Pub/Sub for real-time updates, PostgreSQL for state persistence, and Jaeger for tracing. It implements Human-in-the-Loop for critical actions and OWASP LLM security practices.

The system introduces a user role system with two roles: administrator and operator. Administrators manage users and system settings, while operators use their personal API keys to run multi-agent conferences.

# Key Implemented Features

*   **Human-in-the-Loop Workflow:**
    *   The agent collaboration process is guided by human supervision and direction.
    *   **Stage 1 (Brainstorming):** All agents generate answers to the initial query.
    *   **Stage 2 (Role Assignment & Instruction):** The user reviews each agent's response and assigns roles such as "Drafter," "Verifier," and "Judge." The user can also provide specific instructions (an Elaboration Query) for the next step.
    *   **Stage 3 (Execution):** Agents execute their next tasks based on their assigned roles and the user's instructions.

*   **Multi-language Support:**
    *   The response language of the agents automatically follows the UI language (Japanese or English) selected in the frontend, allowing users to interact with the AI in their preferred language.

*   **Dynamic LLM Provider Support:**
    *   Eliminated the hard-coded dependency on OpenAI and now supports multiple major providers, including Gemini, Grok, and Claude.
    *   The `langchain` integrations for each provider (e.g., `langchain-google-genai`, `langchain-anthropic`) are managed in `requirements.txt`.
    *   Currently supported providers are "ChatGPT", "Gemini", "Grok", "Claude", and "Local Model" (via Ollama).

*   **Local LLM Support:**
    *   Implemented local Retrieval-Augmented Generation (RAG) functionality using `HuggingFaceEmbeddings`.
    *   Supports local model inference via Ollama (`ChatOllama`).

*   **UI/UX Improvements:**
    *   **Real-time Streaming:** Responses from LLMs are now streamed in real-time with a typewriter effect, significantly improving the perceived speed of the application.
    *   Agent settings and queries are saved in `localStorage`, preserving state even when the page is reloaded.
    *   Agent cards are now displayed in a responsive grid layout that wraps horizontally.
    *   Changed the query input field to a multi-line `<textarea>`.
    *   Removed copyright notices from individual file headers and added a global application footer.

*   **Security:**
    *   The security feature for detecting prompt injection is now an optional feature that can be enabled by setting the `INSPECTOR_LLM_API_KEY` environment variable.
    *   The local `multiagent.db` database file is now excluded from version control by `.gitignore`.

*   **Bug Fixes:**
    *   Fixed a `TypeError` when serializing Pydantic models to JSON in the backend, ensuring correct WebSocket communication.
    *   Corrected the WebSocket proxy settings in the development environment.
    *   Ensured that agent permission checks are correctly based on the LLM provider, not the agent name.

# Setup Instructions

## 1. Prerequisites

**Required Tools:**

*   Docker (24.0.5 or higher)
*   Docker Compose (2.20.2 or higher)
*   Node.js (18.x)
*   Python (3.11)
*   Kubernetes (kubectl 1.28 or higher, minikube or a cloud provider's cluster)

**API Keys:**

*   xAI API Key: Obtain from the xAI Console ([https://console.x.ai/team/default/api-keys]).
*   Google API Key: Obtain from the Google Cloud Console.

**Environment:**
Linux/macOS/Windows (WSL2 recommended).

## 2. Environment Variable Setup

Create a `backend/.env` file. The `ENCRYPTION_KEY` is a 32-byte key used to encrypt operator API keys.

**How to generate `ENCRYPTION_KEY`:**
```python
# Run the following in a Python interpreter to generate a key
from cryptography.fernet import Fernet
Fernet.generate_key()
```

**Contents of the `.env` file:**
```
# backend/.env
# OPENAI_API_KEY=your_openai_api_key  # No longer needed
XAI_API_KEY=your_xai_api_key
GOOGLE_API_KEY=your_google_api_key
REDIS_URL=redis://redis:6379/0
JWT_SECRET_KEY=a_very_secret_key_that_is_long_and_secure
ENCRYPTION_KEY=your_generated_32_byte_encryption_key
JAEGER_HOST=jaeger:4317
# Connect to the database via host port mapping for local development
DB_URL=postgresql://user:password@localhost:5433/cogitatedb
# For the optional security feature
# INSPECTOR_LLM_API_KEY=your_openai_api_key
```

**Exporting Environment Variables (Optional):**
```bash
export $(cat backend/.env | xargs)
```

### LLM Provider Settings

This application supports multiple LLM providers. The requirements for each are as follows:

*   **ChatGPT (`chatgpt`)**:
    *   Requires the `langchain-openai` package. You can install it with `pip install langchain-openai`.
    *   Requires a valid OpenAI API key that can be added from the UI.

*   **Gemini (`gemini`)**:
    *   Requires the `langchain-google-genai` package.
    *   Requires a valid Google API key to be added via the UI.

*   **Grok (`grok`)**:
    *   Requires the `xai-sdk` package.
    *   Requires a valid xAI API key to be added via the UI.

*   **Claude (`claude`)**:
    *   Requires the `langchain-anthropic` package.
    *   Requires a valid Anthropic API key to be added via the UI.

*   **Local Model (`local`)**:
    *   This option uses a local model provided by [Ollama](https://ollama.com/).
    *   Requires the `langchain-community` and `ollama` packages.
    *   The Ollama service must be running on your machine. The application will attempt to connect at `http://ollama:11434`.
    *   No API key is required.

*   **Security Inspector (Optional)**:
    *   The optional prompt injection security feature requires an OpenAI API key.
    *   To enable it, set the `INSPECTOR_LLM_API_KEY` environment variable in your `.env` file.

## 3. Installing Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

## 4. Running Locally

Start all services with Docker Compose. The `--profile "*"` flag is required to start all defined services, including `backend` and `frontend`.
```bash
sudo docker compose --profile "*" up --build
```
**Verifying Services:**

*   Frontend: `http://localhost:3000`
*   Backend: `http://localhost:8000/docs` (FastAPI Swagger UI)
*   Jaeger: `http://localhost:16686` (For viewing traces)

**Application Usage:**

1.  Access the frontend (`http://localhost:3000`).
2.  Create a new user account on the **Register** page. The first user automatically gets the **administrator** role.
3.  **Login** with the created account.
4.  If you are logged in as an **operator**, you can set and save your LLM API keys from the UI.
5.  In the main **Meeting View**, add and configure agents, enter a query, and start the conference.

## 5. Verification

**Local Tests:**

Run unit and integration tests for the core logic using `pytest`.
```bash
# Run from the backend directory
python -m pytest
```

**LLM Provider Connectivity Test:**

A script is provided to verify the connection to each supported LLM provider.

*   **Prerequisites:** You must set the API keys for the providers you wish to test in the `backend/.env` file.
    *   `OPENAI_API_KEY`: For ChatGPT
    *   `GOOGLE_API_KEY`: For Gemini
    *   `XAI_API_KEY`: For Grok
    *   `ANTHROPIC_API_KEY`: For Claude
*   **Execution:**
    ```bash
    # Run from the backend directory
    python scripts/test_llm_providers.py
    ```
```bash
# Run from the frontend directory
npm test
```

## 6. Migrating to Production

**Kubernetes Deployment**

Create secrets.
```bash
kubectl create secret generic app-secrets \
--from-literal=XAI_API_KEY=your_xai_api_key \
--from-literal=GOOGLE_API_KEY=your_google_api_key \
--from-literal=JWT_SECRET_KEY=your_kubernetes_secret_key \
--from-literal=ENCRYPTION_KEY=your_kubernetes_encryption_key
```

## 7. Security Measures
*   **OWASP LLM01 (Prompt Injection):** The `inspect_input` feature, which uses an LLM to detect prompt injection, is available as an optional feature. Set `INSPECTOR_LLM_API_KEY` to enable it.
*   **OWASP LLM02 (Handling Malicious Output):** The `sanitize_output` function prevents XSS/RCE.
*   **OWASP LLM06 (Disclosure of Sensitive Information):** The `human_approval` node requires human intervention at critical steps.
*   **API Key Management:** User-specific API keys are stored in the database using AES-256-GCM encryption (via `cryptography.fernet`). The encryption/decryption of keys depends on the `ENCRYPTION_KEY`. In a production environment, please manage this key securely, for example, with Kubernetes Secrets.