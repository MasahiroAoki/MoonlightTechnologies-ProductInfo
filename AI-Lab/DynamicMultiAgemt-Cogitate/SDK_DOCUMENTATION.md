# Backend SDK Documentation

Welcome to the Backend SDK documentation. This document provides all the information you need to interact with the backend API.

## Authentication

Authentication is handled via JWT (JSON Web Tokens). To access protected endpoints, you need to obtain an access token.

### Register a new user

Creates a new user account. The first user to register automatically becomes an administrator.

- **Endpoint:** `POST /auth/register`
- **Status Code:** `201 CREATED`
- **Request Body:**
  ```json
  {
    "username": "your_username",
    "email": "user@example.com",
    "password": "your_password"
  }
  ```
- **Response Body:**
  ```json
  {
    "id": "user_id",
    "username": "your_username",
    "email": "user@example.com",
    "role": "operator"
  }
  ```
- **Example with `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "newuser@example.com",
    "password": "a_secure_password"
  }'
  ```

### Log in and get an access token

Authenticates a user and returns an access token, refresh token, and user role.

- **Endpoint:** `POST /auth/token`
- **Request Body:** `x-www-form-urlencoded`
  ```
  username=your_username&password=your_password
  ```
- **Response Body:**
  ```json
  {
    "access_token": "your_access_token",
    "refresh_token": "your_refresh_token",
    "token_type": "bearer",
    "role": "operator"
  }
  ```
- **Example with `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=newuser&password=a_secure_password"
  ```

### How to Call the API (Making Authenticated Requests)

Once you have an access token, you must include it in the `Authorization` header for all subsequent requests to protected endpoints. The scheme is `Bearer`.

**Example Header:**
`Authorization: Bearer <your_access_token>`

You can save the token to a shell variable for easier use in `curl` commands. This example uses `jq` to parse the JSON response.

```bash
# Save the token from the login response
export TOKEN=$(curl -s -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=newuser&password=a_secure_password" | jq -r .access_token)

# Check if the token was saved
echo $TOKEN

# Use the token to access a protected endpoint
curl -X GET "http://localhost:8000/users/me" \
  -H "Authorization: Bearer $TOKEN"
```

## Users

These endpoints are available to any authenticated user.

### Get my profile

Retrieves the profile of the currently authenticated user.

- **Endpoint:** `GET /users/me`
- **Authentication:** Bearer Token required.
- **Response Body:**
  ```json
  {
    "id": "user_id",
    "username": "your_username",
    "email": "user@example.com",
    "role": "operator"
  }
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable as shown in the "How to Call the API" section
  curl -X GET "http://localhost:8000/users/me" \
    -H "Authorization: Bearer $TOKEN"
  ```

### Update my profile

Updates the email of the currently authenticated user.

- **Endpoint:** `PUT /users/me`
- **Authentication:** Bearer Token required.
- **Request Body:**
  ```json
  {
    "email": "new_email@example.com"
  }
  ```
- **Response Body:**
  ```json
  {
    "id": "user_id",
    "username": "your_username",
    "email": "new_email@example.com",
    "role": "operator"
  }
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable
  curl -X PUT "http://localhost:8000/users/me" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"email": "new_user_email@example.com"}'
  ```

### Add an API key

Adds a new API key to the user's vault. The key is encrypted before being stored.

- **Endpoint:** `POST /users/me/keys`
- **Authentication:** Bearer Token required.
- **Status Code:** `201 CREATED`
- **Request Body:**
  ```json
  {
    "key_name": "My OpenAI Key",
    "api_key": "sk-...",
    "provider": "OpenAI"
  }
  ```
- **Response Body:**
  ```json
  {
    "id": "key_id",
    "key_name": "My OpenAI Key",
    "provider": "OpenAI"
  }
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable
  curl -X POST "http://localhost:8000/users/me/keys" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "key_name": "My Personal OpenAI Key",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "provider": "OpenAI"
    }'
  ```

### List my API keys

Lists all API keys in the user's vault.

- **Endpoint:** `GET /users/me/keys`
- **Authentication:** Bearer Token required.
- **Response Body:**
  ```json
  [
    {
      "id": "key_id",
      "key_name": "My OpenAI Key",
      "provider": "OpenAI"
    }
  ]
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable
  curl -X GET "http://localhost:8000/users/me/keys" \
    -H "Authorization: Bearer $TOKEN"
  ```

### Delete an API key

Deletes a specific API key from the user's vault.

- **Endpoint:** `DELETE /users/me/keys/{key_id}`
- **Authentication:** Bearer Token required.
- **Status Code:** `204 NO CONTENT`
- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable
  # First, get the ID of the key you want to delete (e.g., from the list keys endpoint)
  KEY_ID_TO_DELETE="some-key-id"

  curl -X DELETE "http://localhost:8000/users/me/keys/$KEY_ID_TO_DELETE" \
    -H "Authorization: Bearer $TOKEN"
  ```

## Ollama (Admin only)

These endpoints are for managing Ollama models and require administrator privileges.

### Get local models

Retrieves a list of locally available Ollama models.

- **Endpoint:** `GET /api/ollama/models`
- **Authentication:** Bearer Token required (Admin role).
- **Response Body:**
  ```json
  [
    {
      "name": "llama2:latest",
      "modified_at": "2023-11-25T14:00:00.000Z",
      "size": 7700000000
    }
  ]
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  curl -X GET "http://localhost:8000/api/ollama/models" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

### Pull a new model

Pulls a new model from the Ollama Hub. This is a long-running operation performed in the background.

- **Endpoint:** `POST /api/ollama/pull`
- **Authentication:** Bearer Token required (Admin role).
- **Status Code:** `202 ACCEPTED`
- **Request Body:**
  ```json
  {
    "model_name": "name-of-model-to-pull"
  }
  ```
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  curl -X POST "http://localhost:8000/api/ollama/pull" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model_name": "gemma:2b"}'
  ```

## Admin (Admin only)

These endpoints are for system administration and require administrator privileges.

### User Management

#### Get all users
Retrieves a list of all users in the system.

- **Endpoint:** `GET /admin/users`
- **Authentication:** Bearer Token required (Admin role).
- **Response Body:** A list of user objects.
- **Example with `curl`:**
  ```bash
  # Note: To get an ADMIN_TOKEN, you must log in as a user with the 'administrator' role.
  # The first user to register automatically receives this role.
  export ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8000/auth/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=adminuser&password=admin_password" | jq -r .access_token)

  # Use the admin token to get the list of all users
  curl -X GET "http://localhost:8000/admin/users" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

#### Update a user's role
Updates the role of a specific user.

- **Endpoint:** `PUT /admin/users/{user_id}`
- **Authentication:** Bearer Token required (Admin role).
- **Path Parameter:** `user_id` (string)
- **Query Parameter:** `role` (string, e.g., "administrator", "operator")
- **Response Body:** The updated user object.
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  # Set the ID of the user you want to update
  USER_ID_TO_UPDATE="some-user-id"

  curl -X PUT "http://localhost:8000/admin/users/$USER_ID_TO_UPDATE?role=operator" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

#### Delete a user
Deletes a specific user from the system.

- **Endpoint:** `DELETE /admin/users/{user_id}`
- **Authentication:** Bearer Token required (Admin role).
- **Path Parameter:** `user_id` (string)
- **Status Code:** `204 NO CONTENT`
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  # Set the ID of the user you want to delete
  USER_ID_TO_DELETE="some-user-id"

  curl -X DELETE "http://localhost:8000/admin/users/$USER_ID_TO_DELETE" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

### Assignable LLM Management

#### Create an assignable LLM
Adds a new LLM that can be assigned to users.

- **Endpoint:** `POST /admin/llms`
- **Authentication:** Bearer Token required (Admin role).
- **Request Body:**
  ```json
  {
    "model_name": "ollama/llama2"
  }
  ```
- **Response Body:** The created LLM object.
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  curl -X POST "http://localhost:8000/admin/llms" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model_name": "ollama/gemma:2b"}'
  ```

#### Get all assignable LLMs
Retrieves a list of all LLMs that can be assigned to users.

- **Endpoint:** `GET /admin/llms`
- **Authentication:** Bearer Token required (Admin role).
- **Response Body:** A list of assignable LLM objects.
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  curl -X GET "http://localhost:8000/admin/llms" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

#### Delete an assignable LLM
Deletes an assignable LLM.

- **Endpoint:** `DELETE /admin/llms/{llm_id}`
- **Authentication:** Bearer Token required (Admin role).
- **Path Parameter:** `llm_id` (string)
- **Status Code:** `204 NO CONTENT`
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  # Set the ID of the LLM to delete
  LLM_ID_TO_DELETE="some-llm-id"

  curl -X DELETE "http://localhost:8000/admin/llms/$LLM_ID_TO_DELETE" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

#### Assign an LLM to a user
Assigns a specific LLM to a specific user.

- **Endpoint:** `PUT /admin/users/{user_id}/llm/{llm_id}`
- **Authentication:** Bearer Token required (Admin role).
- **Path Parameters:** `user_id` (string), `llm_id` (string)
- **Response Body:** The updated user object.
- **Example with `curl`:**
  ```bash
  # Ensure you have set the ADMIN_TOKEN variable
  # Set the user ID and LLM ID
  USER_ID_TO_ASSIGN="some-user-id"
  LLM_ID_TO_ASSIGN="some-llm-id"

  curl -X PUT "http://localhost:8000/admin/users/$USER_ID_TO_ASSIGN/llm/$LLM_ID_TO_ASSIGN" \
    -H "Authorization: Bearer $ADMIN_TOKEN"
  ```

### System Settings

#### Get system settings
Retrieves current system settings. (Placeholder)

- **Endpoint:** `GET /admin/settings`
- **Authentication:** Bearer Token required (Admin role).

#### Update system settings
Updates system settings. (Placeholder)

- **Endpoint:** `POST /admin/settings`
- **Authentication:** Bearer Token required (Admin role).

## Agent Workflow Orchestration

This section describes the endpoints used to manage the multi-agent collaboration workflow. The system uses a stateful graph (`LangGraph`) to orchestrate complex interactions between agents, including stages for brainstorming, drafting, and human-in-the-loop review.

### Start a New Workflow

This endpoint initiates a new multi-agent workflow. You provide an initial query and define the agents that will participate in the task. The system will perform an initial brainstorming session and then pause, awaiting human review and role assignment.

- **Endpoint:** `POST /graph/execute`
- **Authentication:** Bearer Token required.
- **Request Body:**
  - `query` (string): The initial prompt or question for the agents.
  - `language` (string): The language for the response (e.g., "en", "ja").
  - `agents` (list of objects): A list of agent configurations. Each object must contain:
    - `name` (string): A unique name for the agent.
    - `role` (string): A descriptive role (e.g., "Researcher", "Critic"). This is for informational purposes at this stage.
    - `llm_provider` (string): The LLM provider for this agent (e.g., "chatgpt", "local", "gemini").
    - `key_id` (string): The ID of the API key from the user's vault. This is required for all non-`local` providers. For `local`, it can be an empty string.
- **Response Body:**
  - `thread_id` (string): A unique ID for this workflow instance, which is used to resume it later.
  - `state` (object): The initial state of the workflow.

- **Example with `curl`:**
  ```bash
  # Ensure you have set the TOKEN variable
  # This example uses an OpenAI key (assumed to be stored) and a local model.
  # The key_id "your-openai-key-id" should be replaced with an actual ID from your vault.
  curl -X POST "http://localhost:8000/graph/execute" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What are the pros and cons of using LangGraph for multi-agent systems?",
      "language": "en",
      "agents": [
        {
          "name": "OpenAI Researcher",
          "role": "Researcher",
          "llm_provider": "chatgpt",
          "key_id": "your-openai-key-id"
        },
        {
          "name": "Local Critic",
          "role": "Critic",
          "llm_provider": "local",
          "key_id": ""
        }
      ]
    }'
  ```

### Resume a Workflow (Human-in-the-Loop)

After a workflow is started, it pauses for human intervention. This endpoint allows you to resume the workflow by assigning specific, functional roles to the agents and providing further instructions. This is how you can direct agents to perform tasks like generating counter-arguments or conducting literature searches.

- **Endpoint:** `POST /graph/{state_id}/resume`
- **Authentication:** Bearer Token required.
- **Path Parameter:**
  - `state_id` (string): The `thread_id` returned from the `/graph/execute` endpoint.
- **Request Body:** A JSON object that updates the workflow's state. Key fields include:
  - `assigned_roles` (object): A dictionary mapping agent names (from the initial configuration) to specific functional roles. The available roles determine the subsequent actions in the workflow:
    - **`Drafter`**: Synthesizes proposals into a comprehensive draft.
    - **`Judge`**: Evaluates the quality of the draft.
    - **`Verifier`**: Reviews the draft for issues and errors.
    - **`Research`**: Elaborates on the draft based on a new query.
    - **`Counter-argument`**: Provides a critical review or counter-argument to the Research agent's output.
    - **`Literature`**: Performs a RAG-based search to find supporting or refuting literature for the Research agent's output.
  - `elaboration_query` (string, optional): A more detailed query to guide the `Research` agent. This is required if the `Research` role is assigned.

- **Example with `curl` (Assigning Roles for Debate and Research):**
  ```bash
  # Ensure you have set the TOKEN variable
  # Use the thread_id from the previous /graph/execute call
  THREAD_ID="your-thread-id-from-previous-step"

  curl -X POST "http://localhost:8000/graph/$THREAD_ID/resume" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "assigned_roles": {
        "OpenAI Researcher": "Research",
        "Local Critic": "Counter-argument"
      },
      "elaboration_query": "Focus on the performance implications and memory usage of LangGraph compared to other orchestration frameworks."
    }'
  ```

### Receiving Real-Time Updates (WebSocket)

This WebSocket endpoint allows the client to receive real-time updates from the agent workflow. Once connected, the server will push events as they happen, such as when an agent starts generating a response or sends a token.

- **Endpoint:** `WS /ws/status/{client_id}`
- **Path Parameter:**
  - `client_id` (string): A unique ID generated by the client to identify the WebSocket connection.
- **Server-to-Client Messages:** The client will receive JSON messages with the following `type`:
  - `stream_start`: Indicates that an agent has started generating a response.
  - `token`: Contains a single token from the agent's streaming response.
  - `stream_end`: Indicates that an agent has finished generating its response for a given message ID.
  - `error`: Indicates that an error occurred during the workflow.

- **Example with JavaScript:**
  ```javascript
  const clientId = "client-" + Math.random().toString(36).substring(2, 15);
  const ws = new WebSocket(`ws://localhost:8000/ws/status/${clientId}`);

  ws.onopen = () => {
    console.log("WebSocket connection established.");
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("Received:", message);

    if (message.type === 'token') {
      // Append message.token to the UI for the corresponding agent
    } else if (message.type === 'error') {
      // Display an error message
      console.error("Workflow Error:", message.message);
    }
  };

  ws.onclose = () => {
    console.log("WebSocket connection closed.");
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
  ```

## Infrastructure Services

This section outlines the core infrastructure services that the backend relies on. These services are defined in the `docker-compose.yml` file.

### PostgreSQL (Database)

The primary database for storing application data.

- **Docker Image:** `postgres:14`
- **Service Name:** `postgres`
- **Ports:** `5433:5432` (Host:Container)
- **Database Name:** `cogitatedb`
- **Username:** `user`
- **Password:** `password`
- **Backend Connection URL:** `postgresql://user:password@postgres:5432/cogitatedb`

### Redis

Used for caching and as a message broker.

- **Docker Image:** `redis/redis-stack:latest`
- **Service Name:** `redis`
- **Ports:** `6379:6379` (Host:Container)
- **Backend Connection URL:** `redis://redis:6379/0`

### Jaeger (Distributed Tracing)

Used for monitoring and troubleshooting microservices-based distributed systems.

- **Docker Image:** `jaegertracing/all-in-one:1.52`
- **Service Name:** `jaeger`
- **Ports:**
  - `16686:16686` (Jaeger UI)
  - `4317:4317` (OTLP Receiver)
- **Backend Connection Host:** `jaeger:4317`