# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# Cogitate User Manual

## 1. Overview

Cogitate is a conference-style interactive platform where multiple AI agents collaborate to generate a single answer. Users can dynamically form a team of agents, assigning each a role and a Large Language Model (LLM) to use. This allows for a process that generates multifaceted and sophisticated answers that would be unattainable with a single AI.

## 2. User Roles

The system has two user roles:

- **Administrator**: Has the authority to manage all user accounts (changing roles, deletion, etc.) and configure system-wide settings.
- **Operator**: The primary user of this system. They log in with their own account, set their personal LLM API keys, and run conferences with AI agents.

## 3. Getting Started

### 3.1 User Registration
If you are using the system for the first time, you need to create a user account.
1.  Click the `Register` link on the login screen.
2.  Enter your username, email address, and password to complete the registration.
3.  Upon completion, you will be automatically redirected to the login page.

### 3.2 Login
1.  On the login page, enter your registered username and password.
2.  Click the `Login` button.

### 3.3 API Key Setup (Operators Only)
When you log in as an operator for the first time, the LLM API key setup screen will be displayed.
1.  Enter the API key for the LLM provider you will use (e.g., OpenAI, xAI, Google).
2.  Click the `Save Key` button to save the key. This key is encrypted and stored securely.
3.  Once the key is saved, you will be redirected to the main conference setup screen.

## 4. Main Screen (Meeting Setup View)

After logging in and completing the initial setup, the Meeting View is displayed. Here, you will form the team of AI agents that will participate in the conference.

- **Agent Card**: Each card displayed on the screen represents a single AI agent.
  - **Agent Name**: The display name of the agent. You can change it freely.
  - **Role**: The initial role to give the agent to indicate its specialty or background (e.g., "Software Engineer"). This role affects the tone of the AI's answer but is different from the functional roles assigned in a later step.
  - **LLM Provider**: Select the LLM for the agent to use (e.g., `chatgpt`, `gemini`, `grok`, `claude`, `local`).
  - **Ollama Model Name** (Local LLM Only): This appears when you select `local` as the `LLM Provider`. Enter the exact name of the Ollama model you wish to use (e.g., `llama3`, `mistral`).
- **+ Add Agent Button**: Clicking this adds a new agent card to the grid.
- **Query Input Field**: In the chat interface at the bottom of the screen, enter the topic or question you want the agent team to discuss.
- **Send Button**: After entering the query, click the send button (paper airplane icon) to start the conference with the configured agent team and switch to the log view.

## 5. Conference Log View (Multi-stage Workflow)

When a conference starts, the screen automatically switches to the Conference Log View. This view consists of a central main panel and a right-hand sidebar.

### 5.1 Operations in the Main Panel

The main panel displays the agents' answers and the controls to proceed to the next step.

1.  **Stage 1: Brainstorming and Role Assignment**
    - **Review Initial Answers**: First, all agents generate answers to the initial query, which are displayed on each agent card.
    - **Assign Functional Roles**: Using the dropdown menu on each card, assign a specialized role to each agent according to the intended workflow. For best results, it is recommended to assign one of each of the following roles: "Research," "Counter-argument," and "Literature."
    - **Enter Elaboration Query**: In the text box labeled `Elaboration Query`, enter new instructions or questions to deepen the discussion in Stage 2.
    - **Continue Workflow**: Click the `Submit Roles and Continue` button to proceed to Stage 2.

2.  **Stage 2: Elaboration and Finalization**
    - **Real-time Updates**: In Stage 2, the agents engage in a more specialized discussion. You can see this unfold in the "Conference Log" in the sidebar, where text appears in real-time with a typewriter effect.
    - **Final Answer**: Once all processes are complete, a final answer integrating the results of the discussion will be generated (to be implemented in the future).

- **Back to Setup Button**: If you want to interrupt the conference and reconfigure the agents at any time, you can return to the setup screen with this button.

### 5.2 Confirmation in the Sidebar

The right-hand sidebar allows you to check the progress of the workflow in detail.

- **Live Workflow**: Visually displays the structure of the current workflow and the active node (processing step).
- **Conference Log**: Shows the entire history of utterances in the conference chronologically. You can see at a glance which agent said what, in which turn, and in what role.

## 6. Administrator Functions

If you are logged in with the administrator role, you can access the admin dashboard from the `Admin` link in the navigation bar.

### 6.1 User Management
- **User List**: A list of all users registered in the system is displayed.
- **Role Change**: You can switch the role of each user between `operator` and `administrator`.
- **User Deletion**: You can delete unnecessary user accounts.

### 6.2 Ollama Model Management
From the dashboard, you can access the "Ollama Manager" to manage locally available Ollama models.
- **List Available Models**: A list of models already downloaded to the server is displayed.
- **Pull New Models**: Enter a new model name (e.g., `codellama:latest`) and click "Pull Model" to start the download in the background. Large models may take a significant amount of time.
- **Refresh List**: You can update the list to the latest state with the "Refresh List" button.
