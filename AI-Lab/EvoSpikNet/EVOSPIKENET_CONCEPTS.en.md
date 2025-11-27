# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet: Key Concepts

**Last Updated:** 2025-11-26

This document provides technical details on the more advanced and unique concepts that form the core of the EvoSpikeNet framework.

---

## 1. SpikingEvoSpikeNetLM (`evospikenet/models.py`)

`SpikingEvoSpikeNetLM` is the flagship language model of EvoSpikeNet, a hybrid model that fuses a standard Transformer architecture with the principles of Spiking Neural Networks (SNNs).

### 1.1. Architectural Overview
- **`TAS-Encoding` (`evospikenet/encoding.py`)**: A Time-Adaptive Spike encoding layer that converts text tokens into spike trains that adapt temporally based on their importance or context.
- **`SpikingTransformerBlock`**: The core building block of the model, encapsulating a spike-based self-attention and a feed-forward network.
- **`ChronoSpikeAttention` (`evospikenet/attention.py`)**: The attention mechanism used within the `SpikingTransformerBlock`, which captures the temporal relationships between spikes to generate a context vector.

### 1.2. Principle of Operation
Input text is tokenized, converted into spike trains by the `TAS-Encoding` layer, passed through a stack of `SpikingTransformerBlock`s, and finally decoded to predict the next token. This architecture allows the model to combine the powerful sequence processing capabilities of traditional language models with the advantages of SNNs, such as temporal dynamics and energy efficiency.

---

## 2. MultiModalEvoSpikeNetLM (`evospikenet/models.py`)

`MultiModalEvoSpikeNetLM` extends the capabilities of EvoSpikeNet to tri-modal data, encompassing text, images, and audio.

### 2.1. Architectural Overview
- **`SpikingVisionEncoder` (`evospikenet/vision.py`)**: Extracts features from images using a spiking convolutional neural network (SCNN).
- **`SpikingAudioEncoder` (`evospikenet/audio.py`)**: Extracts features from audio waveforms using a spike-based audio processing model.
- **Text Encoder**: Extracts features from text prompts.
- **Fusion Mechanism**: Feature vectors from the three modalities are fused via mechanisms like cross-attention to create a rich, unified representation.

---

## 3. Hybrid Search RAG (`evospikenet/rag_milvus.py`)

The Retrieval-Augmented Generation (RAG) system implements a hybrid search architecture, combining vector and keyword search for superior retrieval accuracy.

### 3.1. Architecture
- **Milvus (Vector Search)**: Handles semantic similarity search.
- **Elasticsearch (Keyword Search)**: Handles traditional full-text search.

### 3.2. Workflow
1.  **Parallel Retrieval**: User queries are sent to Milvus and Elasticsearch simultaneously.
2.  **Result Fusion**: Results are intelligently merged using the **Reciprocal Rank Fusion (RRF)** algorithm.
3.  **Data Synchronization**: All CRUD (Create, Update, Delete) operations on the knowledge base are performed synchronously across both databases to ensure data consistency.

---

## 4. Federated Learning (`evospikenet/federated.py`)

Federated Learning is supported through the integration of the `Flower` (`flwr`) framework, enabling collaborative model training on decentralized data while preserving privacy.

### 4.1. Components
- **`EvoSpikeNetClient`**: Inherits from `flwr.client.NumPyClient` to encapsulate local model training logic.
- **`DistributedBrainClient`**: A custom client specialized for the distributed brain architecture. It uses **spike distillation** (transmitting mean firing activity) for knowledge sharing instead of direct parameter averaging.

---

## 5. RESTful API (`evospikenet/api.py`)

A `FastAPI`-based RESTful API serves as the primary interface for programmatically accessing EvoSpikeNet's capabilities.

- **Key Endpoints**:
    - `/api/generate`: Executes text generation.
    - `/api/distributed_brain/status`: Retrieves the real-time status of the distributed brain simulation.
    - `/api/distributed_brain/prompt`: Submits a new prompt (text, image, or audio) to the simulation.
    - `/api/distributed_brain/result`: Fetches the final result from the simulation.
- **Model Lifecycle**: On startup, the API server loads the latest trained model into memory once to minimize per-request latency.

---

## 6. Distributed Brain Architecture (`examples/run_distributed_brain_simulation.py`)

The Distributed Brain Architecture is the framework's most advanced feature, inspired by the functional specialization of the biological brain. It employs a master/slave model where specialized "functional modules" run as separate processes using `torch.distributed`, coordinated by a central "Prefrontal Cortex (PFC) Module."

### 6.1. Architectural Components

-   **Master Process (Rank 0): PFC (Prefrontal Cortex) Module (`evospikenet/pfc.py`)**:
    -   **Role**: The master node, acting as the "cognitive control center" of the architecture.
    -   **Task Routing**: Receives prompts from the API, interprets the task, and dynamically dispatches it to the appropriate functional module (slave process).
    -   **State Management**: Manages the state of the entire simulation and periodically POSTs status updates to the FastAPI backend, enabling real-time monitoring via the web UI.
    -   **Flag Control**: The simulation's start and stop are controlled by a filesystem flag (`/tmp/stop_evospikenet_simulation.flag`).

-   **Slave Processes (Rank > 0): Functional Modules (`evospikenet/functional_modules.py`)**:
    -   **Role**: Slave nodes that act as specialists for executing specific tasks (e.g., vision, language, motor control).
    -   **Hierarchical Structure**: Many functions are implemented as **hierarchical processing pipelines**. For example, in a "Language Focus" simulation, a parent language node (Rank 4) sequentially invokes child nodes for embedding (Rank 7), TAS encoding (Rank 8), etc. Similar hierarchical structures exist for the visual, auditory, and motor domains.
    -   **Dynamic and Multimodal Model Loading**: Each slave process dynamically downloads and loads the model specified in the `--model-config` argument at startup. If the model's `config.json` specifies `"model_type": "MultiModalEvoSpikeNetLM"`, the node will instantiate a `MultiModalEvoSpikeNetLM`. This allows a specialized node, such as a vision module, to exhibit more advanced, multimodal behavior by interpreting both its primary modality (images) and an accompanying text prompt simultaneously.

### 6.2. Inter-Process Communication and Workflow

-   **Communication Backbone**: High-speed, efficient data transfer is achieved using `torch.distributed`'s point-to-point communication (`send`/`recv`). Communication between the PFC and slaves employs a **manifest-based protocol**, where a manifest tensor is sent first to declare the type of data (text, image, audio) that will follow, enabling flexible multimodal data exchange.
-   **Workflow**:
    1.  **Prompt Input**: A user submits a prompt through the UI, which is sent to the FastAPI backend.
    2.  **PFC Polling**: The master process (PFC) periodically polls the API to fetch new prompts.
    3.  **Task Dispatch**: The PFC processes the prompt and dispatches the task to the appropriate functional module (or the parent of a hierarchy).
    4.  **Hierarchical Processing**: The designated parent module sends sub-tasks to its child modules as needed and awaits their results. Processing flows through the hierarchy, and the final result is aggregated at the parent.
    5.  **Result Return**: The parent module returns the aggregated result to the PFC.
    6.  **Status and Result Reporting**: The PFC POSTs the real-time status and the final result to dedicated API endpoints, which the UI polls to update its display.

-   **Debugging and Logging**: Each process generates its own log file (`/tmp/sim_rank_{rank}.log`), facilitating the debugging of complex interactions in the distributed environment.
