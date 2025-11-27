# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet - A Distributed Brain Simulation Framework

**Last Updated:** 2025-11-24

## 1. Project Overview

EvoSpikeNet is a scalable **distributed brain simulation framework** inspired by the principles of functional specialization and integration in the biological brain. In this framework, specialized neural modules (e.g., for vision, language, motor control) operate as independent processes, coordinated and integrated by a central **Prefrontal Cortex (PFC) Module**.

The framework's most unique feature is the **Q-PFC Feedback Loop** implemented within the PFC. This is a sophisticated, self-referential control mechanism where the PFC measures its own decision-making uncertainty (cognitive entropy), uses that value to simulate a quantum-inspired circuit, and then feeds the result back to modulate its own neural activity.

Built on `torch.distributed`, EvoSpikeNet supports multi-process and multi-node execution, enabling the construction and study of large-scale neuromorphic systems that go beyond the limitations of a single device.

## 2. Key Implemented Features

- **Distributed Brain Simulation**:
    - **Cognitive Control Hub (PFC)**: A central control hub featuring `ChronoSpikeAttention` for task routing and the Q-PFC feedback loop for self-modulation.
    - **Hierarchical Functional Modules**: Functions for vision, audition, language, and motor control are implemented as hierarchical processing pipelines with parent and child nodes, enabling multi-step information processing.
    - **Interactive UI**: A web interface for submitting multi-modal prompts (text, image, audio), running simulations, monitoring their real-time state, and retrieving results.

- **Q-PFC Feedback Loop**:
    - The framework's most innovative feature, where the PFC's cognitive load (entropy) dynamically adjusts its own working memory's dynamics via a `QuantumFeedbackSimulator`.

- **Full-fledged SNN Language Model (`SpikingEvoSpikeNetLM`)**:
    - A `snnTorch`-based, spike-driven Transformer model featuring custom components like `TAS-Encoding` and `ChronoSpikeAttention`.

- **Tri-Modal Processing Capability (`MultiModalEvoSpikeNetLM`)**:
    - Integrates three different modalities: text, images (`SpikingVisionEncoder`), and audio (`SpikingAudioEncoder`).

- **Hybrid Search RAG**:
    - Achieves high-accuracy Retrieval-Augmented Generation by querying Milvus (vector search) and Elasticsearch (keyword search) in parallel and fusing the results with the Reciprocal Rank Fusion (RRF) algorithm.
    - **Long Text Support**: Supports documents up to 32,000 characters (Milvus limit: 65,535 characters).
    - **Auto Prompt Truncation**: Automatically adjusts prompt length to fit model constraints (e.g., HuggingFace models).
    - **Interactive Data Management**: User-friendly UI with checkbox row selection, inline editing, and real-time character counter.

- **Diverse SNN Core Engine**:
    - Supports multiple neuron models, including the computationally efficient `LIFNeuronLayer`, the biologically plausible `IzhikevichNeuronLayer`, and the quantum-inspired `EntangledSynchronyLayer`.

- **Federated Learning (Flower)**:
    - Integrates the `Flower` framework to support privacy-preserving, collaborative model training in decentralized environments.

- **RESTful API & Python SDK**:
    - A `FastAPI`-based API provides programmatic access to all framework features, including text generation, data logging, and control of the distributed brain simulation.
    - A comprehensive Python SDK (`EvoSpikeNetAPIClient`) is available for easy interaction with the API.
    - **Optimized Inter-Container Communication**: Switched from file-based to API-based communication for improved reliability between Docker containers.

- **Integrated Web UI (Dash)**:
    - A multi-page web application that provides an interactive interface for all framework functionalities, including data generation, model training, inference, results analysis, and system administration.
    - **Real-time State Monitoring**: Visualizes each node's status, energy, and spike activity in real-time during distributed brain simulations.
    - **Multi-Modal Queries**: Supports complex queries combining text, images, and audio.

## 3. Launching the Web UI

All simulation, parameter tuning, and visualization can be done from the Web UI. The following commands will launch the UI and all necessary backend services (API, Milvus, Elasticsearch, etc.).

```bash
# For environments with a GPU
sudo ./scripts/run_frontend_gpu.sh

# For CPU-only environments
sudo ./scripts/run_frontend_cpu.sh
```
After launching, access the dashboard in your browser at `http://localhost:8050`.

## 4. Environment Setup using Docker (Recommended)

This project fully utilizes Docker Compose, allowing you to set up a complete development and execution environment with just a few commands.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2 or later, `docker compose` command)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using GPU mode)

### Build the Environment
When running for the first time or after any changes to the `Dockerfile`, build the Docker images with the following command (you may need `sudo`):
```bash
docker compose build
```

### Other Commands
- **Start the API server only:** `sudo ./scripts/run_api_server.sh`
- **Run the test suite:** `sudo ./scripts/run_tests_cpu.sh`

## 5. Project Structure

| Path                 | Description                                                              |
| :------------------- | :----------------------------------------------------------------------- |
| `evospikenet/`       | Main source code for the framework.                                      |
| `frontend/`          | Source code for the Dash-based Web UI application.                       |
| `tests/`             | Unit tests using `pytest`.                                               |
| `scripts/`           | Shell scripts to simplify development, testing, and execution.           |
| `examples/`          | Sample programs demonstrating specific uses of the framework.            |
| `docker-compose.yml` | Docker Compose configuration defining all services (UI, API, DBs, etc.). |
| `pyproject.toml`     | Defines project metadata and Python dependencies.                        |
| `README.en.md`       | This file.                                                               |

## 6. Documentation

For more detailed technical information and usage instructions, please refer to the following documents:
- **Technical Concepts**: `EVOSPIKENET_CONCEPTS.en.md`
- **Feature Analysis**: `FEATURE_ANALYSIS.md` (Japanese)
- **UI User Manual**: `UserManual.en.md`
- **SDK Guide**: `EvoSpikeNet_SDK.en.md`
- **Data Formats**: `DATA_HANDLING.en.md`
- **Implementation Status & Roadmap**: `REMAINING_FEATURES.en.md`
