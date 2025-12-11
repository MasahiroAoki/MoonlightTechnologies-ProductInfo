
# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# ⚠️ Important Notice for Commercial Use

The code in this repository is released under the MIT License, but **incorporating it into commercial products or services by enterprises**, and **implementing patents held by our company** to generate revenue requires a separate "Commercial License Agreement for Enterprises".

Main applicable cases:
- Incorporating this framework into your own SaaS, app, or service for paid provision
- Using this framework to provide AI features to customers for a fee
- Large-scale use in internal systems that directly leads to revenue

→ Applicable enterprises must contact us at:
✉️ maoki@moonlight-tech.biz
Web: https://www.moonlight-tech.biz/commercial-license

Personal use, research use, PoC, and startup prototypes are completely free to use under the MIT license.

# EvoSpikeNet - Distributed Brain Simulation Framework

**Last Updated:** December 10, 2025

## 1. Project Overview

EvoSpikeNet is a scalable **Distributed Brain Simulation Framework** inspired by the principles of functional specialization and integration in the biological brain. Specialized neural modules (vision, language, motor, etc.) operate as separate processes, dynamically coordinated and integrated by a central **Prefrontal Cortex (PFC) module**.

The most distinctive feature of this framework is the **Q-PFC Feedback Loop** implemented in the PFC. This is an advanced self-referential control mechanism where the PFC measures the uncertainty of its own decision-making (cognitive entropy), simulates a quantum-inspired circuit using that value, and feeds back the result to its own neuronal activity.

Based on `torch.distributed`, it supports multi-process/multi-node execution, enabling the construction and research of large-scale neuromorphic systems beyond the constraints of a single device.

## 2. Key Implemented Features

- **Distributed Brain Simulation (Zenoh-based)**:
    - **Asynchronous Communication**: Adopted `Zenoh` publish/subscribe model, moving away from the `torch.distributed` based legacy architecture. This significantly improves robustness and scalability.
    - **Cognitive Control by PFC**: Central control hub featuring task routing using `ChronoSpikeAttention` and self-modulation using the Q-PFC feedback loop.
    - **Hierarchical Functional Modules**: Functions such as vision, auditory, language, and motor are implemented as hierarchical processing pipelines consisting of parent nodes and multiple child nodes.
    - **Interaction via UI**: Supports sending multi-modal prompts including text, images, and audio from the Web UI, simulation execution, real-time status monitoring, and result retrieval.

- **Q-PFC Feedback Loop**:
    - The most original feature of this framework, where the PFC dynamically adjusts the dynamics of its working memory via the `QuantumModulationSimulator` according to its own cognitive load (entropy).

- **Full-scale SNN Language Model (`SpikingTextLM`)**:
    - `snnTorch`-based Transformer model operating with spikes. Includes custom components like `TAS-Encoding` and `ChronoSpikeAttention`.

- **Tri-modal Processing Capability (`SpikingMultiModalLM`)**:
    - Integrally processes three modalities: text, image (`SpikingVisionEncoder`), and audio (`SpikingAudioEncoder`).

- **Hybrid Search RAG**:
    - Achieves high-precision retrieval-augmented generation by running Milvus (vector search) and Elasticsearch (keyword search) in parallel and fusing results with the Reciprocal Rank Fusion (RRF) algorithm.
    - **Long Text Support**: Can store documents up to 65,535 characters based on Milvus schema definitions.
    - **Automatic Prompt Truncation**: Automatically adjusts prompts to optimal length according to constraints of Hugging Face models, etc.
    - **Interactive Data Management**: Easy-to-use UI with checkbox row selection, inline editing, real-time character counter, etc.

- **Diverse SNN Core Engines**:
    - Supports multiple neuron models including computationally efficient `LIFNeuronLayer`, biologically plausible `IzhikevichNeuronLayer`, and quantum-inspired `EntangledSynchronyLayer`.

- **Federated Learning (Flower)**:
    - Integrates the `Flower` framework to support collaborative model training in a distributed environment while preserving privacy.

- **RESTful API and Python SDK**:
    - `FastAPI`-based API provides programmatic access to all framework functions, including text generation, data logging, and distributed brain simulation control.
    - Complete Python SDK (`EvoSpikeNetAPIClient`) provided for easy API usage.
    - **Optimized Inter-container Communication**: Changed from file-based to API-based communication, improving reliability between Docker containers.

- **Integrated Web UI**:
    - Dash-based multi-page application that allows interactive operation of all framework functions from the browser, including data generation, model training, inference, result analysis, and system management.
    - **Real-time Status Monitoring**: Visualizes the status, energy, and spike activity of each node in the distributed brain simulation in real-time.
    - **Multi-modal Query**: Supports complex queries combining text, images, and audio.

- **Simulation Data Recording & Analysis**:
    - Optional feature to record spikes, membrane potentials, weights, and control states.
    - Efficient data storage in HDF5 format, automated analysis, and visualization tools (`sim_recorder.py`, `sim_analyzer.py`).
    - Firing rate calculation, spike raster plots, firing rate time series plots, automatic summary report generation.

## 3. Launching Web UI

Simulation execution, parameter tuning, and result visualization can all be done from the Web UI. The following command launches the UI and all necessary backend services (API, Milvus, Elasticsearch, etc.).

```bash
# For environments with GPU
sudo ./scripts/run_frontend_gpu.sh

# For CPU-only environments
sudo ./scripts/run_frontend_cpu.sh
```
After startup, access `http://localhost:8050` in your browser.

## 4. Environment Setup using Docker (Recommended)

This project fully adopts Docker Compose, allowing you to build a complete development and execution environment with a few commands.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+, `docker compose` command)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (When using GPU mode)

### Building the Environment
Build the Docker image with the following command when running for the first time or when `Dockerfile` has changed (may require `sudo`).
```bash
docker compose build
```

### Other Commands
- **Start API Server only:** `sudo ./scripts/run_api_server.sh`
- **Run Test Suite:** `sudo ./scripts/run_tests_cpu.sh`

## 5. Project Structure

| Path                 | Description                                                             |
| :------------------- | :---------------------------------------------------------------------- |
| `evospikenet/`       | Main source code of the framework.                                      |
| `frontend/`          | Source code of the Dash-based Web UI application.                       |
| `tests/`             | Unit tests using `pytest`.                                              |
| `scripts/`           | Shell scripts to simplify development, testing, and execution.          |
| `examples/`          | Sample programs demonstrating specific uses of the framework.           |
| `docker-compose.yml` | Docker Compose configuration defining all services (UI, API, DB, etc.). |
| `pyproject.toml`     | Defines project metadata and Python dependencies.                       |
| `README.md`          | This file.                                                              |

## 6. Documentation

For more detailed technical information and usage instructions, refer to the following documents in the `docs/` directory.

### 📚 List of Main Documents

| Document Name                       | Japanese                                                                              | English                                                                                     | Description                                                     |
| :---------------------------------- | :------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ | :-------------------------------------------------------------- |
| **Concepts**                        | [EVOSPIKENET_CONCEPTS.md](docs/EVOSPIKENET_CONCEPTS.md)                               | [EVOSPIKENET_CONCEPTS.en.md](docs/EVOSPIKENET_CONCEPTS.en.md)                               | Basic concepts and design philosophy of the framework           |
| **User Manual**                     | [UserManual.md](docs/UserManual.md)                                                   | [UserManual.en.md](docs/UserManual.en.md)                                                   | Web UI Operation Guide                                          |
| **SDK**                             | [EvoSpikeNet_SDK.md](docs/EvoSpikeNet_SDK.md)                                         | [EvoSpikeNet_SDK.en.md](docs/EvoSpikeNet_SDK.en.md)                                         | Python SDK Detailed Guide                                       |
| **SDK Quickstart**                  | [SDK_QUICKSTART.md](docs/SDK_QUICKSTART.md)                                           | [SDK_QUICKSTART.en.md](docs/SDK_QUICKSTART.en.md)                                           | Simple Start Guide for SDK                                      |
| **Data Handling**                   | [DATA_HANDLING.md](docs/DATA_HANDLING.md)                                             | [DATA_HANDLING.en.md](docs/DATA_HANDLING.en.md)                                             | Data formats and processing methods                             |
| **Distributed Brain System**        | [DISTRIBUTED_BRAIN_SYSTEM.md](docs/DISTRIBUTED_BRAIN_SYSTEM.md)                       | [DISTRIBUTED_BRAIN_SYSTEM.en.md](docs/DISTRIBUTED_BRAIN_SYSTEM.en.md)                       | Details of Distributed Brain Simulation                         |
| **RAG System**                      | [RAG_SYSTEM_DETAILED.md](docs/RAG_SYSTEM_DETAILED.md)                                 | [RAG_SYSTEM_DETAILED.en.md](docs/RAG_SYSTEM_DETAILED.en.md)                                 | Implementation details of Hybrid Search RAG                     |
| **Implementation Status & Roadmap** | [REMAINING_FEATURES.md](docs/REMAINING_FEATURES.md)                                   | [REMAINING_FEATURES.en.md](docs/REMAINING_FEATURES.en.md)                                   | Implemented features and future plans                           |
| **L5 Self-Evolution Plan**          | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.md)     | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md) | Detailed plan for L5-level self-evolution functions ⭐           |
| **L5 Feature Breakdown**            | [L5_FEATURE_BREAKDOWN.md](docs/L5_FEATURE_BREAKDOWN.md)                               | [L5_FEATURE_BREAKDOWN.en.md](docs/L5_FEATURE_BREAKDOWN.en.md)                           | Detailed breakdown and implementation policy of L5 features ⭐   |
| **LLM Integration Strategy**        | [LLM_INTEGRATION_STRATEGY.md](docs/LLM_INTEGRATION_STRATEGY.md)                       | [LLM_INTEGRATION_STRATEGY.en.md](docs/LLM_INTEGRATION_STRATEGY.en.md)                   | Strategy for integrating Large Language Models                  |
| **AEG-Comm Implementation Plan**    | [AEG_COMM_IMPLEMENTATION_PLAN.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.md)               | [AEG_COMM_IMPLEMENTATION_PLAN.en.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.en.md)           | Implementation plan for intelligent communication control ⭐ NEW |
| **Simulation Recording Guide**      | [SIMULATION_RECORDING_GUIDE.md](docs/SIMULATION_RECORDING_GUIDE.md)                   | [SIMULATION_RECORDING_GUIDE.en.md](docs/SIMULATION_RECORDING_GUIDE.en.md)                   | How to use data recording/analysis application functions ⭐      |
| **Simulation Recording README**     | [SIMULATION_RECORDING_README.md](docs/SIMULATION_RECORDING_README.md)                 | [SIMULATION_RECORDING_README.en.md](docs/SIMULATION_RECORDING_README.en.md)                 | Overview of recording function                                  |
| **Spike Communication Analysis**    | [SPIKE_COMMUNICATION_ANALYSIS.md](docs/SPIKE_COMMUNICATION_ANALYSIS.md)               | [SPIKE_COMMUNICATION_ANALYSIS.en.md](docs/SPIKE_COMMUNICATION_ANALYSIS.en.md)               | Analysis methods for spike communication                        |
| **Pipeline Analysis**               | [distributed_brain_pipeline_analysis.md](docs/distributed_brain_pipeline_analysis.md) | [distributed_brain_pipeline_analysis_en.md](docs/distributed_brain_pipeline_analysis_en.md) | Detailed analysis of distributed brain pipeline                 |
| **Documentation Update Summary**    | [DOCUMENTATION_UPDATE_SUMMARY.md](docs/DOCUMENTATION_UPDATE_SUMMARY.md)               | [DOCUMENTATION_UPDATE_SUMMARY.en.md](docs/DOCUMENTATION_UPDATE_SUMMARY.en.md)               | Documentation update history                                    |

### 📁 Other Documents

- **Architecture Diagrams**: `docs/architecture/` directory
- **SDK Details**: `docs/sdk/` directory
