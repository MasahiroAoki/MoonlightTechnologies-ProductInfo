<!-- AUTO-TRANSLATED from README.md on 2025-12-31. Please review. -->

# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

IMPORTANT NOTICE FOR COMMERCIAL USERS

The code in this repository is published under the MIT License; however, companies seeking to integrate this framework into commercial products or services, or to implement any patented technology owned by Moonlight Technologies for commercial gain, must enter into a separate "Enterprise Commercial License Agreement."

Typical cases that require a commercial license:
- Incorporating this framework into a paid SaaS, application, or service
- Providing AI functionality to customers using this framework in exchange for payment
- Large-scale use within corporate systems that directly generates revenue

If your organization falls into one of these categories, please contact: maoki@moonlight-tech.biz

Personal, academic, research, and prototype (PoC) use by startups remain free under the MIT License.

EvoSpikeNet — Distributed Brain Simulation Framework

Last updated: 2025-12-31
Version: v0.1.2
Status: 🟢 Production Ready (Plan C+ Phase 1 planned)

Project Status (high level):
- Core features: implemented (SNN, evolution, distributed processing, multimodality)
- P3 production features: 8 features implemented (latency monitoring, snapshots, scalability, hardware optimizations, availability monitoring, asynchronous communication, distributed consensus, data upload)
- Test infrastructure: implemented (integration, E2E, performance benchmarks, target 90% coverage)
- Type safety: implemented (mypy integration, modern Python typing)
- CI/CD pipeline: implemented (GitHub Actions, Docker builds, security scans, automated deploy)
- API docs: auto-generated (OpenAPI 3.0, Swagger UI, ReDoc, Postman collection)
- Episodic (long-term) memory: implemented (improved learning efficiency and adaptability)
- Long-Term Memory Nodes ⭐ NEW (December 31, 2025): New long-term memory system integrating FAISS vector search and Zenoh distributed communication (Episodic/Semantic memory, Memory Integrator Node)

See `docs/REMAINING_FEATURES.md` for implementation details and the 18-month roadmap.

1. Project overview

EvoSpikeNet is a scalable distributed brain simulation framework inspired by functional specialization and integration principles observed in biological brains. Specialized neural modules (vision, language, motor, etc.) run as independent processes, coordinated and dynamically integrated by a central prefrontal cortex (PFC) module.

The framework's distinctive feature is the Q-PFC feedback loop implemented within the PFC. The PFC measures uncertainty of its decisions (cognitive entropy) and uses that signal to simulate a quantum-inspired circuit; the modulation from that simulator is fed back into the PFC's neuronal activity to perform advanced self-referential control.

Built on `torch.distributed`, EvoSpikeNet supports multi-process and multi-node execution for scaling beyond single-device constraints.

2. Web UI and operation

- Distributed brain simulation (Zenoh-based): asynchronous Zenoh pub/sub communication, ChronoSpikeAttention for task routing, Q-PFC feedback loop for self-modulation, hierarchical functional modules (vision, auditory, language, motor), and a web UI for real-time monitoring and multimodal prompt submission.
- Spiking language model (`SpikingEvoTextLM`): snnTorch-based spiking Transformer with TAS-Encoding and ChronoSpikeAttention. Backwards-compatible `SpikingTextLM` remains for legacy support (planned removal in v2.0).
- Trimodal processing (`SpikingEvoMultiModalLM`): integrated text/image/audio processing.
- Hybrid RAG search: Milvus (vector) + Elasticsearch (keyword) with Reciprocal Rank Fusion (RRF) for high-precision retrieval and generation.

3. Running the Web UI

To start the UI and required backend services (API, Milvus, Elasticsearch):

```bash
# GPU environment
sudo ./scripts/run_frontend_gpu.sh

# CPU-only environment
sudo ./scripts/run_frontend_cpu.sh
```

Then open http://localhost:8050 in your browser.

4. Integration test system

An interactive test menu is provided to exercise integrated test suites across eight categories. Run:

```bash
python3 tests/unit/test_menu.py
```

See `docs/INTEGRATED_TEST_UI.md` for details.

5. Infrastructure as Code (IaC)

EvoSpikeNet provides IaC tooling (Terraform, Ansible, Kubernetes, Docker Compose) for reproducible environment setup and deployment. Common commands and make targets are provided in the README.

6. Docker-based setup (recommended)

Requirements: Docker, Docker Compose (v2+), NVIDIA Container Toolkit for GPU usage. Build and run via `docker compose build` and the provided `make` commands.

7. Project layout

Key directories and files:
- `evospikenet/` — core framework source
- `frontend/` — Dash-based Web UI
- `tests/` — pytest unit tests
- `scripts/` — development and run scripts
- `examples/` — sample programs
- `docker-compose.yml` — service definitions
- `pyproject.toml` — project metadata and dependencies

8. Documentation

See the `docs/` directory for comprehensive technical documentation and guides.

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

**Version:** v0.1.2  
**Last Updated:** December 31, 2025  
**Python Version:** >=3.8

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

- **Plugin Architecture** ⭐ NEW (December 20, 2025):
    - **Dynamic Plugin System**: Dynamic loading of seven plugin types (Neuron, Encoder, Plasticity, Functional, Learning, Monitoring, Communication), dramatically improving development efficiency
    - **70% Reduction in Feature Addition Time**: Feature addition time reduced from 4-5 days to 1-1.5 days through plugin isolation
    - **Built-in Plugins**: Includes 9 production-ready plugins (LIF/Izhikevich/EntangledSynchrony neurons, Rate/TAS/Latency encoders, STDP/MetaPlasticity/Homeostasis)
    - **Plugin Lifecycle Management**: Comprehensive plugin state management through registration, initialization, activation, and deactivation

- **Microservices Architecture** ⭐ NEW (December 20, 2025):
    - **Service-Oriented Architecture**: Transformed monolithic structure into 5 independent microservices (Training, Inference, Model Registry, Monitoring) + API Gateway
    - **80% Scalability Improvement**: Resource efficiency improved from 60% to 85% through independent scaling of each service
    - **Fault Isolation**: Service-level fault isolation reduces system-wide impact, improving overall reliability
    - **Docker Compose Orchestration**: Complete containerized deployment with `docker-compose.microservices.yml`, enabling easy multi-service management

- **Static Analysis Integration** ⭐ NEW (December 20, 2025):
    - **Automated Code Quality Checks**: Integrated 7 tools (Black, isort, Flake8, Pylint, mypy, Bandit, interrogate)
    - **Pre-commit Hooks**: Automatically run 10+ quality checks before commits
    - **CI/CD Automation**: GitHub Actions perform security scans and docstring coverage verification
    - **Quality Dashboard**: HTML visualization of Pylint/Bandit/Flake8 results
    - **Developer Efficiency**: Makefile, setup scripts, and comprehensive guides provided
    - **Quality Targets**: Pylint ≥7.0, Security issues ≤5, Docstring coverage ≥60%

- **Fine-grained Load Balancing** ⭐ NEW (December 20, 2025):
    - **Dynamic Load Distribution for Same Module Types**: 5 distribution strategies (least response time, weighted round-robin, consistent hashing, dynamic capacity, queue length)
    - **Instance Pooling**: Manages multiple instances per module type
    - **Real-time Metrics Monitoring**: Continuous tracking of response time, throughput, error rate
    - **Adaptive Capacity Management**: Automatic adjustment and rebalancing based on load
    - **Health-based Routing**: Health checking and automatic failover
    - **25% Throughput Improvement**: Benchmark shows 100→125 req/s, 24% response time reduction

- **Data Upload and Sharing**:
    - Upload training datasets to API server for sharing across distributed environments
    - Automatic fallback to local training when API server is unavailable
    - Support for multi-modal datasets (images + captions) with ZIP compression

- **Hybrid Search RAG**:
    - Achieves high-precision retrieval-augmented generation by running Milvus (vector search) and Elasticsearch (keyword search) in parallel and fusing results with the Reciprocal Rank Fusion (RRF) algorithm.
    - **Long Text Support**: Can store documents up to 65,535 characters based on Milvus schema definitions.
    - **Automatic Prompt Truncation**: Automatically adjusts prompts to optimal length according to constraints of Hugging Face models, etc.
    - **Interactive Data Management**: Easy-to-use UI with checkbox row selection, inline editing, real-time character counter, etc.
    - **Debug Visualization** ⭐ NEW (December 17, 2025): Visualize internal query processing (language detection, keyword extraction, vector/keyword search results, RRF fusion, generation details) in the UI.

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
    - **Model Classification Management** ⭐ NEW (December 17, 2025): When uploading models, you can select brain node type (Vision, Motor, Auditory, Speech, Executive, General), model category (20+ types), and model variant (Lightweight, Standard, High Accuracy, Realtime, Experimental). Enables systematic model management aligned with distributed brain architecture.

- **Simulation Data Recording & Analysis**:
    - Optional feature to record spikes, membrane potentials, weights, and control states.
    - Efficient data storage in HDF5 format, automated analysis, and visualization tools (`sim_recorder.py`, `sim_analyzer.py`).
    - Firing rate calculation, spike raster plots, firing rate time series plots, automatic summary report generation.

- **Long-Term Memory Nodes** ⭐ NEW (December 31, 2025):
    - **Episodic Memory (`EpisodicMemoryNode`)**: Storage and retrieval of time-sequenced events. Manages experiences and events in chronological order, enabling context-based recall.
    - **Semantic Memory (`SemanticMemoryNode`)**: Storage of concepts and knowledge. Links related concepts with automatic importance-based management.
    - **Memory Integrator Node (`MemoryIntegratorNode`)**: Cross-modal association and integration of episodic/semantic memories. Enables higher-level cognitive functions.
    - **FAISS Fast Vector Search**: Cosine similarity-based memory retrieval in milliseconds, 1000+ queries/sec throughput.
    - **Zenoh Distributed Communication Integration**: Real-time memory sharing between nodes, PTP-synchronized timestamps, automatic cleanup.
    - **Performance**: Store latency <10ms, search latency <5ms, supports up to 10,000 entries.

## 3. Infrastructure as Code (IaC) for Environment Management ⭐ NEW (December 20, 2025)

EvoSpikeNet adopts **Infrastructure as Code (IaC)**, achieving 100% environment reproducibility. A comprehensive environment management system integrating Terraform, Ansible, Kubernetes, and Docker Compose is provided.

### Key IaC Features

- **100% Environment Reproducibility**: Automatically reproduce identical configurations in any environment
- **Multi-environment Support**: Clear separation of Dev/Staging/Production environments
- **One-command Setup**: Complete automation from environment validation to deployment
- **Kubernetes Ready**: Large-scale production deployment with autoscaling
- **Integrated Health Checks**: Automatic health verification for all services

### Quick Start (IaC)

```bash
# 1. Environment validation and automatic setup
make env-setup

# 2. Build environment with Terraform
make terraform-init
make terraform-apply

# 3. Start Docker services
make docker-up

# 4. Health check
make health-check
```

### IaC Tool Integration

| Tool                   | Purpose           | Key Features                                                                         |
| ---------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| **Terraform**          | Infrastructure    | Docker network/volume management, automatic .env generation, health check generation |
| **Ansible**            | System Setup      | Automated Docker/Python/dependency installation (20+ tasks)                          |
| **Kubernetes**         | Production Deploy | StatefulSet, HPA (3-10 replicas), Ingress, autoscaling                               |
| **Environment Script** | Validation        | Python≥3.9, Docker, disk≥10GB, port availability checks                              |

### IaC-related Commands

```bash
# Environment Management
make env-setup         # Setup and validate environment
make env-validate      # Validation only

# Terraform
make terraform-init    # Initialize
make terraform-plan    # Show execution plan
make terraform-apply   # Apply infrastructure
make terraform-destroy # Destroy infrastructure

# Ansible
make ansible-setup     # System setup with Ansible

# Kubernetes
make k8s-deploy        # Deploy to Kubernetes

# Docker
make docker-build      # Build images
make docker-up         # Start services
make docker-down       # Stop services
make docker-logs       # View logs
```

### Environment Types

| Environment | GPU | API Port | UI Port | Purpose                  |
| ----------- | --- | -------- | ------- | ------------------------ |
| Development | CPU | 8000     | 8050    | Local development        |
| Staging     | GPU | 8100     | 8150    | Verification environment |
| Production  | GPU | 8200     | 8250    | Production operation     |

For details, see [docs/INFRASTRUCTURE_AS_CODE.en.md](docs/INFRASTRUCTURE_AS_CODE.en.md).

## 4. Launching Web UI

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
| **L5 Self-Evolution Plan**          | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.md)     | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md)     | Detailed plan for L5-level self-evolution functions ⭐           |
| **L5 Feature Breakdown**            | [L5_FEATURE_BREAKDOWN.md](docs/L5_FEATURE_BREAKDOWN.md)                               | [L5_FEATURE_BREAKDOWN.en.md](docs/L5_FEATURE_BREAKDOWN.en.md)                               | Detailed breakdown and implementation policy of L5 features ⭐   |
| **LLM Integration Strategy**        | [LLM_INTEGRATION_STRATEGY.md](docs/LLM_INTEGRATION_STRATEGY.md)                       | [LLM_INTEGRATION_STRATEGY.en.md](docs/LLM_INTEGRATION_STRATEGY.en.md)                       | Strategy for integrating Large Language Models                  |
| **AEG-Comm Implementation Plan**    | [AEG_COMM_IMPLEMENTATION_PLAN.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.md)               | [AEG_COMM_IMPLEMENTATION_PLAN.en.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.en.md)               | Implementation plan for intelligent communication control ⭐ NEW |
| **Simulation Recording Guide**      | [SIMULATION_RECORDING_GUIDE.md](docs/SIMULATION_RECORDING_GUIDE.md)                   | [SIMULATION_RECORDING_GUIDE.en.md](docs/SIMULATION_RECORDING_GUIDE.en.md)                   | How to use data recording/analysis application functions ⭐      |
| **Simulation Recording README**     | [SIMULATION_RECORDING_README.md](docs/SIMULATION_RECORDING_README.md)                 | [SIMULATION_RECORDING_README.en.md](docs/SIMULATION_RECORDING_README.en.md)                 | Overview of recording function                                  |
| **Spike Communication Analysis**    | [SPIKE_COMMUNICATION_ANALYSIS.md](docs/SPIKE_COMMUNICATION_ANALYSIS.md)               | [SPIKE_COMMUNICATION_ANALYSIS.en.md](docs/SPIKE_COMMUNICATION_ANALYSIS.en.md)               | Analysis methods for spike communication                        |
| **Pipeline Analysis**               | [distributed_brain_pipeline_analysis.md](docs/distributed_brain_pipeline_analysis.md) | [distributed_brain_pipeline_analysis_en.md](docs/distributed_brain_pipeline_analysis_en.md) | Detailed analysis of distributed brain pipeline                 |
| **Documentation Update Summary**    | [DOCUMENTATION_UPDATE_SUMMARY.md](docs/DOCUMENTATION_UPDATE_SUMMARY.md)               | [DOCUMENTATION_UPDATE_SUMMARY.en.md](docs/DOCUMENTATION_UPDATE_SUMMARY.en.md)               | Documentation update history                                    |

### 📁 Other Documents

- **Architecture Diagrams**: `docs/architecture/` directory
- **SDK Details**: `docs/sdk/` directory
