# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet - A Distributed, Evolutionary Neuromorphic Framework

## 1. Project Overview

EvoSpikeNet is a simulation framework for scalable and energy-efficient "Dynamic Spiking Neural Networks (SNNs)" that mimic brain plasticity, leveraging the parallel computing power of GPUs or CPUs. By eliminating floating-point operations and specializing in integer arithmetic and sparse computation, it aims for real-time time-series data processing and edge AI applications.

## 2. Key Implemented Features

Based on the phases defined in `REMAINING_FEATURES.md`, this framework implements the following major feature sets:

- **Core SNN Engine (Phase 1):**
    - `LIFNeuronLayer`: An LIF neuron layer based on integer arithmetic.
    - `SynapseMatrixCSR`: Memory-efficient sparse synaptic connections.
    - `SNNModel`: A core model that integrates multiple layers and synapses to run simulations.
- **Dynamic Evolution and Visualization (Phase 2):**
    - `STDP`: Online learning through Spike-Timing-Dependent Plasticity (STDP).
    - `DynamicGraphEvolutionEngine`: Dynamic graph evolution that performs synapse creation and deletion.
    - `InsightEngine`: Visualizes the network's internals, such as raster plots of firing activity and connection structure graphs.
- **Energy-Driven Computing (Phase 3):**
    - `EnergyManager`: Manages neuron firing based on energy levels to achieve more biologically realistic simulations.
- **Multi-Modal Processing (Text & Image):**
    - `WordEmbeddingLayer`, `PositionalEncoding`, `RateEncoder` for converting text data into spike trains.
    - Implementation of a multi-modal language model (`MultiModalEvoSpikeNetLM`) utilizing `torchvision` for image processing.
- **Quantum-Inspired Features (Phase 3):**
    - `EntangledSynchronyLayer`: A special layer that dynamically controls the synchronization of neuron groups based on context.
    - `HardwareFitnessEvaluator`: A fitness evaluation function for evolutionary algorithms that considers hardware performance metrics.
    - `GraphAnnealingRule`: A rule that uses simulated annealing to optimize graph structures and promote self-organization.
- **LLM-based Data Distillation:**
    - `evospikenet/distillation.py`: A feature to generate high-quality synthetic data for testing and training using Large Language Models (LLMs).
- **Full-fledged SNN Language Model (`SpikingEvoSpikeNetLM`):**
    - A Transformer-based language model, built on `snnTorch`, that actually operates on spikes.
    - `TAS-Encoding`: An encoding layer that converts text into time-adaptive spike trains.
    - `ChronoSpikeAttention`: A hybrid attention mechanism that operates in the spike domain.
- **Retrieval-Augmented Generation (RAG) Feature:**
- **Self-Supervised Learning (SSL) Feature**:
    - `evospikenet/ssl.py`: Implements a self-supervised learning layer with contrastive learning (NT-Xent loss) to enable representation learning from unlabeled data.

## 3. Web UI
A web frontend is available for interactively performing simulations, adjusting parameters, and visualizing results from a browser. The Docker scripts will launch the Milvus database alongside the UI.

The new UI has been rebuilt as a multi-page application where each feature is on a separate page. You can access the dashboard at `http://localhost:8050` and navigate to all features from the navigation bar.

You can launch it using the following commands:
```bash
# Launch the Web UI and Milvus in GPU mode
./scripts/run_frontend_gpu.sh

# Launch the Web UI and Milvus in CPU mode
./scripts/run_frontend_cpu.sh
```

## 4. Environment Setup and Execution using Docker (Recommended)

This project allows for easy setup of CPU or GPU execution environments using Docker. The **Milvus vector database is also launched as a container**, providing a complete development and execution environment.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2 or later, `docker compose` command)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using GPU mode)

### Build and Run the Environment
First, build the Docker image by running the following command in the project's root directory:
```bash
docker compose build
```
Then, use the scripts in the `scripts` directory to launch the desired services. For example, to run the Web UI in GPU mode, use the following command. This will start both the `frontend` service and its `milvus-standalone` dependency.
```bash
./scripts/run_frontend_gpu.sh
```
You can then access the dashboard at `http://localhost:8050`.

### Other Commands
- **Development Environment (GPU):** `./scripts/run_dev_gpu.sh`
- **Run Tests (CPU):** `./scripts/run_tests_cpu.sh`

## 5. Local Development Environment Setup (Not Recommended)

Without Docker, you must manually start a Milvus database and install Python dependencies. To avoid issues from environment differences, **using Docker (Section 4) is strongly recommended.**

1.  **Start Milvus:** Launch Milvus separately using Docker or another method.
2.  **Install dependencies:**
    ```bash
    # Install PyTorch (adjust for your environment)
    pip install torch
    # Install the project
    pip install -e .[test]
    ```

## 6. Basic Execution Steps

The following is sample code to build and run a simple two-layer SNN model using `EvoSpikeNet`.

```python
import torch
import os
from evospikenet.core import LIFNeuronLayer, SynapseMatrixCSR, SNNModel

def run_simple_snn():
    """
    A sample function demonstrating the basic usage of EvoSpikeNet.
    """
    # Select device from environment variable, or default to CUDA/CPU
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. Define model components
    layers = [
        LIFNeuronLayer(num_neurons=784, device=device),
        LIFNeuronLayer(num_neurons=10, device=device)
    ]
    synapses = [
        SynapseMatrixCSR(pre_size=784, post_size=10, connectivity_ratio=0.05, device=device)
    ]

    # 2. Build the SNN model
    model = SNNModel(layers=layers, synapses=synapses)
    print("SNNModel created successfully.")

    # 3. Generate dummy input data
    time_steps = 20
    input_size = 784
    input_spikes = torch.randint(0, 2, (time_steps, input_size), dtype=torch.int8, device=device)
    print(f"\nCreated random input spikes with shape: {input_spikes.shape}")

    # 4. Run the simulation
    print("Running simulation...")
    output_spikes = model.forward(input_spikes)
    print("Simulation finished.")

    # 5. Check the results
    print(f"Output spikes shape: {output_spikes.shape}")
    print(f"Total spikes in output: {torch.sum(output_spikes)}")

if __name__ == '__main__':
    run_simple_snn()
```

## 7. Project Structure

| Path | Description |
| :--- | :--- |
| `evospikenet/` | Main source code for the framework. |
| `frontend/` | Source code for the Dash-based Web UI. `app.py` is the entry point, and the `pages/` directory contains each feature page. |
| `tests/` | Unit tests using `pytest`. |
| `scripts/` | Shell scripts for CPU/GPU to simplify development, testing, and execution. |
| `examples/` | Sample programs demonstrating specific uses of the framework. |
| `data/` | Sample datasets and knowledge bases for RAG (`knowledge_base.json`). |
| `Dockerfile` | The blueprint for the Docker image that defines the project's execution environment. |
| `docker-compose.yml` | Docker service definition for CPU mode, including the Milvus service. |
| `docker-compose.gpu.yml`| Additional configuration file for GPU mode. |
| `pyproject.toml` | Defines project metadata and Python dependencies. |
| `REMAINING_FEATURES.md`| The project's feature implementation status and future roadmap. |
| `README.md` | This file. Describes the project overview, setup instructions, usage, etc. |

## 8. Roadmap

EvoSpikeNet aims to achieve more advanced and energy-efficient neuromorphic computing through continuous research and development. See `FEATURE_ANALYSIS.md` for a detailed breakdown.

## 9. Codebase Documentation

As of October 2025, the entire codebase has been thoroughly reviewed and annotated. All Python source files (`.py`) now include:
- A file header summarizing the module's functionality, copyright, and author information.
- Detailed comments for each function and class, explaining their purpose, arguments, and behavior.

This effort was undertaken to improve the readability, maintainability, and overall quality of the framework, making it more accessible for future development and collaboration.