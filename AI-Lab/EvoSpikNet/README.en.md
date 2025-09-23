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
- **Text Processing (Phase 4):**
    - Conversion of text data into spike trains using `WordEmbeddingLayer`, `PositionalEncoding`, and `RateEncoder`.
- **Quantum-Inspired Features (Phase 3):**
    - `EntangledSynchronyLayer`: A special layer that dynamically controls the synchronization of neuron groups based on context.
    - `HardwareFitnessEvaluator`: A fitness evaluation function for evolutionary algorithms that considers hardware performance metrics.
- **Experimental Gradient-Based Learning (Phase 6):**
    - `examples/run_gradient_training_demo.py`: A self-contained demo showing how to train an SNN using Surrogate Gradients without compromising core library stability.
- **LLM-based Data Distillation:**
    - `evospikenet/distillation.py`: A feature to generate high-quality synthetic data for testing and training using Large Language Models (LLMs). It adopts a common interface design for flexible backend support.
- **Full-fledged SNN Language Model (`SpikingEvoSpikeNetLM`):**
    - A Transformer-based language model, built on `snnTorch`, that actually operates on spikes.
    - `TAS-Encoding`: An encoding layer that converts text into time-adaptive spike trains.
    - `ChronoSpikeAttention`: A hybrid attention mechanism that operates in the spike domain.
    - `MetaSTDP` / `AEG`: Advanced learning and control mechanisms considering meta-learning and energy efficiency.
    - **Training, Evaluation, and Tuning:** Provides a complete workflow from model training, perplexity evaluation, to hyperparameter tuning via scripts (`examples/train_spiking_evospikenet_lm.py`, `scripts/run_hp_tuning.sh`).
    - **UI Visualization:** Allows visualization of internal spike activity and attention weights during text generation from the Web UI.

## 3. Web UI
A web frontend is available for interactively performing simulations, adjusting parameters, and visualizing results from a browser. Advanced features such as **training, evaluating, and visualizing the internal operations of the SNN language model** can also be executed and checked from the UI. Launch it with the following script and access `http://localhost:8050` in your browser.

```bash
# Launch the Web UI in CPU mode
./scripts/run_frontend_cpu.sh
```

## 4. Environment Setup and Execution using Docker (Recommended)

This project allows for easy setup of CPU or GPU execution environments using Docker.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using GPU mode)

### Build the Environment
First, build the Docker image by running the following command in the project's root directory:
```bash
docker compose build
```

### Selecting Execution Mode (CPU / GPU)
The execution scripts in the `scripts` directory are divided into CPU mode (`*_cpu.sh`) and GPU mode (`*_gpu.sh`). Please select the appropriate script for your purpose.

---

### Development Environment
Launches a development container and connects to an interactive shell.

- **GPU Mode:**
  ```bash
  ./scripts/run_dev_gpu.sh
  ```
- **CPU Mode:**
  ```bash
  ./scripts/run_dev_cpu.sh
  ```
To exit the container, run `exit`. The container will continue to run in the background. To stop it completely, run `docker stop evospikenet-dev-gpu` (or `...-cpu`).

---

### Running Tests
Runs the test suite using `pytest`.

- **GPU Mode:**
  ```bash
  ./scripts/run_tests_gpu.sh
  ```
- **CPU Mode:**
  ```bash
  ./scripts/run_tests_cpu.sh
  ```

---

### Running the Sample Program
Runs the sample program `example.py`.

- **GPU Mode:**
  ```bash
  ./scripts/run_prod_gpu.sh
  ```
- **CPU Mode:**
  ```bash
  ./scripts/run_prod_cpu.sh
  ```
---

### Running the Web UI
Launches the interactive Web UI.

- **GPU Mode:**
  ```bash
  ./scripts/run_frontend_gpu.sh
  ```
- **CPU Mode:**
  ```bash
  ./scripts/run_frontend_cpu.sh
  ```
After launching, access `http://localhost:8050` in your browser.

## 5. Local Development Environment Setup (Advanced)

**Note:** This method assumes that the correct versions of Python and CUDA are installed on your local machine and that the paths are set correctly. For most users, **execution using Docker (Section 4) is recommended** to avoid issues due to environmental differences.

This framework can run on either a CPU or a GPU (NVIDIA). For maximum performance, an **NVIDIA GPU and CUDA** setup is recommended.

1.  **Clone the repository and navigate into it**
2.  **Create and activate a Python virtual environment**
3.  **Install dependencies:**
    - **For GPU:** `PyTorch` must be installed corresponding to your CUDA version.
      ```bash
      # PyTorch for CUDA 12.1
      pip install torch --index-url https://download.pytorch.org/whl/cu121
      ```
    - **For CPU:**
      ```bash
      pip install torch
      ```
    - **Common dependencies:**
      ```bash
      # Install the main package
      pip install -e .
      # Install dependencies for testing
      pip install -e '.[test]'
      ```
4.  **Verify installation:**
    ```bash
    # Test on GPU
    DEVICE=cuda pytest
    # Test on CPU
    DEVICE=cpu pytest
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
| `frontend/` | Source code for the Dash-based Web UI application. |
| `tests/` | Unit tests using `pytest`. |
| `scripts/` | Shell scripts for CPU/GPU to simplify development, testing, and execution. |
| `examples/` | Sample programs demonstrating specific uses of the framework. |
| `Dockerfile` | The blueprint for the Docker image that defines the project's execution environment. |
| `docker-compose.yml` | Docker service definition for CPU mode. |
| `docker-compose.gpu.yml`| Additional configuration file for GPU mode. |
| `pyproject.toml` | Defines project metadata and Python dependencies. |
| `REMAINING_FEATURES.md`| The project's feature implementation status and future roadmap. |
| `README.md` | This file. Describes the project overview, setup instructions, usage, etc. |
