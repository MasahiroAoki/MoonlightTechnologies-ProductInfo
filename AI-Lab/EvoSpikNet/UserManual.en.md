# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet Dashboard User Manual

## 1. Introduction

This document explains how to use each feature of the `EvoSpikeNet Dashboard` and the technical implementation behind them. This dashboard is a tool for interactively running and visualizing the various functions of the EvoSpikeNet framework from a browser.

## 2. Setup and Launch

This project fully utilizes Docker Compose to launch the application and its required services (the Milvus database) simultaneously.

### 2.1. Prerequisites
- Docker
- Docker Compose (v2 or later)
- (For GPU use) NVIDIA Container Toolkit

### 2.2. How to Launch
First, build the Docker image by running the following command in the project's root directory:
```bash
docker compose build
```
Then, launch the web UI and its related services using the following scripts:

```bash
# To run in GPU mode:
./scripts/run_frontend_gpu.sh

# To run in CPU mode:
./scripts/run_frontend_cpu.sh
```
Access the dashboard at `http://localhost:8050`.

## 3. Main Interface

The dashboard has been refactored into a multi-page application with a single access point. You can access each page via the navigation bar at the top of the screen.

- **Home:** The default page, which displays this user manual.
- **SNN Data Creation:** Generates sample spike data for the 4-layer SNN model.
- **Model & Spike Visualization:** Visualizes SNN firing activity (raster plots) and model structures (graphs).
- **EvoSpikeNetLM:** Handles training and inference (prompting) for the standard language model.
- **SpikingEvoSpikeNetLM:** Handles training, inference, and visualization of internal states (spikes, attention) for the spike-based language model.
- **RAG System:** Manages and interacts with the Retrieval-Augmented Generation system. This page features two main tabs:
    - **Query:** Perform query-based retrieval and generation.
    - **Data CURD:** A comprehensive interface for managing the knowledge base in Milvus. You can **C**reate (add), **U**pdate (edit), **R**ead (view), and **D**elete documents directly from a data table.
- **Multi-Modal LM:** Trains and runs inference for the multi-modal model that handles both images and text.
- **Text Classifier:** Trains and runs inference for the text classification model.
- **System Utilities:** Executes system-level commands, such as running the test suite or performing data distillation.

---

## 4. Detailed Feature Explanations

### 4.1. Standard SNN Simulation
- **Purpose:** Simulates the behavior of a core SNN model composed of basic LIF neurons and sparse synaptic connections.
- **Output:** A raster plot showing the firing activity of the output layer neurons.

### 4.2. Entangled Synchrony Layer
- **Purpose:** Verifies the behavior of a special layer inspired by quantum entanglement.
- **Output:** A heatmap showing the generated synchronous spike pattern.

### 4.3. Hardware Fitness Evaluator
- **Purpose:** Demonstrates a fitness function that considers hardware metrics like power consumption.
- **Output:** A single calculated fitness score.

### 4.4. Text Classification
- **Purpose:** Classifies the sentiment of an input sentence in real-time using a pre-trained model.
- **Output:** The predicted sentiment label and confidence score.

### 4.5. LM Training (Advanced)
- **Purpose:** Flexibly trains the `EvoSpikeNetLM` language model on various data sources like Wikipedia.
- **Logical Implementation:** Asynchronously runs the training script via `subprocess.Popen` and polls the log file with `dcc.Interval` to display real-time progress.


---

## 5. Detailed Page Functions

The features on each page are a reorganization of the tabs and functions from the previous UI. For example, the "System Utilities" page contains the command execution features (like running tests and data distillation) that were formerly in the "System & Data" tab. For specific operations on each page, please follow the on-screen instructions.

---
(Remaining sections omitted as they are similar to README.md)