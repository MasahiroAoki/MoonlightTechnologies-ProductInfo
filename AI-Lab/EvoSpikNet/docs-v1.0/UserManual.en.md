<!-- AUTO-TRANSLATED from docs/UserManual.md on 2025-12-21. Please review. -->
# EvoSpikeNet Dashboard User Manual

Last updated: 2025-12-09

Purpose: Quick reference describing the dashboard pages and their functions for operators, testers, and new contributors.

Navigation (high level): Home, Language & MultiModal Models, RAG & Knowledge, Advanced Systems, Tools & Analysis, Model Management, System Utilities, Tuning.

Highlights:
- Home: access project docs and README
- Language & MultiModal Models: pages for `SpikingEvoTextLM`, multimodal LM, vision/audio encoders, and text classifier
- RAG & Knowledge: hybrid retrieval (Milvus + Elasticsearch) and RAG chat UI with query-processing debug view
- Distributed Brain: configure and run Zenoh-based distributed simulations; node configuration and runtime monitoring
- Motor Cortex: robot learning pipeline with imitation learning, RL (SpikePPO), zero-shot generalization, live control and monitoring
- Tools & Analysis: visualization, model management (categorization, upload, filterable table), system utilities, and Optuna tuning integration

See the detailed manual pages for full workflow and advanced features.
# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet Dashboard User Manual

**Last Updated:** December 9, 2025

## Purpose and How to Use This Document
- Purpose: Give dashboard users a quick view of page roles and operations.
- Audience: Dashboard users, ops/validation members, new joiners.
- Read order: UI navigation and page functions → Detailed features as needed.
- Related links: Distributed brain implementation details implementation/PFC_ZENOH_EXECUTIVE.md; distributed brain script examples/run_zenoh_distributed_brain.py.

## 1. Introduction

This document describes how to use the `EvoSpikeNet Dashboard`. This dashboard provides a unified interface for visualizing and intuitively operating the complex functions of the EvoSpikeNet framework from a browser.

## 2. Setup and Launch

This project fully adopts Docker Compose, allowing you to start the application and all necessary backend services (API, Database, Milvus, etc.) at once.

### 2.1. Prerequisites
- Docker
- Docker Compose (v2 or later)
- (For GPU usage) NVIDIA Container Toolkit

### 2.2. How to Launch
In the project root directory, first run the following command to build the Docker image (for first-time use or updates). `sudo` might be required.
```bash
docker compose build
```
Next, run one of the following scripts to launch the Web UI and related services.

```bash
# For environments with available GPU
sudo ./scripts/run_frontend_gpu.sh

# For CPU-only environments
sudo ./scripts/run_frontend_cpu.sh
```
After that, the dashboard will be available at `http://localhost:8050`.

---

## 3. UI Navigation and Page Functions

You can access each function from the navigation menu at the top of the screen. Pages are organized by functionality.

### 3.1. Home
This is the default page displaying basic documents such as this user manual and the project README.

### 3.2. Language & MultiModal Models

- **SpikingEvoTextLM**:
    - Page for interacting with and training the SNN-based language model (`SpikingEvoTextLM`).
- **MultiModal LM**:
    - **Vision-Language**: Performs training and inference of multi-modal models combining images and text.
    - **Audio**: Integrates functions for audio transcription (ASR) and training audio models. (Formerly Audio Tools)
- **Vision Encoder**:
    - Performs standalone training and inference tests for the Spiking Vision Encoder for image recognition.
- **Audio Encoder**:
    - Performs standalone training and inference tests for the Spiking Audio Encoder for audio feature extraction.
- **Text Classifier**:
    - Handles SNN models for text classification tasks.

### 3.3. RAG & Knowledge

- **RAG System**:
    - Performs RAG (Retrieval-Augmented Generation) chat using hybrid search (Milvus + Elasticsearch). Knowledge base management is also done here.
    - **Query Processing Debug Feature** ⭐ NEW: When the "Show Query Processing Details" checkbox is enabled, the following information is displayed:
        - Query analysis (language detection, extracted keywords)
        - Vector search results (L2 distance scores, document previews)
        - Keyword search results (BM25 scores, document previews)
        - RRF fusion process (ranking, score calculation)
        - Generation details (context length, prompt preview, response type)
    - **Knowledge Base Management**: Add, edit, delete documents; real-time character counter; batch delete functionality
- **Data Creation**:
    - Generates and transforms data used for simulation and training.

### 3.4. Advanced Systems

- **Distributed Brain**:
    - The main console for configuring, executing, and monitoring real-time distributed brain simulations.
- **Motor Cortex**:
    - Motor cortex simulation for robot control. Manages the pipeline from imitation learning to reinforcement learning.

### 3.5. Tools & Analysis

- **Visualization**:
    - Uploads saved neuron data (`.pt` files) and visualizes spike activity and attention maps in detail.
- **Model Management** ⭐ UPDATED:
    - Manages model artifacts (list, download, upload) stored in the database.
    - **Model Classification System** (Added December 17, 2025):
        - **Brain Node Type**: Vision (Rank 1), Motor (Rank 2), Auditory (Rank 5), Speech (Rank 6), Executive (Rank 0), General
        - **Model Category**: 20+ types corresponding to each node (image classification, object detection, motion control, speech recognition, text generation, etc.)
        - **Model Variant**: Lightweight, Standard, High Accuracy, Realtime, Experimental
    - **Classification on Upload**: When uploading models, you can select node type, category, and variant in addition to LLM type. Categories are dynamically filtered based on the selected node type.
    - **Table Display**: Classification information is displayed in the model list table and can be filtered.
    - **Recommended Models by Node Type**: Display recommended models and use cases for each node type.
- **System Utilities**:
    - Utility functions for checking system status, clearing cache, running tests, etc.
- **Tuning**:
    - Executes automated hyperparameter optimization using `Optuna` and analyzes results.

---

## 4. Detailed Feature Description

### 4.1. Distributed Brain

The "Distributed Brain" page in the navigation menu is the most advanced feature of the framework. It manages and executes a distributed brain simulation composed of multiple processes (nodes) via the official Zenoh script `examples/run_zenoh_distributed_brain.py` (`run_distributed_brain_simulation.py` is a deprecated wrapper kept for backward compatibility).
For implementation details of PFC/Zenoh/ExecutiveControl, see [implementation/PFC_ZENOH_EXECUTIVE.md](implementation/PFC_ZENOH_EXECUTIVE.md).

- **Node Configuration Tab**: Defines the simulation architecture. Assigns different functional modules (e.g., "Language", "Vision") to run on different machines (local or remote via SSH) and assigns specific trained models to each node.
- **Brain Simulation Tab**: Interacts with the running simulation. Sends multi-modal prompts (text, image, audio) and monitors internal states of the entire distributed system in real-time, including communication paths, PFC energy levels, and individual node logs.

### 4.2. Motor Cortex

The "Motor Cortex" page provides a complete workflow for training adaptive robot motor systems based on an advanced 4-stage learning pipeline. This is orchestrated by the `evo_motor_master.py` backend script.

- **Configuration Tab**:
    - Here, you define the physical hardware of the robot in a YAML file. This includes defining motor groups, degrees of freedom, joint limits, and safety parameters. This configuration is used in all subsequent learning and simulation stages.

- **Learning Pipeline Tab**:
    - This is the main control center for training the agent. The workflow is divided into consecutive stages:
    1.  **Stage 1: Imitation Learning**: Starts the process by uploading a video file of a human performing the task. Clicking "Start Stage 1" runs a behavior cloning process to give the robot a basic understanding of movement.
    2.  **Stage 2: Real-world Reinforcement Learning**: The robot enters a self-improvement phase. Give a high-level goal (e.g., "pick up the cup and place it on the shelf") and click "Start Stage 2". The robot repeatedly practices the task, using reinforcement learning (`SpikePPO`) to acquire faster, smoother, and more successful movements.
    3.  **Stage 3: Zero-shot Generalization**: After Stage 2, you can test the robot's generalization ability by entering a completely new command in the text box and clicking "Attempt New Task".

- **Live Control Tab**:
    - After training, use this tab to monitor the live status of the trained agent and issue new commands.
    - You can also enable **Stage 4: Human Collaboration Mode**, where the robot uses force sensors to react to and assist human movements.

- **Monitoring**:
    - Throughout all stages, the "Master Log & Progress" panel provides a live stream of logs from the backend script. During Stage 2, the "Learning Progress" graph updates in real-time to show the agent's success rate and reward improvement.
