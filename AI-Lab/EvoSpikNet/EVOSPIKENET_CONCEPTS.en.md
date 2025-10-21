# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet: Key Concepts

This document provides technical details on the more advanced and unique concepts that form the core of the EvoSpikeNet framework.

---

## 1. Core SNN Architecture (`evospikenet/core.py`)

The `evospikenet.core` module defines the fundamental components for the stateful SNN model at the heart of EvoSpikeNet. Initially designed to process single data streams, it has been fully extended to support **batch processing** for efficient training.

- **`SNNModel`**: A high-level container that bundles multiple neuron layers and synapses to run time-step-based simulations. It automatically detects the input data's dimensionality, transparently handling both single instances (2D tensors) and batched data (3D tensors).
- **`LIFNeuronLayer`**: Implements a layer of Leaky Integrate-and-Fire (LIF) neurons. Its internal state, the membrane potential (`potential`), is dynamically allocated according to the batch size, enabling parallel processing of multiple inputs.
- **`SynapseMatrixCSR`**: Represents synaptic connections efficiently using sparse CSR-formatted matrices. The `forward` method in this class has also been enhanced to handle both single and batched inputs.

This support for batch processing is a critical foundation for applying modern deep learning techniques, such as self-supervised learning, to SNNs.

---

## 2. Self-Supervised Learning (SSL) (`evospikenet/ssl.py`)

The `evospikenet.ssl` module implements self-supervised learning mechanisms to learn feature representations from unlabeled data. This allows for pre-training SNNs on large amounts of data, improving performance and data efficiency for downstream tasks.

### 2.1. ContrastiveSelfSupervisedLayer

Inspired by research like SimCLR, the `ContrastiveSelfSupervisedLayer` is the core module for applying contrastive learning to SNNs.

#### Principle of Operation
1.  **Data Augmentation and Pair Generation**: Different augmented views generated from the same data sample are "positive pairs," while other samples in the batch are "negative pairs."
2.  **Feature Extraction and Projection**: The SNN model generates feature embeddings from spike trains, which are then passed through a "projection head."
3.  **Learning with NT-Xent Loss**: The NT-Xent loss function is used to maximize the similarity between positive pairs and minimize it between negative pairs. The loss is backpropagated using surrogate gradients.

This process enables the model to learn high-level feature representations from input data without requiring any labels.

---

## 3. Advanced Control Systems (`evospikenet/control.py`)

The `evospikenet.control` module provides mechanisms to control the learning and behavior of the SNN from an external, meta-level.

### 3.1. MetaSTDP (Plasticity Control via Meta-Learning)

The `MetaSTDP` class functions as a meta-learning agent that dynamically adjusts the learning-related parameters (e.g., membrane potential decay rate) of all neurons based on a global reward signal.

### 3.2. AEG (Activity-driven Energy Gating)

The `AEG` class is a mechanism that dynamically manages energy according to the activity of neuron groups, gating spikes when energy is low and replenishing it based on external rewards.

---

## 4. Quantum-Inspired Self-Organization (`evospikenet/annealing.py`)

The `GraphAnnealingRule` class, found in `evospikenet/annealing.py`, is a quantum-inspired algorithm designed to optimize a graph's structure using simulated annealing. It maps the graph to an Ising model and seeks a low-energy configuration, revealing hierarchical structures and community patterns.

---

## 5. RAG System and Data Management (`evospikenet/rag_milvus.py`)

The Retrieval-Augmented Generation (RAG) system integrates external knowledge into the language model by retrieving relevant documents from a Milvus vector database. The `rag_milvus.py` module manages all interactions with this database.

### 5.1. CURD Operations
The system provides a full suite of data management functions, exposed through the "Data CURD" tab in the UI:
- **`get_all_data()`**: Retrieves and displays all documents from the Milvus collection.
- **`add_document()`**: Adds a new text document, which is automatically converted to a vector embedding.
- **`update_document()`**: Modifies the text and source of an existing document.
- **`delete_document()`**: Removes a document from the collection by its ID.

This allows for direct, real-time management of the knowledge base without interacting with intermediate files like JSON.

### 5.2. LLM Integration
The RAG module also integrates with various Large Language Models (LLMs) to generate responses based on the retrieved documents. The system now includes a fully functional client for the **Grok API**, in addition to other models like OpenAI and Huggeing Face.