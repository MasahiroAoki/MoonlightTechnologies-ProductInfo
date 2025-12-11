# Architectural Study Report for Distributed Brain Simulation

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Date:** 2025-12-04
**Project:** EvoSpikeNet
**Topic:** LLM Functional Specialization vs. Multi-modal Integration

## 1. Overview
In constructing the Distributed Brain Simulation, we studied the optimal configuration of LLM/SNN models for each node (brain region).
Specifically, we compared whether to **"link multiple specialized models for each function (Specialized)"** or **"use a single, massive multi-modal model (Unified)"**, or a combination thereof.

## 2. Comparative Study

| Feature                    | A. Specialized (Modular)                                                                                                    | B. Unified (Monolithic)                                                                                    |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Configuration**          | Visual, auditory, language, and motor cortices are independent models that cooperate over a network.                            | A single, massive model processes all inputs (images, audio, text) and handles all outputs.                |
| **Biological Plausibility**| **High**. The actual brain has functional localization in different regions, connected by a connectome.                      | **Low**. The entire brain is not a homogeneous neural network.                                             |
| **Computational Resources**| **Distributable**. Each node can be a lightweight SNN, allowing for parallel operation on edge devices or different servers. | **Centralized**. Requires a single powerful GPU server with massive VRAM, diminishing the benefits of distributed simulation. |
| **Communication Load**     | **Low**. Only "spikes" or "compressed embedding representations" are exchanged between nodes, conserving bandwidth.             | **High**. Raw input data (images, audio) must be sent to the central node, consuming network bandwidth.   |
| **Flexibility/Maintenance**| **High**. It's easy to update or replace a specific function (e.g., vision) without affecting the whole system.          | **Low**. Retraining to improve one function risks negatively impacting others (catastrophic forgetting).    |
| **Information Integration**| **Challenging**. Requires careful interface design to link different modalities (e.g., the word "red" with the visual of "red"). | **High**. Cross-modal learning occurs naturally within the model, making information integration easy.    |

## 3. Recommended Architecture: Hierarchical Hybrid Configuration

To maximize the strengths of EvoSpikeNet—"SNNs," "Distributed Processing (Zenoh)," and "Knowledge Distillation"—we recommend a hierarchical hybrid configuration: **"Specialized SNNs at the periphery, an integrated multi-modal model at the core."**

### 3.1. Configuration Details

#### Level 1: Sensory/Motor Cortices (Edge Nodes) - Specialized Spiking LLMs
These are the nodes that interface with the external world, requiring fast responses and efficient data compression.

*   **Visual Node**: Processes image input and converts it into visual features or spike trains.
*   **Auditory/Speech Node**: Specializes in speech recognition and synthesis.
*   **Motor Node**: Specializes in generating motor control commands.
*   **Characteristics**: Uses lightweight SNNs created by extracting specific capabilities from a Teacher model using the newly implemented "Knowledge Distillation" feature.

#### Level 2: Association/Prefrontal Cortices (Center Node / PFC) - Integrated Multi-modal Spiking LLM
The command center that integrates information from the sensory cortices, makes decisions, references memory, and performs long-term planning.

*   **PFC (Prefrontal Cortex) Node**: Integrates and processes visual, auditory, and language information.
*   **Characteristics**: A somewhat larger `Multi-modal Spiking LLM` is deployed here, with "reasoning and integration capabilities" distilled from a powerful Teacher model like those from Hugging Face.

### 4. Strategy for Utilizing the Knowledge Distillation Feature

The `Distillation Control Center` implemented in this project serves as the factory for realizing this architecture.

1.  **Creating the PFC Model**:
    *   Teacher: A high-performance model like Llama-2-13b.
    *   Student: `Multi-modal Spiking LLM`.
    *   Objective: To imbue it with advanced reasoning and multi-modal integration capabilities.

2.  **Creating Models for Edge Nodes**:
    *   Teacher: The same high-performance model (or a dedicated image/audio model).
    *   Student: `Spiking LM` (Text Only), `Audio Spiking LLM`, `Speech Spiking LLM`.
    *   Objective: To specialize in specific I/O processing and reduce parameter count for high speed.

## 5. Conclusion

The optimal configuration maintains the fundamental principle of a distributed brain—"functional specialization"—while leveraging a multi-modal model as a hub to integrate them.
This strikes a balance between biological plausibility, computational efficiency, and advanced cognitive capabilities.
