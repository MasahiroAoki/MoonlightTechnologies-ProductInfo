# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# Project Feature Implementation Status

This document tracks the features and implementation status of the EvoSpikeNet project.

---
**Notes:**
- **Regarding Phase 8 (Distributed Grid):** An attempt was made to implement this feature using Python's `multiprocessing` library, but it faced an unresolvable deadlock issue (permanent process halt). This was potentially due to the current execution environment. To avoid leaving unstable code in the project, the implementation of this feature was abandoned to prioritize project stability. The related code has been deleted.
- **Regarding GraphAnnealingRule:** `GraphAnnealingRule` (multi-scale self-organization), proposed as part of the quantum-inspired features in Phase 3, was found to cause a difficult-to-debug low-level crash during the conversion between PyTorch sparse tensors and the `networkx` library. To ensure the overall stability of the project, this feature has been temporarily removed.
- **Regarding Spiking Transformer:** The initial Phase 5 implemented a standard (float-based) Transformer block due to the complexity of a pure spike-based implementation. Later, based on a detailed design proposal from the user, a full-fledged hybrid spiking language model (`SpikingEvoSpikeNetLM`) was newly implemented as **Phase SNN-LM**, utilizing `snnTorch`.

---

| Phase | Feature | Implementation Status | Test/Validation Script | Test Status |
| :--- | :--- | :---: | :--- | :---: |
| **Phase 1** | **Core SNN Engine** | | | |
| | `LIFNeuronLayer`, `SynapseMatrixCSR`, `SNNModel` | ✔️ | `tests/test_core.py` | ✅ Success |
| | Validation Program | ✔️ | `example.py` | ✅ Success |
| **Phase 2** | **Dynamic Graph Evolution & Insight Engine** | | | |
| | Plasticity Rules (`STDP`, `Homeostasis`) | ✔️ | `tests/test_plasticity.py` | ✅ Success |
| | `MetaPlasticity` | ✔️ | `tests/test_evolution.py` | ✅ Success |
| | `GraphUpdateManager` | ✔️ | `tests/test_evolution.py` | ✅ Success |
| | Monitoring/Visualization (`DataMonitorHook`, `InsightEngine`) | ✔️ | `tests/test_insight.py` | ✅ Success |
| | Validation Program | ✔️ | `examples/run_plasticity_demo.py` | ✅ Success |
| **Phase 3** | **Energy-Driven / Quantum-Inspired** | | | |
| | Energy-Driven Computing (`EnergyManager`) | ✔️ | `tests/test_energy.py` | ✅ Success |
| | Operation Verification Demo | ✔️ | `examples/run_energy_demo.py` | ✅ Success |
| | Quantum-Inspired Features (2/3 implemented) | ✔️ | `tests/test_quantum_layers.py`, `tests/test_fitness.py` | ✅ Success |
| **Phase 4** | **Text Learning - Encoding Methods** | | | |
| | Word Embedding Layer | ✔️ | `tests/test_text.py` | ✅ Success |
| | Encoding Modules (`RateEncoder`, `LatencyEncoder`) | ✔️ | `tests/test_text.py` | ✅ Success |
| | Positional Encoding | ✔️ | `tests/test_text.py` | ✅ Success |
| **Phase 5** | **Text Learning - Spiking Transformer** | | | |
| | Spiking Self-Attention | ✔️ | `tests/test_transformer.py` | ✅ Success |
| | Network Components (Residual Conn., Layer Norm) | ✔️ | `tests/test_transformer.py` | ✅ Success |
| **Phase 6** | **Text Learning - Gradient-Based Learning** | | | |
| | Surrogate Gradient Implementation | ✔️ | `evospikenet/surrogate.py`, `tests/test_surrogate.py` | ✅ Success |
| | Loss Function and Decoding | ✔️ | `examples/run_gradient_training_demo.py` | ✅ Success |
| **Phase 7** | **Text Learning - Integration and Experiments** | | | |
| | Model Integration | ✔️ | `evospikenet/models.py`, `tests/test_models.py` | ✅ Success |
| | Running Experiments | ✔️ | `examples/run_text_classification_experiment.py` | ✅ Success |
| **Phase 8** | **Distributed Neuromorphic Grid** | | | |
| | Distributed Model (`DistributedEvoSpikeNet`) | ❌ | (Not implemented) | ❌ Not run |
| | Asynchronous Communication (`SpikeCommunicator`) | ❌ | (Not implemented) | ❌ Not run |
| **Phase EX**| **LLM Data Distillation** | | | |
| | Data Distillation Module (`DataDistiller`) | ✔️ | `evospikenet/distillation.py` | ✅ Success |
| | Data Generation Demo | ✔️ | `examples/generate_distilled_dataset.py` | ✅ Success |
| **Phase 9** | **Frontend - Basic UI and Visualization** | | | |
| | Basic Dash App Skeleton | ✔️ | `frontend/app.py` | ✅ Success |
| | SNN Execution and Result Visualization | ✔️ | `frontend/app.py` | ✅ Success |
| **Phase 10**| **Frontend - Interactive Control** | | | |
| | Parameter Input UI | ✔️ | `frontend/app.py` | ✅ Success |
| | Dynamic Simulation Update Feature | ✔️ | `frontend/app.py` | ✅ Success |
| **Phase 11**| **Frontend - Command Panel** | | | |
| | Data Gen/Test Execution UI | ✔️ | `frontend/app.py` | ✅ Success |
| | Backend Script Integration | ✔️ | `frontend/app.py` | ✅ Success |
| **Phase 12**| **Frontend - Advanced Training UI** | | | |
| | LM Training UI from External Data Sources | ✔️ | `frontend/app.py` | ✅ Success |
| | Training Log Streaming Display | ✔️ | `frontend/app.py` | ✅ Success |
| **Phase SNN-LM-DATA**| **SNN-LM - Data Pipeline** | | | |
| | External Data Loaders (`Wikipedia`, `Aozora`) | ✔️ | `evospikenet/dataloaders.py` | ✅ Success |
| | `TAS-Encoding` (Text to Spike Conversion) | ✔️ | `evospikenet/encoding.py`, `tests/test_encoding.py` | ✅ Success |
| **Phase SNN-LM-MODEL**| **SNN-LM - Model Architecture** | | | |
| | `ChronoSpikeAttention` (Hybrid) | ✔️ | `evospikenet/attention.py`, `tests/test_attention.py` | ✅ Success |
| | `SpikingTransformerBlock` | ✔️ | `evospikenet/attention.py`, `tests/test_attention.py` | ✅ Success |
| | `SpikingEvoSpikeNetLM` (Model Integration) | ✔️ | `evospikenet/models.py` | ✅ Success |
| | `AEG`, `MetaSTDP` (Control Mechanisms) | ✔️ | `evospikenet/control.py`, `tests/test_control.py` | ✅ Success |
| **Phase SNN-LM-TRAIN**| **SNN-LM - Training and Evaluation** | | | |
| | Training/Evaluation Script | ✔️ | `examples/train_spiking_evospikenet_lm.py` | ✅ Success |
| | Hyperparameter Tuning Script | ✔️ | `scripts/run_hp_tuning.sh` | ✅ Success |
| **Phase SNN-LM-VIZ**| **SNN-LM - Visualization** | | | |
| | Internal Spike Activity Visualization | ✔️ | `frontend/app.py` | ✅ Success |
| | Attention Weight Visualization | ✔️ | `frontend/app.py` | ✅ Success |

**Legend:**
*   ✔️: Implemented
*   ❌: Not implemented
*   ✅: Test Success
*   ⚠️: Not Verified (Test execution not possible due to environmental factors)
*   (N/A): Not Applicable

---
## Next Development Plan: Multi-Modal SNN

| Phase | Feature | Implementation Status | Test/Validation Script | Test Status |
| :--- | :--- | :---: | :--- | :---: |
| **Phase MM-1**| **Vision Encoder** | | | |
| | Convolutional SNN Encoder Implementation | ✔️ | `evospikenet/vision.py`, `tests/test_vision.py` | ✅ Success |
| **Phase MM-2**| **Multi-Modal Model Integration** | | | |
| | Text and Vision Feature Fusion | ✔️ | `evospikenet/models.py` | ✅ Success |
| | `MultiModalEvoSpikeNetLM` Construction | ✔️ | `tests/test_models.py`, `examples/train_multi_modal_lm.py` | ⚠️ **Verified** |
| | **Note:** | `train_multi_modal_lm.py` is a demo to confirm that the model's forward/backward pass can run with dummy data. Real data loading, model saving, and inference functions are not yet implemented. |
