# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet: Key Concepts

This document provides technical details on the more advanced and unique concepts that form the core of the EvoSpikeNet framework.

---

## 1. Control System (`evospikenet/control.py`)

The `evospikenet.control` module provides mechanisms to control the learning and behavior of the SNN from an external, meta-level.

### 1.1. MetaSTDP (Plasticity Control via Meta-Learning)

The `MetaSTDP` class functions as a kind of meta-learning agent that dynamically adjusts learning-related parameters of all neurons based on the overall performance of the model (given as a reward signal).

#### Principle of Operation
1.  **Target Parameter:** In the current implementation, the target for adjustment is the **membrane potential decay rate (`beta`)** of all `snn.Leaky` (LIF neuron) layers present in the model. `beta` is a crucial parameter that determines the length of a neuron's "memory" (a value closer to 1 means a longer memory).
2.  **Reward-Based Updates:**
    -   At each training step, a reward calculated from the loss or other metrics is passed to the `update` method.
    -   `MetaSTDP` internally maintains a baseline of the reward (a moving average of its expected value).
    -   If the current reward exceeds the baseline (meaning the loss is smaller than expected), it updates `beta` in the direction of `1` (lengthening the memory).
    -   Conversely, if the reward is below the baseline, it updates `beta` to be smaller (shortening the memory).
3.  **Learning Rate Annealing:**
    -   `MetaSTDP` has its own learning rate (`eta`). This `eta` determines how much `beta` changes.
    -   To stabilize learning, this `eta` is slightly decayed (annealed) each time the `update` method is called. This allows `beta` to fluctuate significantly in the early stages of learning and then converge to a stable state as the fluctuations become smaller over time.

This mechanism allows the model to self-adjust the fundamental properties of its neurons while observing its own overall performance.

### 1.2. AEG (Activity-driven Energy Gating)

The `AEG` class is a mechanism that dynamically manages energy according to the activity of neuron groups.

#### Principle of Operation
1.  **Concept of Energy:** Each neuron (or group of neurons) maintains an internal "energy" level. This energy varies from `255` (maximum) to `0`.
2.  **Consumption by Firing:** When a neuron fires (spikes), its energy decreases based on the `consumption_rate`. This consumption is weighted by the "importance" of each spike. This makes firing for less important information more energy-costly and thus more likely to be suppressed.
3.  **Firing Suppression by Energy (Gating):** If a neuron's energy falls below a certain `threshold`, it is temporarily unable to fire. Even if its membrane potential exceeds the firing threshold, no spike is output (it is gated).
4.  **Energy Supply by Reward:** When a reward signal is given from an external source via the `supply` method, the energy levels of all neurons are restored based on the `supply_rate`.

This system allows the network to use the resource of energy efficiently and is expected to self-adjust to focus on processing high-importance information. It is used in the `SpikingEvoSpikeNetLM` model to suppress unnecessary spikes and increase computational efficiency.
