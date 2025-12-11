# Analysis of Node Pipeline Determination for Rank 0 Input in Distributed Brain Simulation

**Created:** 2025-12-06  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki

## 1. Overview

This document summarizes the analysis of how **Rank 0** (primarily the **PFC: Prefrontal Cortex** node) determines the subsequent processing nodes (pipeline) upon receiving external input in the Distributed Brain Simulation. The analysis is based on the source code.

The source codes analyzed are:
- **Execution Script**: `examples/run_zenoh_distributed_brain.py`
- **PFC Logic Definition**: `evospikenet/pfc.py`
- **API Definition**: `evospikenet/api.py`

Currently, there are two layers of implementation: the **simplified implementation** for the demo and the **advanced decision engine** defined as a library.

---

## 2. Implementation in Current Distributed Simulation (`run_zenoh_distributed_brain.py`)

In the currently running distributed brain simulation demo (`run_zenoh_distributed_brain.py`), the pipeline determination for Rank 0 (PFC node) is **static (hardcoded)**.

### 2.1 Input Flow

1.  **User Input**: The user sends a prompt to the API (`/api/distributed_brain/prompt`).
2.  **API Processing**: `evospikenet/api.py` receives the request and publishes a message to the Zenoh topic `evospikenet/api/prompt`.
3.  **PFC Reception**: The PFC node (Rank 0) running in `run_zenoh_distributed_brain.py` subscribes to this topic and receives it in the `_on_api_prompt` method.

### 2.2 Pipeline Determination Logic

In the `_on_api_prompt` method within the PFC node, the processing is performed as follows:

```python
# examples/run_zenoh_distributed_brain.py (Excerpt)

def _on_api_prompt(self, data: Dict):
    """Handle prompt received directly from API via Zenoh."""
    text_prompt = data.get("prompt")
    prompt_id = data.get("prompt_id")
    
    if text_prompt and prompt_id:
        # (Omitted: Duplicate check)
        
        self.logger.info(f"Received prompt via Zenoh (id: {prompt_id}): '{text_prompt}'")
        
        # Dispatch to Lang-Main
        # Statically sending to "pfc/text_prompt" topic here
        self.comm.publish("pfc/text_prompt", {"prompt": text_prompt, "prompt_id": prompt_id})
        self.active_task = True
```

**Analysis Results**:
- In the current demo code, the input text prompt is **unconditionally routed to the `Lang-Main` node (Language Processing Main Node)**.
- Dynamic decision logic (e.g., deciding whether to send to the image processing node or the language node) is not implemented in this layer.

---

## 3. Implementation of Advanced Decision Engine (`evospikenet/pfc.py`)

The core part of the library, `evospikenet/pfc.py`, implements `PFCDecisionEngine`, a dynamic pipeline determination logic based on patented technology: the **Quantum-Modulated Feedback Loop**. It is anticipated that this logic will be integrated into distributed nodes in the future.

### 3.1 Structure of PFCDecisionEngine

The `PFCDecisionEngine` class consists of the following elements:

1.  **Working Memory**: Recurrent short-term memory using `LIFNeuronLayer`.
2.  **Attention Router**: Extracts important features from input spikes using `ChronoSpikeAttention`.
3.  **Quantum Modulation Simulator**: Generates a modulation coefficient $\alpha(t)$ from cognitive entropy.

### 3.2 Pipeline Determination Algorithm

The determination of the pipeline (routing destination) is performed within the `forward` method.

#### Step 1: Calculation of Cognitive Entropy
Calculate the uncertainty (entropy) in the current decision from the routing scores (`route_scores`) for each module.

```python
# Entropy calculation (Entropy of Softmax distribution)
entropy = -torch.sum(torch.softmax(route_scores, dim=-1) * torch.log_softmax(route_scores, dim=-1), dim=-1).mean()
```

#### Step 2: Generation of Quantum Modulation Coefficient $\alpha(t)$
Map the entropy to the rotation angle $\theta$ of a qubit, and let the probability of observing state $|0\rangle$ be $\alpha(t)$.

$$ \theta = \pi \times \frac{\text{Entropy}}{\text{MaxEntropy}} $$
$$ \alpha(t) = P(|0\rangle) = \cos^2(\frac{\theta}{2}) $$

#### Step 3: Control of Routing Temperature
Dynamically control the "Temperature" of routing using $\alpha(t)$.

- **High Entropy (Uncertain)** $\rightarrow$ Low $\alpha(t)$ $\rightarrow$ **High Temperature** $\rightarrow$ **Exploratory Routing**
- **Low Entropy (Certain)** $\rightarrow$ High $\alpha(t)$ $\rightarrow$ **Low Temperature** $\rightarrow$ **Exploitative Routing**

```python
routing_temp = 1.0 / (alpha_t + 1e-9)
route_probs = torch.softmax(route_scores / routing_temp, dim=-1)
```

#### Step 4: Modulation of Self-Dynamics
$\alpha(t)$ modulates not only routing but also the firing threshold of the PFC's own neurons (control of plasticity).

```python
# Lower alpha (exploratory) -> lower threshold -> easier to fire (hyperplasticity)
modulation_factor = 0.5 + alpha_t
self.working_memory.threshold = (self.base_lif_threshold * modulation_factor).to(torch.int16)
```

### 3.3 Conclusion

The pipeline determination in `evospikenet/pfc.py` is not a simple conditional branch but a **dynamic probabilistic routing incorporating quantum mechanical modulation based on cognitive state (entropy)**.

---

## 4. Summary

| Item                     | Current Demo (`run_zenoh_distributed_brain.py`)                       | Core Library (`evospikenet/pfc.py`)                                   |
| :----------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Determination Method** | ✅ **Dynamic Probabilistic Routing (Implemented 2025-12-05)**          | **Dynamic Probabilistic Routing**                                     |
| **Logic**                | Temperature control via Quantum Modulated Feedback Loop               | Temperature control via Quantum Modulated Feedback Loop               |
| **Parameters**           | Cognitive Entropy, Quantum Modulation Coefficient $\alpha(t)$         | Cognitive Entropy, Quantum Modulation Coefficient $\alpha(t)$         |
| **Purpose**              | Autonomous decision making and balancing exploration vs. exploitation | Autonomous decision making and balancing exploration vs. exploitation |

**✅ Implementation Completed (December 5, 2025)**

The pipeline determination logic `PFCDecisionEngine` for Rank 0 (PFC) has been integrated into `ZenohBrainNode` in `run_zenoh_distributed_brain.py`, enabling flexible task distribution according to the situation (e.g., attention to the visual cortex, commands to the motor cortex, queries to the language cortex).

### Implementation Details

**Integration Overview:**
1. In `ZenohBrainNode.__init__()`, `PFCDecisionEngine` is instantiated when `module_type == "pfc"`
2. Module mapping: Defines 4 downstream modules: `["visual", "audio", "lang-main", "motor"]`
3. Dynamic routing implemented in `_on_api_prompt()` method:
   - Convert prompt to tensor
   - Call `PFCDecisionEngine.forward()` to obtain `route_probs` and `entropy`
   - Sample target module based on probability distribution
   - Publish message to Zenoh topic corresponding to selected module
4. Logging: Records entropy, probability distribution, and selected module with `[Q-PFC ROUTING]` tag

**Technical Features:**
- **Quantum Modulation Coefficient $\alpha(t)$**: Dynamically calculated from entropy to control routing temperature
- **Exploration-Exploitation Balance**: Exploratory (high temperature) during high entropy, exploitative (low temperature) during low entropy
- **Fallback Mechanism**: Automatically falls back to legacy static routing if PFC engine initialization fails
- **Backward Compatibility**: Maintains compatibility with existing `Lang-Main` nodes
