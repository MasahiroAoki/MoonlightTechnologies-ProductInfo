# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# Investigation Report on the Feasibility of 3B and 7B EvoSpikeNetLM Models

## 1. Conclusion

Scaling the EvoSpikeNetLM architecture to 3B (3 billion) and 7B (7 billion) parameter sizes is **technically feasible**.

This investigation has identified the key hyperparameters (`num_blocks`, `d_model`) that determine the total number of model parameters and has derived specific configurations to achieve the target model sizes.

## 2. Parameter Calculation Formula

The total number of parameters in the model can be estimated by the following formula:

- `L`: `num_transformer_blocks` (number of layers)
- `d`: `d_model` (model dimension)
- `V`: `vocab_size` (vocabulary size, 30,522 for `bert-base-uncased`)

**Total Parameters ≈ `2 * V * d` + `L * 12 * d^2`**

This formula is composed of the following elements:
- **Embedding & Output Layers:** `2 * V * d`
- **Transformer Blocks:** `L * 12 * d^2`
  - Each block contains `ChronoSpikeAttention` (`4 * d^2`) and `SpikingFFN` (`8 * d^2`).

## 3. Recommended Model Configurations

Based on the calculation formula above, the following hyperparameter configurations are proposed to achieve the target scales.

### 3.1. Candidates for a 3B Parameter Model

| Option | `num_blocks` (L) | `d_model` (d) | Estimated Parameters | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Balanced** | 32 | 2560 | **~2.7B** | Similar to GPT-3 Small |
| **2. Wider** | 32 | 2816 | **~3.2B** | |
| **3. Deeper** | 40 | 2560 | **~3.3B** | |

### 3.2. Candidates for a 7B Parameter Model

| Option | `num_blocks` (L) | `d_model` (d) | Estimated Parameters | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Llama-2 7B Style** | 32 | 4096 | **~6.7B** | Similar to Llama-2 7B |
| **2. Larger Variant** | 40 | 4096 | **~8.3B** | |

## 4. Challenges and Considerations for Realization

While building these large-scale models is possible, the following points must be considered for success:

- **Computational Resources:**
  - **GPU:** A 3B model would require at least multiple A100 80GB GPUs, while a 7B model would require an even larger GPU cluster.
  - **Data:** A high-quality, large-scale text corpus is essential.
  - **Time:** Training could take several weeks to months.

- **Training Stability:**
  - Training large models is inherently unstable. Techniques such as learning rate warm-up and scheduling, gradient clipping, and proper weight initialization (e.g., `xavier`, `kaiming`) will be mandatory.

- **SNN-Specific Challenges:**
  - **Surrogate Gradients:** Careful validation is needed to determine if the surrogate gradients used in `snntorch` will function stably in such a large-scale network. The risk of vanishing or exploding gradients may increase.
  - **Custom Modules:** The impact of custom control modules like `MetaSTDP` and `AEG` on the large-scale training process (whether they stabilize or destabilize it) is unknown. An approach of scaling up from smaller experiments is recommended.

## 5. Summary

By design, EvoSpikeNetLM has an architecture that allows for scaling up to large language models. By considering the model configurations and challenges presented in this report, it is possible to begin the development of 3B and 7B models.
