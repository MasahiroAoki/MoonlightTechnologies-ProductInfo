# Distributed Brain Simulation Update: Custom Model Support

## Overview
The distributed brain simulation has been updated to support custom model assignment for all major functional nodes, not just the Language module. This allows for a more flexible and realistic simulation where specialized models (LLMs, VLMs, etc.) can be used for different cognitive tasks.

## Changes

### Frontend (`distributed_brain.py`)
- **Expanded Model Selection**: The `MODEL_CAPABLE_NODES` list now includes:
  - `lang-main`, `lang-parent` (Language)
  - `visual`, `vis-parent` (Visual)
  - `auditory`, `aud-parent` (Auditory)
  - `speech` (Speech Synthesis)
  - `motor` (Motor Control)
  - `compute` (General Compute)
- **UI Update**: The "Language Model" column in the node configuration table has been renamed to "Model" to reflect the broader support.

### Backend (`run_distributed_brain_simulation.py`)
- **Generic Model Loading**: Implemented a `load_custom_model` function that can load PyTorch models (`model.pt`) and configurations (`config.json`) from a specified directory.
- **Module Updates**: Updated the initialization logic for the following modules to use the new loading mechanism:
  - **Visual Module**: Supports custom vision models.
  - **Auditory Module**: Supports custom audio processing models.
  - **Speech Module**: Supports custom speech synthesis models.
  - **Motor Module**: Supports custom motor control models.
  - **Compute Module**: Supports custom compute models.

## How to Use
1.  **Upload Model**: Upload your custom model artifact (zip file containing `model.pt` and optional `config.json`) via the API or UI (if available).
2.  **Configure Nodes**: In the "Node Configuration" tab of the Distributed Brain page:
    - Select the target node (e.g., `Node 1 - Visual`).
    - In the "Model" dropdown, select your uploaded model artifact.
3.  **Start Simulation**: Start the simulation. The selected nodes will automatically download and load their assigned models.
