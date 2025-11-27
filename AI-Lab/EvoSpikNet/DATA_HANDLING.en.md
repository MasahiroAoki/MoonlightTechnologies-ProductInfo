# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet: Data Handling Guide

**Last Updated:** 2025-11-20

This document details how to create, format, and validate the various types of AI data used in the `EvoSpikeNet` framework, including spike data, text corpora, RAG knowledge bases, and multi-modal datasets.

---

## 1. Spike Data Generation and Format

Spike data, the direct input for SNN models, is represented as a `torch.Tensor`.

- **Format**: `torch.Tensor`
- **Shape**: A 2D tensor of `(time_steps, num_input_neurons)`.
- **dtype**: `torch.int8`
- **Values**: `0` (no spike) or `1` (spike).

Artificial spike data for testing can be generated on the "SNN Models" page under the "Data Generation" menu in the UI.

---

## 2. Text Corpora

The `evospikenet/dataloaders.py` module provides several data sources for training language models (`EvoSpikeNetLM`, `SpikingEvoSpikeNetLM`).

- **`WikipediaLoader`**: Dynamically loads articles from Wikipedia.
- **`AozoraBunkoLoader`**: Extracts text from Aozora Bunko HTML.
- **`LocalFileLoader`**: Loads text from local files.

These loaders are utilized in training scripts like `examples/train_spiking_evospikenet_lm.py`.

---

## 3. RAG Knowledge Base Management

The Retrieval-Augmented Generation (RAG) feature stores external knowledge in Milvus and Elasticsearch. Data management is primarily handled through the UI.

- **Data Structure**: Each document consists of an `id` (unique), `embedding` (vector), `text` (body), and `source` (origin).
- **CRUD Operations via UI**: The "Knowledge Base" page under the "Data Generation" menu is a powerful interface for managing the knowledge base directly.
    - **Create**: Add a row with the `add row` button and input the `text` and `source`. The `embedding` is generated and stored automatically.
    - **Read**: All data in Milvus is displayed in the table.
    - **Update**: Editing a cell in the table updates the database in real-time.
    - **Delete**: Select a row and click the `delete row` button to remove it.

---

## 4. Multi-modal Datasets

The `MultiModalEvoSpikeNetLM` is trained on pairs of images and captions.

- **Directory Structure**:
  ```
  data/multi_modal_dataset/
  ├── images/ (Contains image files)
  └── captions.csv (Maps image paths to captions)
  ```
- **`captions.csv` Format**:
  ```csv
  image_path,caption
  images/image_0.png,"Caption 1"
  images/image_1.jpg,"Caption 2"
  ```
This dataset is used for model training with the `examples/train_multi_modal_lm.py` script.

---

## 5. Visualization Data

The framework saves neuron activity data as `.pt` files for interactive analysis in the UI and detailed offline visualization.

- **Data Structure**: All files are saved as a dictionary containing keys like `spikes` and `membrane_potential`.
- **Generation Points**:
    - **RAG Chat**: Neuron data can be saved when the SNN backend is selected.
    - **Spiking LM Chat**: Neuron data can be saved during text generation.
    - **SNN Models**: Generated when running the 4-layer SNN simulation (e.g., `4_layer_snn_data_lif.pt`).
- **Usage**: The generated `.pt` files can be uploaded to the "Generic Visualization" page under the "Data Analysis" menu for re-visualization or analyzed offline with the `examples/visualize_*.py` scripts.

---

## 6. Synthetic Data Generation (`Data Distillation`)

The `evospikenet/distillation.py` module provides functionality to generate high-quality synthetic data using an LLM (e.g., OpenAI). This is useful for efficiently creating datasets for specific tasks.

You can run this from the "Distill Data" command on the "System Utilities" page under the "System Settings" menu.

---

## 7. Audio Data

The multi-modal model supports audio inputs.

- **Format**: Standard audio file formats supported by `torchaudio`, such as `.wav`, `.mp3`, and `.flac`.
- **Usage via UI**: Upload an audio file from the "Brain Simulation" tab on the "Distributed Brain" page to use it as an input for the simulation, along with text prompts or images.
- **Data Processing**: On the backend, the uploaded file is converted into a waveform and sample rate using `torchaudio.load` and preprocessed into a format that the model's `SpikingAudioEncoder` can handle.

---

## 8. Federated Learning Datasets

In federated learning, each client maintains its own independent, local dataset.

- **Format**: CSV (`.csv`) file.
- **Data Structure**: The current implementation assumes a text classification task, where each row must consist of two columns: `text` and `label`.
- **Usage**: Specify the path to the local CSV file using the `--data-path` argument when running the `examples/run_fl_client.py` script.

---

## 9. Distributed Brain Simulation Data Flow

The distributed brain simulation involves the exchange of several types of data between the UI, the API, and the simulation processes.

- **Input Data (UI → API → Simulation)**:
    1. The user enters a text prompt and uploads optional image or audio files in the "Distributed Brain" UI.
    2. Upon clicking "Execute Query," the media files are Base64-encoded and sent along with the text to the `/api/distributed_brain/prompt` API endpoint.
    3. The API temporarily holds this prompt data in an **in-memory store**.
    4. The Rank 0 (PFC) simulation process periodically polls the `/api/distributed_brain/get_prompt` API endpoint to fetch the new prompt and begin processing.

- **Output Data (Simulation → API → UI)**:
    - **Status**: The Rank 0 process periodically POSTs the current state of the simulation (status of each node, edge activity, PFC entropy, etc.) to the `/api/distributed_brain/status` endpoint. The UI polls this endpoint to update its visualizations in real-time.
    - **Result**: The Rank 0 process POSTs the final textual result of the task to the `/api/distributed_brain/result` endpoint. The UI polls this endpoint to display the response in the "Query Response" area.
    - **Logs**: Each simulation process (Rank 0, 1, ...) writes its log to a file at `/tmp/sim_rank_{rank}.log`. The UI reads these log files (via the API, if necessary) to display them.
    - **Artifacts**: During the simulation, any process can upload tensors of its internal state (e.g., spikes, membrane potentials) as `.pt` file artifacts to the database. This is done via the `EvoSpikeNetAPIClient`'s `upload_artifact` method.
