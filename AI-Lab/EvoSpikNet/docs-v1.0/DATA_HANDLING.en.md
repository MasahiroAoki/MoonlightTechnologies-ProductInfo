# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet: Data Handling Guide

**Last Updated:** 2025-12-15

This document details how to create, format, and validate the various types of AI data used in the `EvoSpikeNet` framework, including spike data, text corpora, RAG knowledge bases, and multi-modal datasets.

## Purpose and How to Use This Document
- Purpose: Provide a quick view of creation/format/validation steps for spike/text/RAG/multimodal data.
- Audience: Data pipeline owners, research/training operators.
- Read order: Data upload → Spike data → Text corpus → Knowledge base → Multimodal datasets.
- Related links: Distributed brain script examples/run_zenoh_distributed_brain.py; PFC/Zenoh/Executive details implementation/PFC_ZENOH_EXECUTIVE.md.

---

## 0. Data Upload Feature

EvoSpikeNet provides functionality to upload training data to an API server, enabling sharing in distributed environments. This allows LLM training to be performed not only with local files but also with uploaded datasets.

### Supported Data Formats

- **Multi-modal data**: Image and caption pairs (`captions.csv` + `images/` directory)
- **Future extensions**: Planned support for audio data, text corpora, etc.

### Upload Procedure

1. **Data Preparation**: Create training data locally
2. **Script Execution**: Start training using the `--upload-data` flag
3. **API Upload**: Data is automatically ZIP compressed and uploaded to the API server
4. **Shared Usage**: Uploaded data can be reused by other users or systems

### Usage Example

```bash
# Training with data upload
python examples/train_multi_modal_lm.py \
  --mode train \
  --dataset custom \
  --data-dir your_data_dir \
  --run-name your_run \
  --upload-data \
  --data-name your_dataset_name \
  --epochs 10 \
  --batch-size 4
```

### When API Server is Unavailable

If the API server is unavailable, training automatically falls back to using local data. Data upload is skipped and training continues with local files.

---

## 1. Spike Data Generation and Format

Spike data, which serves as direct input to SNN models, is represented as a `torch.Tensor`.

- **Format**: `torch.Tensor`
- **Shape**: A 2D tensor of shape `(time_steps, num_input_neurons)`.
- **dtype**: `torch.int8`
- **Values**: `0` (no spike) or `1` (spike).

Artificial spike data for testing can be generated on the "SNN Models" page, accessible from the "Data Generation" menu in the UI.

---

## 2. Text Corpora

For training language models (`EvoSpikeNetLM`, `SpikingEvoSpikeNetLM`), various data sources supported by the `evospikenet/dataloaders.py` module are available.

- **`WikipediaLoader`**: Dynamically loads articles from Wikipedia.
- **`AozoraBunkoLoader`**: Extracts text from HTML of Aozora Bunko (a Japanese digital library).
- **`LocalFileLoader`**: Reads local text files.

These loaders are utilized in training scripts such as `examples/train_spiking_evospikenet_lm.py`.

---

## 3. RAG Knowledge Base Management

The Retrieval-Augmented Generation (RAG) feature stores external knowledge in Milvus and Elasticsearch. Data management is primarily handled through the UI.

- **Data Structure**: Each document consists of `id` (unique), `embedding` (vector), `text` (content), and `source` fields.
- **CRUD Operations via UI**: The "Data Management" tab on the "RAG System" page provides a powerful interface for directly managing the knowledge base.
    - **Create**: Add a new row using the `add row` button and input the `text` and `source`. The `embedding` is automatically generated and saved.
    - **Read**: All data in Milvus is displayed in the table.
    - **Update**: Editing a cell in the table updates the database in real-time.
    - **Delete**: Select a row and click the `delete row` button to remove it.

---

## 4. Multi-modal Datasets

`MultiModalEvoSpikeNetLM` is trained on pairs of images and captions.

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

For interactive analysis in the UI or detailed offline visualization, the framework saves neuron activity data in `.pt` format.

- **Data Structure**: All files are saved as dictionaries with keys such as `spikes` and `membrane_potential`.
- **Generation Locations**:
    - **RAG Chat**: Neuron data can be saved when the SNN backend is selected.
    - **Spiking LM Chat**: Neuron data can be saved during text generation.
    - **SNN Models**: Generated when running a 4-layer SNN simulation (e.g., `4_layer_snn_data_lif.pt`).
- **Usage**: The generated `.pt` files can be uploaded to the "Generic Visualization" page under the "Data Analysis" menu for re-visualization or analyzed offline with `examples/visualize_*.py` scripts.

---

## 6. Synthetic Data Generation (`Data Distillation`)

The `evospikenet/distillation.py` module provides functionality to generate high-quality synthetic data using an LLM (e.g., OpenAI). This is useful for efficiently creating datasets for specific tasks like sentiment analysis or QA pair generation.

This can be run from the "Distill Data" command on the "System Utilities" page under the "System Settings" menu, by specifying the task type, number of samples, and a prompt.

---

## 7. Audio Data

The multi-modal model supports audio input.

- **Format**: Standard audio file formats supported by `torchaudio`, such as `.wav`, `.mp3`, and `.flac`.
- **UI Usage**: Audio files can be uploaded from the "Brain Simulation" tab on the "Distributed Brain" page and used as input for the simulation along with text prompts and images.
- **Data Processing**: On the backend, uploaded files are converted into waveforms and sample rates using `torchaudio.load` and preprocessed into a format that the model's `SpikingAudioEncoder` can handle.

---

## 8. Federated Learning Datasets

In federated learning, each client maintains an independent local dataset.

- **Format**: CSV (`.csv`) file.
- **Data Structure**: The current implementation assumes a text classification task, where each row must consist of `text` and `label` columns.
- **Usage**: When running the `examples/run_fl_client.py` script, specify the path to the local CSV file using the `--data-path` argument.

---

## 9. Distributed Brain Simulation Data Flow

The distributed brain simulation exchanges data between the UI, API, and simulation processes.

- **Input Data (UI → API → Simulation)**:
    1. The user inputs a text prompt and uploads image or audio files in the "Distributed Brain" UI.
    2. Clicking the "Execute Query" button sends the media files (Base64 encoded) and text to the `/api/distributed_brain/prompt` API endpoint.
    3. The API writes the received prompt data and media files as a JSON file and associated files with a unique ID to the server's `/tmp` directory.
    4. The simulation process (specifically Rank 0, the PFC) periodically scans (polls) this `/tmp` directory to detect and process new prompt files.

- **Output Data (Simulation → API → UI)**:
    - **Status**: The Rank 0 process periodically POSTs the current state of the simulation (status of each node, edge activity, PFC entropy, etc.) to the `/api/distributed_brain/status` API endpoint. The UI polls this endpoint to update its display in real-time.
    - **Result**: When the simulation completes a task, it writes the final text result to a result file in the `/tmp` directory. The UI polls the `/api/distributed_brain/result` API endpoint, which reads the corresponding result file, returns its content to the UI, and deletes the file.
    - **Logs**: Each simulation process (Rank 0, 1, ...) writes its own logs to a file at `/tmp/sim_rank_{rank}.log`. The UI reads and displays the logs for the selected node (via the API if necessary).
    - **Artifacts**: During the simulation, each process can upload its internal state (like spikes and membrane potential) as tensors in `.pt` files to the database as artifacts. This is done via the `EvoSpikeNetAPIClient`'s `upload_artifact` method.
