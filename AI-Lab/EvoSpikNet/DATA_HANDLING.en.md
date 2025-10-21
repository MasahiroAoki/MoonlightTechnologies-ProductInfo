# EvoSpikeNet: Data Handling Guide

This document details how to create, format, and validate the various types of AI data used in the `EvoSpikeNet` framework, including spike data, text corpora, RAG knowledge bases, and multi-modal datasets.

---

## 1. Spike Data Generation and Format

This section covers the specifications and generation methods for spike data, which serves as the direct input to SNN models.

### 1.1. Data Specifications
- **Format**: `torch.Tensor`
- **Shape**: A 2D tensor of `(time_steps, num_input_neurons)`. For batch processing, a 3D tensor of `(batch_size, time_steps, num_input_neurons)`.
- **dtype**: `torch.int8`
- **Values**: `0` (no spike) or `1` (spike).

### 1.2. Data Generation Script
Artificial spike data for testing can be generated with `scripts/generate_spike_data.py`. It creates spike data with a specified firing rate based on a Poisson distribution.

```bash
python scripts/generate_spike_data.py --num-samples 100 --time-steps 200
```

---

## 2. Data Sources for Text Corpora

Training the language models (`EvoSpikeNetLM`, `SpikingEvoSpikeNetLM`) requires a text-based corpus. The `evospikenet/dataloaders.py` module supports loading data from various sources.

- **`WikipediaLoader`**: Loads articles from Wikipedia.
- **`AozoraBunkoLoader`**: Extracts text from Aozora Bunko HTML pages.
- **`LocalFileLoader`**: Reads text from local files.

These loaders can be utilized by scripts such as `examples/train_evospikenet_lm.py`.

---

## 3. Data Handling for RAG Knowledge Base

The Retrieval-Augmented Generation (RAG) feature utilizes a Milvus vector database to store external knowledge. Data within the knowledge base is managed directly in Milvus via the `evospikenet/rag_milvus.py` module.

### 3.1. Data Structure
Each document stored in Milvus consists of the following fields:
- `id`: A unique identifier (auto-incremented).
- `embedding`: A 384-dimensional vector generated from the text.
- `text`: The body of the document.
- `source`: A string indicating the document's origin (e.g., "wikipedia", "user_input").

### 3.2. Data Management (via UI)
The "Data CURD" tab on the "RAG System" page of the dashboard provides a comprehensive interface for managing the knowledge base.
- **Read:** All documents stored in Milvus are displayed in a table.
- **Create:** New documents can be added by clicking the "Add Row" button and entering text and a source.
- **Update:** Existing documents can be modified by directly editing the cells in the table.
- **Delete:** Documents can be removed by selecting a row and clicking the "Delete Selected Row" button.

This interface allows for direct interaction with the database, eliminating the need for intermediate files like `knowledge_base.json`.

---

## 4. Format for Multi-modal Datasets

The multi-modal model (`MultiModalEvoSpikeNetLM`) is trained on pairs of images and their corresponding captions. The framework expects the following data structure.

### 4.1. Directory Structure
```
data/
└── multi_modal_dataset/
    ├── images/
    │   ├── image_0.png
    │   ├── image_1.jpg
    │   └── ...
    └── annotations.json
```

### 4.2. `annotations.json` Format
This JSON file is a list of objects, each describing the path to an image file and its corresponding caption.

- **Format**:
  ```json
  [
    {
      "image_path": "images/image_0.png",
      "caption": "A lighthouse stands along the coastline at dusk."
    },
    {
      "image_path": "images/image_1.jpg",
      "caption": "A tranquil lake against a backdrop of snow-capped mountains."
    }
  ]
  ```
The `examples/train_multi_modal_lm.py` script reads this `annotations.json` file to provide the model with image-caption pairs for training.