# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet SDK Quickstart Guide

**Last Updated:** December 15, 2025
**Get started with the EvoSpikeNet SDK in 30 seconds.**

## Purpose and How to Use This Document
- Purpose: Provide the fastest path to set up the SDK and run the API client.
- Audience: Developers starting SDK usage.
- Read order: Installation → API server start → Minimal example.
- Related links: Distributed brain script examples/run_zenoh_distributed_brain.py (for runtime context); PFC/Zenoh/Executive details implementation/PFC_ZENOH_EXECUTIVE.md.

## Installation

```bash
pip install -e .
```

## Start the API Server

```bash
sudo ./scripts/run_api_server.sh
```

## Minimal Example

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

# Initialize client
client = EvoSpikeNetAPIClient()

# Check server status
if client.wait_for_server():
    # Generate text
    result = client.generate("What is artificial intelligence?")
    print(result['generated_text'])
```

---

## Common Patterns

### 1️⃣ Simple Text Generation

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()
result = client.generate("List five applications of machine learning.")
print(result['generated_text'])
```

### 2️⃣ Processing Multiple Prompts

```python
prompts = ["What is AI?", "Explain machine learning", "Deep learning basics"]
results = client.batch_generate(prompts, max_length=100)

for prompt, result in zip(prompts, results):
    print(f"{prompt}: {result.get('generated_text', 'Failed')}")
```

### 3️⃣ Multi-modal Processing with Images

```python
response = client.submit_prompt(
    prompt="What is shown in this image?",
    image_path="./image.jpg"
)
result = client.poll_for_result(timeout=60)
print(result['response'])
```

### 4️⃣ Execution with Error Handling

```python
# Validate prompt
if client.validate_prompt("Test prompt"):
    # Execute with automatic retries
    result = client.with_error_handling(
        client.generate,
        prompt="Test prompt",
        max_length=100,
        retries=3
    )
    if result:
        print("Success:", result['generated_text'])
```

### 5️⃣ Monitoring Asynchronous Tasks

```python
# Submit task
client.submit_prompt(prompt="A complex task")

# Poll for result
result = client.poll_for_result(timeout=120, interval=5)

if result:
    print("Result:", result['response'])
else:
    print("Timeout")
```

### 6️⃣ Saving and Restoring Models

```python
import torch
import io

# Create session
session = client.create_log_session("Model training experiment")
session_id = session['session_id']

# Save model
model_buffer = io.BytesIO()
torch.save(model.state_dict(), model_buffer)
model_buffer.seek(0)

# Upload
artifact = client.upload_artifact(
    session_id=session_id,
    artifact_type="model",
    name="model.pth",
    file=model_buffer
)

# Download
client.download_artifact(
    artifact_id=artifact['artifact_id'],
    destination_path="./downloaded_model.pth"
)
```

### 7️⃣ Dataset Upload

```python
import zipfile
import os

# Prepare training data directory
data_dir = "./training_data"
os.makedirs(f"{data_dir}/images", exist_ok=True)

# Compress dataset to ZIP
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add CSV file
    zf.write(f"{data_dir}/captions.csv", arcname='captions.csv')
    # Add image files
    for root, _, files in os.walk(f"{data_dir}/images"):
        for file in files:
            full_path = os.path.join(root, file)
            archive_name = os.path.join('images', os.path.relpath(full_path, f"{data_dir}/images"))
            zf.write(full_path, arcname=archive_name)

zip_buffer.seek(0)
zip_buffer.name = "training_dataset.zip"

# Upload dataset
dataset_artifact = client.upload_artifact(
    session_id=session_id,
    artifact_type="dataset",
    name="vision_training_data",
    file=zip_buffer,
    llm_type="SpikingEvoMultiModalLM"
)

print(f"Dataset upload completed: {dataset_artifact['artifact_id']}")
```

---

## Checking Server Information

```python
# Server health check
is_healthy = client.is_server_healthy()
print(f"Is the server healthy?: {'Yes' if is_healthy else 'No'}")

# Monitor status
status = client.get_simulation_status()
print(f"Current prompt status: {status.get('last_prompt_status', 'N/A')}")
print(f"Number of active nodes: {len(status.get('nodes', []))}")
```

---

## Common Errors and Solutions

| Error             | Cause                               | Solution                               |
|-------------------|-------------------------------------|----------------------------------------|
| `ConnectionError` | API server is not running           | Start it with `sudo ./scripts/run_api_server.sh` |
| `Timeout`         | Processing is slow                  | Increase the `timeout` parameter       |
| `Invalid prompt`  | Prompt does not meet criteria       | Pre-validate with `validate_prompt()`  |

---

## Next Steps

- 📖 Read the [Full SDK Documentation](./EvoSpikeNet_SDK.en.md)
- 📁 Check the [Sample Code](./docs/sdk/)
- 🔧 Refer to [Troubleshooting](./EvoSpikeNet_SDK.en.md#11-troubleshooting)
- 💬 Ask questions on GitHub Issues

---

## Cheat Sheet

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()

# Server Check
client.wait_for_server()           # Wait for startup
client.is_server_healthy()         # Health check

# Text Generation
client.generate(prompt)            # Simple generation
client.batch_generate(prompts)     # Batch processing
client.submit_prompt(prompt)       # Async submission
client.poll_for_result()           # Wait for result

# Validation & Control
client.validate_prompt(prompt)     # Validate prompt
client.with_error_handling(func)   # Execute with retries

# Status & Logs
client.get_simulation_status()     # Get status
client.get_simulation_result()     # Get result
client.get_remote_log()            # Get logs

# Artifact Management
client.create_log_session()        # Create session
client.upload_artifact()           # Upload
client.download_artifact()         # Download
client.list_artifacts()            # List
```

## 6️⃣ Running LLM Training Jobs (New Feature)

### Vision Encoder Training

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()

# Submit Vision Encoder training job
job_data = {
    "category": "Vision",
    "model_name": "google/vit-base-patch16-224",
    "dataset_path": "data/llm_training/Vision/vision_data.jsonl",
    "output_dir": "saved_models/Vision/vision-training-run",
    "gpu": True,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.00001
}

response = client.submit_training_job(job_data)
print(f"Training job started: {response['job_id']}")

# Check job status
status = client.get_training_status(response['job_id'])
print(f"Job status: {status['status']}")
```

### Audio Encoder Training

```python
# Submit Audio Encoder training job
job_data = {
    "category": "Audio",
    "model_name": "openai/whisper-base",
    "dataset_path": "data/llm_training/Audio/audio_data.jsonl",
    "output_dir": "saved_models/Audio/audio-training-run",
    "gpu": True,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.00001
}

response = client.submit_training_job(job_data)
print(f"Audio training job started: {response['job_id']}")
```

### Monitoring Training Jobs

```python
# List all training jobs
jobs = client.list_training_jobs()
for job in jobs:
    print(f"Job ID: {job['job_id']}, Status: {job['status']}, Category: {job['category']}")

# Get details of a specific job
job_details = client.get_training_job_details("vision_training_job_001")
print(f"Job details: {job_details}")
```

### Distributed Brain Node Training

```python
# Training configurations for different node types
node_configs = {
    "Vision": {
        "model_name": "google/vit-base-patch16-224",
        "node_types": ["Vision-Primary", "Vision-Secondary"]
    },
    "Audio": {
        "model_name": "openai/whisper-base",
        "node_types": ["Audio-Primary", "Audio-Secondary"]
    },
    "LangText": {
        "model_name": "microsoft/DialoGPT-medium",
        "node_types": ["Lang-Primary", "Lang-Secondary"]
    }
}

# Submit training job for Vision nodes
vision_job = {
    "category": "Vision",
    "model_name": node_configs["Vision"]["model_name"],
    "dataset_path": "data/llm_training/Vision/vision_data.jsonl",
    "output_dir": "saved_models/Vision/distributed-vision-run",
    "gpu": True,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.00002
}

client.submit_training_job(vision_job)
```

Happy coding! 🚀
