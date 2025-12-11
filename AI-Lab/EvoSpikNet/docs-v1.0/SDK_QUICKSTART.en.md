# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet SDK Quickstart Guide

**Get started with the EvoSpikeNet SDK in 30 seconds.**

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
- 📁 Check the [Sample Code](./examples/sdk/)
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

Happy coding! 🚀
