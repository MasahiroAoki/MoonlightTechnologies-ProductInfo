# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki

# EvoSpikeNet Python SDK Documentation

**Last Updated:** 2025-11-24

## 1. Overview

The `EvoSpikeNet Python SDK` is a client library that provides a high-level interface for interacting with the `EvoSpikeNet API`. Using this SDK, developers can easily integrate EvoSpikeNet's text generation, data logging, and distributed brain simulation capabilities into their applications with just a few lines of Python code, without worrying about HTTP request details.

---

## 2. Setup and Installation

### 2.1. Prerequisites
- Python 3.8 or later
- `requests` library
- Running EvoSpikeNet API server

### 2.2. Installation Steps
This SDK is provided as part of the `evospikenet` package. Install the project in editable mode by running the following command in the project root directory:

```bash
pip install -e .
```

### 2.3. Starting the API Server
Before using the SDK, the API server must be running:

```bash
# Using Docker Compose (recommended)
sudo ./scripts/run_api_server.sh

# Or start all services (including UI)
sudo ./scripts/run_frontend_cpu.sh
```

---

## 3. `EvoSpikeNetAPIClient` Class

The central class that manages all communication with the API.

### 3.1. Initialization

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

# If the API server is running at the default URL (http://localhost:8000)
client = EvoSpikeNetAPIClient()

# Connecting from within Docker environment
client = EvoSpikeNetAPIClient(base_url="http://api:8000")

# Custom URL
client = EvoSpikeNetAPIClient(base_url="http://your-api-server:8000")
```

### 3.2. Health Check

#### `is_server_healthy() -> bool`
Checks if the API server is running and healthy.

**Example:**
```python
client = EvoSpikeNetAPIClient()

if client.is_server_healthy():
    print("✅ API server is running normally")
else:
    print("❌ Cannot connect to API server")
```

---

## 4. Text Generation

### 4.1. Basic Text Generation

#### `generate(prompt: str, max_length: int = 50) -> Dict[str, str]`
Calls the standard text generation endpoint.

**Parameters:**
- `prompt` (str): Text prompt
- `max_length` (int): Maximum number of tokens to generate (default: 50)

**Returns:** 
Dictionary containing the generated text

**Example:**
```python
client = EvoSpikeNetAPIClient()

# Simple text generation
result = client.generate("Artificial Intelligence is", max_length=100)
print(f"Generated text: {result.get('generated_text', '')}")
```

---

## 5. Distributed Brain Simulation

Provides comprehensive functionality for interacting with distributed brain simulations.

### 5.1. Submitting Multimodal Prompts

#### `submit_prompt(prompt: str = None, image_path: str = None, audio_path: str = None) -> Dict`
Submits a multimodal prompt to the simulation.

**Parameters:**
- `prompt` (str, optional): Text prompt
- `image_path` (str, optional): Path to an image file
- `audio_path` (str, optional): Path to an audio file

**Note:** At least one modality must be provided.

**Example 1: Text Only**
```python
client = EvoSpikeNetAPIClient()

# Submit text prompt
response = client.submit_prompt(prompt="What is the capital of Japan?")
print(f"Prompt submission result: {response}")
```

**Example 2: Text + Image**
```python
# Submit combined text and image
response = client.submit_prompt(
    prompt="What do you see in this image?",
    image_path="./examples/sample_image.jpg"
)
print(f"Multimodal prompt submitted: {response}")
```

**Example 3: All Modalities**
```python
# Submit text, image, and audio together
response = client.submit_prompt(
    prompt="Analyze the audio and image",
    image_path="./data/image.png",
    audio_path="./data/audio.wav"
)
```

### 5.2. Monitoring Simulation Status

#### `get_simulation_status() -> Dict`
Retrieves the current status of the simulation.

**Returns:**
Dictionary containing status information for all nodes
- `nodes`: Information about each node (ID, label, status, etc.)
- `last_prompt_status`: Processing status of the last prompt
- Other metadata

**Example:**
```python
status = client.get_simulation_status()

print(f"Prompt status: {status.get('last_prompt_status', 'N/A')}")
print(f"Active nodes: {len(status.get('nodes', []))}")

# Display details for each node
for node in status.get('nodes', []):
    print(f"  - {node.get('label')}: {node.get('status', 'unknown')}")
```

### 5.3. Retrieving Simulation Results

#### `get_simulation_result() -> Dict`
Retrieves the latest result from a completed query.

**Returns:**
```python
{
    "response": "Generated text response",
    "timestamp": 1234567890.123
}
```

Returns `{"response": None}` if no result is available.

**Example:**
```python
result = client.get_simulation_result()

if result.get("response"):
    print(f"Simulation response: {result['response']}")
    print(f"Timestamp: {result.get('timestamp', '')}")
else:
    print("No result available yet")
```

### 5.4. Polling for Results

#### `poll_for_result(timeout: int = 120, interval: int = 5) -> Optional[Dict]`
Periodically polls the result endpoint until a result is available.

**Parameters:**
- `timeout` (int): Maximum time to wait in seconds (default: 120)
- `interval` (int): Polling interval in seconds (default: 5)

**Returns:**
Result content if found, `None` if timeout occurs

**Example:**
```python
client = EvoSpikeNetAPIClient()

# Submit prompt
client.submit_prompt(prompt="Tell me about the future of AI")

# Wait for result (max 2 minutes)
print("Waiting for result...")
result = client.poll_for_result(timeout=120, interval=5)

if result:
    print(f"✅ Response: {result['response']}")
else:
    print("❌ Timeout: Could not retrieve result")
```

---

## 6. Data Logging and Artifact Management

Provides robust functionality for experiment reproducibility and data management.

### 6.1. Creating Sessions

#### `create_log_session(description: str) -> Dict`
Starts a new experiment session and obtains a unique session ID.

**Parameters:**
- `description` (str): Description of the session's purpose or content

**Returns:**
Dictionary containing session information (including `session_id`)

**Example:**
```python
session = client.create_log_session(
    description="SNN model hyperparameter tuning experiment"
)
session_id = session['session_id']
print(f"✅ Session ID: {session_id}")
```

### 6.2. Uploading Artifacts

#### `upload_artifact(session_id: str, artifact_type: str, name: str, file: io.BytesIO) -> Dict`
Uploads data artifacts (models, datasets, config files) associated with a specific session.

**Parameters:**
- `session_id` (str): Session ID to associate the artifact with
- `artifact_type` (str): Type of artifact (e.g., `model`, `config`, `simulation_data`)
- `name` (str): Artifact filename
- `file` (io.BytesIO): File object to upload

**Example:**
```python
import io
import torch

# Save model
model_buffer = io.BytesIO()
torch.save(model.state_dict(), model_buffer)
model_buffer.seek(0)
model_buffer.name = 'model.pth'

# Upload
result = client.upload_artifact(
    session_id=session_id,
    artifact_type="model",
    name="spiking_lm_v1.pth",
    file=model_buffer
)
print(f"✅ Artifact ID: {result['artifact_id']}")
```

### 6.3. Listing Artifacts

#### `list_artifacts(artifact_type: str = None) -> List[Dict]`
Retrieves a list of all artifacts stored in the database.

**Parameters:**
- `artifact_type` (str, optional): Filter artifacts by type

**Example:**
```python
# Get all artifacts
all_artifacts = client.list_artifacts()

# Get only models
models = client.list_artifacts(artifact_type="model")

for artifact in models:
    print(f"ID: {artifact['artifact_id']}")
    print(f"Name: {artifact['name']}")
    print(f"Created: {artifact['created_at']}")
    print("---")
```

### 6.4. Downloading Artifacts

#### `download_artifact(artifact_id: str, destination_path: str)`
Downloads a file with the specified artifact ID.

**Parameters:**
- `artifact_id` (str): Unique ID of the artifact to download
- `destination_path` (str): Local path to save the file

**Example:**
```python
# Download latest model
models = client.list_artifacts(artifact_type="model")
if models:
    latest_model = models[0]
    client.download_artifact(
        artifact_id=latest_model['artifact_id'],
        destination_path="./downloaded_model.pth"
    )
    print("✅ Model downloaded")
```

---

## 7. Retrieving Remote Logs

For multi-PC distributed simulations, retrieves log files from remote machines.

#### `get_remote_log(user: str, ip: str, key_path: str, log_file_path: str) -> Dict`

**Parameters:**
- `user` (str): SSH username
- `ip` (str): IP address of remote host
- `key_path` (str): Local path to SSH private key
- `log_file_path` (str): Absolute path to log file on remote host

**Example:**
```python
log_content = client.get_remote_log(
    user="ubuntu",
    ip="192.168.1.100",
    key_path="/home/user/.ssh/id_rsa",
    log_file_path="/home/appuser/app/simulation_rank1.log"
)
print(f"Log content:\n{log_content.get('log_content', '')}")
```

---

## 8. Comprehensive Usage Examples

### 8.1. Executing Text Queries

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
import time

def simple_text_query():
    """Simple text query example"""
    client = EvoSpikeNetAPIClient()
    
    # 1. Health check
    if not client.is_server_healthy():
        print("❌ API server not responding")
        return
    
    # 2. Submit prompt
    prompt = "Tell me about the future of artificial intelligence"
    print(f"📤 Submitting prompt: {prompt}")
    client.submit_prompt(prompt=prompt)
    
    # 3. Wait for result
    print("⏳ Waiting for processing...")
    result = client.poll_for_result(timeout=60, interval=3)
    
    # 4. Display result
    if result and result.get('response'):
        print(f"\n✅ Response:\n{result['response']}\n")
    else:
        print("❌ Could not retrieve response")

if __name__ == "__main__":
    simple_text_query()
```

### 8.2. Executing Multimodal Queries

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

def multimodal_query():
    """Multimodal query example with image"""
    client = EvoSpikeNetAPIClient()
    
    # Combined image and text query
    response = client.submit_prompt(
        prompt="Describe the objects in this image",
        image_path="./data/sample_image.png"
    )
    
    print("Prompt submission complete")
    
    # Wait for result
    result = client.poll_for_result(timeout=120)
    
    if result:
        print(f"Visual processing result: {result['response']}")

if __name__ == "__main__":
    multimodal_query()
```

### 8.3. Complete ML Workflow with Artifact Management

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
from evospikenet.models import SpikingEvoSpikeNetLM
import torch
import json
import io

def complete_ml_workflow():
    """Complete workflow from model training to artifact storage"""
    
    client = EvoSpikeNetAPIClient()
    
    # --- Step 1: Create session ---
    print("\n=== Step 1: Create new session ===")
    session = client.create_log_session(
        description="Training experiment for SpikingEvoSpikeNetLM"
    )
    session_id = session['session_id']
    print(f"✅ Session ID: {session_id}")
    
    # --- Step 2: Model training (dummy) ---
    print("\n=== Step 2: Model training ===")
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 4,
        'num_transformer_blocks': 2,
        'time_steps': 10
    }
    
    model = SpikingEvoSpikeNetLM(**config)
    print("✅ Model initialized")
    
    # Training would happen here
    # model.train()...
    
    # --- Step 3: Save model and config ---
    print("\n=== Step 3: Upload artifacts ===")
    
    # Save model
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    model_buffer.seek(0)
    model_buffer.name = 'model.pth'
    
    model_artifact = client.upload_artifact(
        session_id=session_id,
        artifact_type="model",
        name="spiking_lm.pth",
        file=model_buffer
    )
    print(f"✅ Model uploaded: {model_artifact['artifact_id']}")
    
    # Save config
    config_buffer = io.BytesIO()
    config_buffer.write(json.dumps(config).encode('utf-8'))
    config_buffer.seek(0)
    config_buffer.name = 'config.json'
    
    config_artifact = client.upload_artifact(
        session_id=session_id,
        artifact_type="config",
        name="config.json",
        file=config_buffer
    )
    print(f"✅ Config uploaded: {config_artifact['artifact_id']}")
    
    # --- Step 4: Verify artifacts ---
    print("\n=== Step 4: Verify artifact list ===")
    artifacts = client.list_artifacts()
    print(f"Total stored artifacts: {len(artifacts)}")
    for artifact in artifacts[-5:]:  # Show last 5
        print(f"  - {artifact['name']} ({artifact['artifact_type']})")
    
    # --- Step 5: Download and restore model ---
    print("\n=== Step 5: Download and restore model ===")
    models = client.list_artifacts(artifact_type="model")
    if models:
        latest_model_id = models[0]['artifact_id']
        client.download_artifact(
            artifact_id=latest_model_id,
            destination_path="./restored_model.pth"
        )
        
        # Restore model
        restored_model = SpikingEvoSpikeNetLM(**config)
        restored_model.load_state_dict(torch.load("./restored_model.pth"))
        print("✅ Model restored")
        
        # Cleanup
        import os
        os.remove("./restored_model.pth")
    
    print("\n" + "="*50)
    print("Workflow complete!")

if __name__ == "__main__":
    complete_ml_workflow()
```

### 8.4. Real-time State Monitoring

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
import time

def monitor_simulation():
    """Monitor simulation status in real-time"""
    
    client = EvoSpikeNetAPIClient()
    
    # Submit prompt
    client.submit_prompt(prompt="Complex computational task")
    
    print("Starting simulation monitoring...")
    print("="*60)
    
    # Check status every 3 seconds for max 60 seconds
    max_time = 60
    interval = 3
    elapsed = 0
    
    while elapsed < max_time:
        status = client.get_simulation_status()
        prompt_status = status.get('last_prompt_status', 'Unknown')
        
        print(f"[{elapsed}s] Status: {prompt_status}")
        
        # Display node information
        for node in status.get('nodes', [])[:5]:  # First 5 nodes only
            label = node.get('label', 'N/A')
            node_status = node.get('status', 'N/A')
            print(f"  → {label}: {node_status}")
        
        # Check completion
        if 'completed' in prompt_status.lower() or 'idle' in prompt_status.lower():
            print("\n✅ Simulation complete!")
            
            # Get result
            result = client.get_simulation_result()
            if result and result.get('response'):
                print(f"\nResponse: {result['response']}")
            break
        
        time.sleep(interval)
        elapsed += interval
        print("-"*60)
    
    if elapsed >= max_time:
        print("\n⚠️ Timeout")

if __name__ == "__main__":
    monitor_simulation()
```

---

## 9. Error Handling and Best Practices

### 9.1. Basic Error Handling

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_api_call():
    """Robust API call with error handling"""
    
    client = EvoSpikeNetAPIClient()
    
    try:
        # API server health check
        if not client.is_server_healthy():
            raise ConnectionError("API server not responding")
        
        # Submit prompt
        response = client.submit_prompt(prompt="Test query")
        
        # Wait for result (with timeout)
        result = client.poll_for_result(timeout=30)
        
        if not result:
            print("⚠️ Could not retrieve result (timeout)")
            return None
        
        return result
        
    except ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("Please verify API server is running")
        
    except Timeout:
        print("❌ Timeout: API server response too slow")
        
    except RequestException as e:
        print(f"❌ API request error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return None

if __name__ == "__main__":
    result = robust_api_call()
```

### 9.2. Best Practices

1. **Always perform health checks**: Verify server status with `is_server_healthy()` before API calls

2. **Set appropriate timeouts**: Adjust timeout in `poll_for_result()` based on processing complexity

3. **Error handling**: Wrap all API calls in `try-except` blocks

4. **Resource cleanup**: Properly delete files after use

5. **Session management**: Group related experiments under the same session ID

6. **Logging**: Properly log important operations and errors

---

## 10. Troubleshooting

### Common Issues and Solutions

**Issue 1: `is_server_healthy()` returns `False`**
```
Solutions:
1. Check if API server is running: docker ps | grep api
2. Verify correct URL is specified
3. Check firewall and network settings
```

**Issue 2: `poll_for_result()` times out**
```
Solutions:
1. Increase timeout duration
2. Check simulation logs for errors
3. Verify distributed brain simulation is started correctly
```

**Issue 3: Artifact upload fails**
```
Solutions:
1. Check if file size is too large
2. Verify session ID is valid
3. Check API server disk space
```

---

## 11. Summary

This SDK makes it easy to use EvoSpikeNet's powerful features from Python. It provides comprehensive functionality from basic text generation to complex multimodal distributed brain simulations and experiment management.

For detailed information and latest updates, please refer to the project's [GitHub repository](https://github.com/MasahiroAoki/EvoSpikeNet).
