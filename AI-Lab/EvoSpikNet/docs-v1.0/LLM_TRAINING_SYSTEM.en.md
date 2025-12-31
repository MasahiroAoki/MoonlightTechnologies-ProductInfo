<!-- Reviewed against source: 2025-12-21. English translation completed. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# LLM Training System

This system provides automated training for different types of Large Language Models (LLMs) with API support and Docker integration.

## Features

- **Multi-Modal Training**: Support for LangText, Vision, Audio, and MultiModal models
- **API-Driven**: RESTful API for job submission and monitoring
- **GPU/CPU Support**: Optimized for both GPU and CPU environments
- **Docker Integration**: Easy deployment with containerization
- **Load Balancing**: Nginx-based load balancer for multiple instances

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA Docker (for GPU training)
- At least 16GB RAM recommended

### 1. Data Collection

First, collect training data using the data collection script:

```bash
python scripts/collect_llm_training_data.py --config config/data_config.yaml
```

This will create the following directory structure:
```
data/llm_training/
├── LangText/
├── Vision/
├── Audio/
└── MultiModal/
```

### 2. Start Training Servers

#### GPU Training
```bash
./scripts/train_launcher.sh gpu
```

#### CPU Training
```bash
./scripts/train_launcher.sh cpu
```

#### Both GPU and CPU with Load Balancer
```bash
./scripts/train_launcher.sh all
```

### 3. Access API

- **GPU Server**: http://localhost:8000
- **CPU Server**: http://localhost:8001
- **Load Balancer**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs (FastAPI)

## API Usage

### Submit Training Job

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "LangText",
    "model_name": "microsoft/DialoGPT-medium",
    "dataset_path": "data/llm_training/LangText/langtext_data.jsonl",
    "output_dir": "saved_models/LangText",
    "gpu": true,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.00002
  }'
```

### Check Job Status

```bash
curl http://localhost:8000/jobs
curl http://localhost:8000/jobs/{job_id}
```

## Supported Categories

### LangText
- **Models**: GPT, BERT, DialoGPT
- **Features**: Text generation, language understanding
- **Use Cases**: Chatbots, text completion

### Vision
- **Models**: ViT, ResNet, CLIP
- **Features**: Image classification, object detection
- **Use Cases**: Image recognition, visual QA

### Audio
- **Models**: Whisper, Wav2Vec2
- **Features**: Speech recognition, audio classification
- **Use Cases**: Transcription, voice commands

### MultiModal
- **Models**: CLIP, LLaVA
- **Features**: Text-image understanding
- **Use Cases**: Visual question answering, image captioning

## Configuration

### Training Configuration (`config/training_config.yaml`)

```yaml
# Model configurations for each category
langtext:
  model_name: "microsoft/DialoGPT-medium"
  max_length: 512

vision:
  model_name: "google/vit-base-patch16-224"
  num_labels: 1000

audio:
  model_name: "openai/whisper-small"
  language: "en"

multimodal:
  model_name: "openai/clip-vit-base-patch32"

# Training parameters
training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
```

## Docker Commands

### Build and Run Manually

```bash
# GPU
docker build -f Dockerfile.train.gpu -t llm-trainer-gpu .
docker run -p 8000:8000 -v $(pwd)/data:/app/data llm-trainer-gpu

# CPU
docker build -f Dockerfile.train.cpu -t llm-trainer-cpu .
docker run -p 8000:8000 -v $(pwd)/data:/app/data llm-trainer-cpu
```

### Using Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.train.yml up -d

# View logs
docker-compose -f docker-compose.train.yml logs -f

# Stop services
docker-compose -f docker-compose.train.yml down
```

## Monitoring and Troubleshooting

### Check Server Status
```bash
./scripts/train_launcher.sh status
```

### View Logs
```bash
./scripts/train_launcher.sh logs gpu
./scripts/train_launcher.sh logs cpu
```

### Test API
```bash
./scripts/train_launcher.sh test
```

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training config
2. **Port conflicts**: Change ports in docker-compose.yml
3. **Model download fails**: Check internet connection and Hugging Face access
4. **NVIDIA Docker not available**: Use CPU mode or install NVIDIA Docker

## Advanced Usage

### Custom Model Training

Modify `config/training_config.yaml` to use different models:

```yaml
langtext:
  model_name: "meta-llama/Llama-2-7b-hf"  # Requires access token
```

### Scaling

For production deployment, consider:
- Kubernetes for orchestration
- Model versioning with MLflow
- Distributed training with Accelerate
- GPU resource management

## Web UI Integration

The LLM Training System is fully integrated with the EvoSpikeNet Web UI, providing user-friendly interfaces for training different types of models.

### Vision Encoder Training UI

**Location**: `frontend/pages/vision_encoder.py`

**Features**:
- **API Training Tab**: Dedicated tab for LLM training via REST API
- **Category Selection**: Dropdown to select model category (LangText, Vision, Audio, MultiModal)
- **Dynamic Type/Category Display**: Shows selected model type and category in real-time
- **Training Parameters**: Configurable epochs, batch size, learning rate, run name
- **GPU Support**: Checkbox to enable/disable GPU training
- **Training Status**: Real-time status updates and output display

**UI Components**:
```python
- vision-category-dropdown: Category selection (LangText/Vision/Audio/MultiModal)
- vision-selected-type-category: Dynamic display of model type and category
- vision-new-epochs-input: Number of training epochs
- vision-new-batch-size-input: Training batch size
- vision-new-learning-rate-input: Learning rate
- vision-new-run-name-input: Custom run name
- vision-new-gpu-checkbox: GPU training toggle
- vision-new-training-output: Training output display
- vision-new-training-status: Status messages
```

### Audio Encoder Training UI

**Location**: `frontend/pages/audio_encoder.py`

**Features**:
- **API Training Tab**: Dedicated tab for LLM training via REST API
- **Category Selection**: Dropdown to select model category (LangText, Vision, Audio, MultiModal)
- **Dynamic Type/Category Display**: Shows selected model type and category in real-time
- **Training Parameters**: Configurable epochs, batch size, learning rate, run name
- **GPU Support**: Checkbox to enable/disable GPU training
- **Training Status**: Real-time status updates and output display

**UI Components**:
```python
- audio-category-dropdown: Category selection (LangText/Vision/Audio/MultiModal)
- audio-selected-type-category: Dynamic display of model type and category
- audio-new-epochs-input: Number of training epochs
- audio-new-batch-size-input: Training batch size
- audio-new-learning-rate-input: Learning rate
- audio-new-run-name-input: Custom run name
- audio-new-gpu-checkbox: GPU training toggle
- audio-new-training-output: Training output display
- audio-new-training-status: Status messages
```

### Integration with Distributed Brain

The UI components automatically map categories to appropriate model types:
- **LangText** → `text` model type
- **Vision** → `vision` model type  
- **Audio** → `audio` model type
- **MultiModal** → `multimodal` model type

This ensures that training jobs are submitted with the correct parameters for the distributed brain node configuration.

### API Endpoints Used

The UI integrates with the following training API endpoints:
- `POST /train`: Submit training job
- Training jobs are submitted with category-specific model configurations
- Real-time status updates via callback functions

## License

This project is part of EvoSpikeNet and follows the same licensing terms.