# Release Notes

Version history and changelog for the EvoSpikeNet framework.

## 📋 Version History

---

## [v1.0.0] - 2025-12-17

### 🎉 Initial Release - Production Ready

**Status**: 🟢 Production Ready

### ✨ Key Features

#### Distributed Brain System
- **Zenoh Communication**: High-speed asynchronous messaging
- **Node Discovery**: Automatic node discovery and health checks
- **PFC Feedback**: Advanced decision-making with prefrontal cortex

#### RAG System
- **Hybrid Search**: Milvus (vector search) + Elasticsearch (full-text search)
- **Japanese Optimization**: High-precision search with morphological analysis
- **Debug Features**: Error tracking and performance analysis visualization

#### SDK & API
- **Type Safety**: Complete type hints, Enum, and Dataclass support
- **Error Handling**: Custom exceptions, automatic retry, and connection pooling
- **Jupyter Integration**: Magic commands and rich HTML output
- **Validation Tools**: APIValidator, performance metrics, and benchmarks

#### Web UI
- **3D Visualization**: Neuron visualization with Three.js
- **Real-time Monitoring**: Dynamic display of neuron activity
- **Management Features**: Dataset/model management and configuration UI

#### CI/CD
- **GitHub Actions**: Automated testing, building, and deployment
- **Docker Integration**: Automatic container builds
- **Quality Assurance**: Test coverage over 92%

### 🔧 Technical Specifications

#### Architecture
- **SNN Models**: LIF, AdEx, and Izhikevich neurons
- **Plasticity**: STDP learning rules
- **Parallel Processing**: Multi-process and asynchronous processing support

#### Databases
- **Milvus**: Vector database (cosine similarity search)
- **Elasticsearch**: Full-text search engine
- **RRF Fusion**: Result integration with Reciprocal Rank Fusion

#### API
- **REST API**: FastAPI-based high-speed API
- **Authentication**: JWT authentication support
- **OpenAPI**: Automatic Swagger generation

### 🐛 Major Bug Fixes

- Fixed RAG system UnboundLocalError (re import)
- Fixed rag() method return value inconsistency
- Resolved Web UI port conflict errors
- Resolved Docker volume permission issues

### 📚 Documentation

- **Integrated Documentation Site**: MkDocs + Material theme
- **Multilingual Support**: Japanese and English bilingual (mkdocs-static-i18n)
- **80+ Pages**: Comprehensive guides and references
- **Search Functionality**: Full-text search support

### 📊 Statistics

- **Total Lines of Code**: 50,000+ lines
- **Test Coverage**: 92%
- **Documentation Pages**: 80+ pages
- **Implementation Completion**: 100%

### 🚀 Installation

```bash
# Install from PyPI (planned)
pip install evospikenet

# Or development version
git clone https://github.com/moonlight-tech/EvoSpikeNet.git
cd EvoSpikeNet
pip install -e .
```

### 💻 Quick Start

```python
from evospikenet import EvoSpikeNetSDK

# Initialize SDK client
sdk = EvoSpikeNetSDK(api_url="http://localhost:8000")

# RAG search
results = sdk.rag_search(
    query="What is a spiking neural network?",
    top_k=5
)

for doc, score in results:
    print(f"Score: {score:.3f}, Document: {doc}")
```

---

## 📊 Version Comparison

| Version    | Release Date | Key Features                               | Status       |
| ---------- | ------------ | ------------------------------------------ | ------------ |
| **v1.0.0** | 2025-12-17   | Distributed Brain, RAG, SDK, Web UI, CI/CD | 🟢 Production |

---

## 📧 Feedback

Please report bugs and feature requests via [GitHub Issues](https://github.com/moonlight-tech/EvoSpikeNet/issues).

---

## [v0.1.2] - 2025-12-31

### 🧠 Long-Term Memory Nodes Implementation ⭐ NEW

**Long-Term Memory System for Distributed Brain Simulation**
- **`LongTermMemoryNode`**: Base class integrating FAISS vector search and Zenoh distributed communication
- **`EpisodicMemoryNode`**: Storage and retrieval of time-sequenced events (experiences)
- **`SemanticMemoryNode`**: Persistence of concepts and knowledge with linking
- **`MemoryIntegratorNode`**: Cross-modal integration of episodic/semantic memories
- **Implementation File**: `evospikenet/memory_nodes.py` (355 lines)
- **Tests**: `tests/test_memory_nodes.py` (242 lines)

### 🎵 Audio Training UI Improvements

**Unified Audio Training Callbacks**
- Eliminated duplicate callback registration across multiple pages
- Improved maintainability through shared callback system
- **Implementation File**: `frontend/pages/audio_training_callbacks.py`

### 📚 Documentation Updates

**New English Translations (10 files)**
- DEV_BATCH_SHAPING.en.md
 - DISTRIBUTED_BRAIN_24NODE_EVALUATION.en.md
- DISTRIBUTED_BRAIN_MODEL_ARTIFACTS.en.md
- DISTRIBUTED_BRAIN_NODE_TYPES.en.md
- DISTRIBUTED_BRAIN_SEQUENCE.en.md
- LLM_TRAINING_SYSTEM.en.md
- SDK_CONFIGURATION.en.md
- api/README.en.md
- implementation/ARTIFACT_MANIFESTS.en.md

**Updated Documentation**
- DISTRIBUTED_BRAIN_SEQUENCE.md: 24-node full brain architecture and long-term memory integration
- REMAINING_FEATURES.md: Implementation status updates
- EPISODIC_MEMORY_IMPLEMENTATION.md: New memory node information

### 🔧 Internal Improvements
- RAG system frontend improvements
- Vision/Audio encoder improvements (800+ lines updated)
 - Distributed brain 24-node evaluation documentation

---

**Last Updated**: December 31, 2025  
**Current Version**: v0.1.2
