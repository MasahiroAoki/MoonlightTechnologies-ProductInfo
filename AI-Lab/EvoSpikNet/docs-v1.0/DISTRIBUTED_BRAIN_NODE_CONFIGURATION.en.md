# Distributed Brain Simulation - Node Configuration and Model Mapping

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Last Updated:** December 10, 2025

> **Note:** This is a summary document. For complete technical specifications, please refer to the Japanese version: [DISTRIBUTED_BRAIN_NODE_CONFIGURATION.md](DISTRIBUTED_BRAIN_NODE_CONFIGURATION.md)

## Overview

EvoSpikeNet's distributed brain simulation adopts a hierarchical architecture where multiple nodes work cooperatively. Each node simulates a specific brain region and exchanges spike signals via the Zenoh communication protocol.

## Simulation Types

The distributed brain simulation supports the following 5 types:

1. **Language Focus** (10 processes) - Specialized in language processing
2. **Image Focus** (11 processes) - Specialized in visual processing
3. **Audio Focus** (12 processes) - Specialized in auditory and speech processing
4. **Motor Focus** (11 processes) - Specialized in motor control
5. **Full Brain** (21 processes) - Integrated all functions

## Node Hierarchy

### 1. Language Focus (10 processes)

Configuration centered on language processing, including advanced language functions such as embedding vector generation, task decomposition, and RAG retrieval.

```
Rank 0: PFC (Executive Control)
  ├─ Rank 1: Visual (Visual Input)
  ├─ Rank 2: Motor (Motor Output)
  ├─ Rank 3: Compute (Computational Processing)
  ├─ Rank 4: Lang-Main (Main Language Processing)
  ├─ Rank 5: Auditory (Auditory Input)
  ├─ Rank 6: Speech (Speech Output)
  ├─ Rank 7: Lang-Embed (Embedding Vector Generation)
  ├─ Rank 8: Lang-TAS (Task Decomposition)
  └─ Rank 9: Lang-RAG (RAG Retrieval)
```

**Models Used:**
- PFC: `SimpleLIFNode`
- Lang-Main, Lang-Embed, Lang-TAS, Lang-RAG: `SpikingEvoTextLM`
- Visual: `SpikingEvoVisionEncoder`
- Auditory, Speech: `SpikingEvoAudioEncoder`
- Motor: `SimpleLIFNode` (fallback)
- Compute: `SpikingEvoTextLM`

### 2. Image Focus (11 processes)

Emphasizes the visual processing hierarchy with 3-stage processing: edge detection, shape recognition, and object recognition.

**Additional Nodes:**
- Vis-Edge: Edge detection
- Vis-Shape: Shape recognition
- Vis-Object: Object recognition

### 3. Audio Focus (12 processes)

Emphasizes the auditory and speech processing hierarchy with 5-stage processing: MFCC feature extraction, phoneme recognition, semantic understanding, phoneme generation, and waveform synthesis.

**Additional Nodes:**
- Aud-MFCC: MFCC feature extraction
- Aud-Phoneme: Phoneme recognition
- Aud-Semantic: Semantic understanding
- Speech-Phoneme: Phoneme generation
- Speech-Wave: Waveform synthesis

### 4. Motor Focus (11 processes)

Emphasizes the motor control hierarchy with 3-stage processing: trajectory planning, cerebellar coordination, and PWM control.

**Additional Nodes:**
- Motor-Plan: Trajectory planning
- Motor-Cerebellum: Cerebellar coordination
- Motor-PWM: PWM control

### 5. Full Brain (21 processes)

Integrates all functional modules for comprehensive brain simulation.

## Model Mappings

Each node type is associated with specific SNN model classes:

- **Language Nodes**: Use `SpikingEvoTextLM` for text processing
- **Visual Nodes**: Use `SpikingEvoVisionEncoder` for image processing
- **Auditory/Speech Nodes**: Use `SpikingEvoAudioEncoder` for audio processing
- **Control Nodes**: Use `SimpleLIFNode` or `PFCDecisionEngine`

## Key Design Principles

1. **Hierarchical Processing**: Specialized sub-nodes handle specific aspects of processing
2. **Modular Architecture**: Each node can be developed and tested independently
3. **Scalable Communication**: Zenoh pub/sub enables efficient inter-node communication
4. **Dynamic Routing**: PFC dynamically routes tasks to appropriate functional modules

## References

- Implementation: `examples/run_zenoh_distributed_brain.py`
- Frontend Configuration: `frontend/pages/distributed_brain.py`
- Model Definitions: `evospikenet/models.py`

For detailed configuration parameters, node-specific behaviors, and complete model specifications, please refer to the Japanese documentation: [DISTRIBUTED_BRAIN_NODE_CONFIGURATION.md](DISTRIBUTED_BRAIN_NODE_CONFIGURATION.md)
