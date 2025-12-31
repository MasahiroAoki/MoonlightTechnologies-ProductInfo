<!-- This is the English version of DISTRIBUTED_BRAIN_NODE_TYPES.md. Translation pending. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# Complete Brain (Full Brain) Implementation for Distributed Brain Nodes: Concepts, Configuration, and 24-Node Pipeline

This document serves as a design specification for implementing the Distributed Brain as a "Complete Brain". It summarizes concepts, overall overview, process configuration, specific 24-node process pipeline, and clarifies the functions provided by the complete brain, including models (LLM/encoders/detectors), data, and learning methods assigned to each node.

---

## 1. Concept
- The complete brain implements biological brain functional differentiation (sensation → encoding → cognition → decision-making → motor) as distributed nodes, where each node has specialization and cooperates to achieve higher-order functions.
- Each node has "models (LLM/encoders/detectors)" and "data/memory/communication channels", exchanging messages (observations, embeddings, inference results, commands) with low latency.

> Implementation Note (Artifacts): For model artifact generation per node, refer to `docs/implementation/ARTIFACT_MANIFESTS.md`. It describes required items in `artifact_manifest.json` and CLI flag specifications (`--artifact-name` / `--node-type`, etc.).

## 2. Overall Overview
- Layer structure:
	- Sensing layer: Acquires data from physical sensors
	- Encoding layer: Converts observation data to features/embeddings
	- Cognition/Inference layer: Performs semantic understanding, classification, and generation using embeddings
	- Memory layer: Maintains context/history and provides search
	- Learning layer: Handles continuous learning and fine-tuning of models
	- Decision-making layer: Determines actions from high-level goals and outputs to actuators
	- Management layer: Authentication, monitoring, logging, configuration distribution

## 3. Process Configuration
- Message format: Standardized payload with metadata (timestamp, node_id, model_version, embedding_dims, confidence)
- Communication: Hybrid of Pub/Sub (Zenoh, etc.) and REST/gRPC. Important operations are authenticated (`X-API-Key` / mTLS).
- Consistency: Minimize state and process event-driven. Long-term state is stored in memory nodes.

## 4. 24-Node Process Pipeline (Proposal)
Below is a concrete example expressing the "Complete Brain" in 24 nodes. Each node shows role, assigned model, main data source, and learning method.

### Node Distribution (Summary)
- Sensing: 3 nodes
- Encoders: 4 nodes
- Inference/LM: 5 nodes
- Decision: 3 nodes
- Long-Term Memory: 2 nodes
- Memory/Retriever: 5 nodes
- Trainer: 1 node
- Aggregator/Federator: 2 nodes
- Management/Utility: 2 nodes

### Details of Each Node

- Nodes 1-3: Sensing Nodes (Sensing x3)
	- Role: Collect camera input, microphone input, environmental sensors (temperature/IMU, etc.) and perform initial filtering/synchronization
	- Model: Lightweight preprocessing (noise removal, normalization). Possibly on-device simple encoders in some cases
	- Data: Camera streams (video), microphone (WAV), IoT sensor time series
	- Learning: Data augmentation, self-supervised filters (noise resistance improvement)

- Node 4: Vision Encoder
	- Role: Image → embedding conversion
	- Model: ViT/Vision Transformer series or ResNet→Projection, or Spiking-ViT for events
	- Data: ImageNet, COCO, domain-specific data (with metadata at collection time)
	- Learning: Pre-training (large-scale data) → domain fine-tuning, continuous learning if applicable

- Node 5: Audio Encoder
	- Model: Wav2Vec2 / HuBERT → embedding
	- Data: LibriSpeech, AudioSet, domain-specific audio corpus
	- Learning: Pre-training + task fine-tuning (audio classification/transcription)

- Node 6: Text Encoder
	- Model: SentenceTransformer (SBERT series) or lightweight transformer embedding
	- Data: Wikipedia, CC-News, specialized domain corpus
	- Learning: Pre-training → task fine-tuning (semantic search)

- Node 7: Spiking Encoder
	- Model: SNN (Spiking Neural Network) based encoder (for event cameras)
	- Data: DVS (Dynamic Vision Sensor) datasets, etc.
	- Learning: STDP/Surrogate-gradient training / transfer learning

- Nodes 8-12: Inference Nodes (Inference x5)
	- Node 8: LM-Inference (short text/dialogue)
		- Model: Small to medium-sized transformer LM (hundreds of M to billions of parameters)
		- Data: Conversation corpus, system prompts, history
		- Learning: Online fine-tuning of pre-trained models (retraining managed by Trainer node)

	- Node 9: Classifier/Detector
		- Model: YOLOvX / Faster-RCNN / ResNet-based classifier
		- Data: COCO, OpenImages, dedicated annotations
		- Learning: Transfer learning + continuous labeling (Human-in-the-loop)

	- Node 10: Spiking-LM (biologically-oriented generation)
		- Model: Generation/memory interface using spiking neural networks
		- Data: Sensor time series + event history
		- Learning: Online adaptation mimicking biology (fine-tuning with small data)

	- Node 11: Ensemble / Multimodal Inference
		- Role: Integrate outputs from encoder groups/inference nodes to generate high-confidence output
		- Method: Weighted ensemble, meta-learning for confidence estimation

	- Node 12: Retriever-Augmented Generation (RAG)
		- Role: Support LM inference by adding context from memory nodes
		- Model: Lightweight retriever + LM

- Nodes 13-14: Long-Term Memory Nodes (Long-Term Memory x2)
	- Role: Manage episodic memory (event-based) and semantic memory (knowledge-based)
	- Model: FAISS-based vector search, Zenoh communication integration
	- Data: Spike embeddings, time-series events, metadata