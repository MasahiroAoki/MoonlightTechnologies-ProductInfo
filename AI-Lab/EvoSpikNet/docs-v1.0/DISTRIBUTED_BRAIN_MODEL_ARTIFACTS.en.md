<!-- Reviewed against source: 2025-12-21. English translation completed. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# Model Artifacts List and Frontend Learning Parameter Mapping

> Implementation Note (Artifacts): For `artifact_manifest.json` output by training scripts and recommended CLI flags, refer to `docs/implementation/ARTIFACT_MANIFESTS.md`.

Created: December 21, 2025

This document determines and lists "implementable model artifact names" for the previously defined 24-node configuration. It also clarifies which artifact settings correspond to the main parameters specified in the frontend learning forms (LLM/encoder training).

---

## 1. Node-Specific Model Artifact Candidates

- Sensing Nodes (x4)
  - `sensing-camera-preproc-v1` (Image preprocessing pipeline)
  - `sensing-audio-preproc-v1` (Audio preprocessing pipeline)
  - `sensing-iot-normalizer-v1` (Sensor normalization)

- Encoder Nodes (x4)
  - Vision
    - `vit-base16-embed-v1` (ViT-base/16 → 768d embedding)
    - `resnet50-proj-v1` (ResNet50 + projector → 512d)
  - Audio
    - `wav2vec2-base-embed-v1` (wav2vec2-base → 512d)
    - `hubert-large-embed-v1` (HuBERT-large → 1024d)
  - Text
    - `sbert-all-mpnet-v1` (SBERT / mpnet-base-v2 → 768d)
  - Spiking / Event
    - `snn-dvs-embed-v1` (SNN / DVS embedding)

- Inference Nodes (x6)
  - LM (Short text/dialogue)
    - `gpt-small-v1` (GPT-based small ~300M)
    - `gpt-medium-v1` (GPT-based medium ~1.5B)
    - `gpt-large-v1` (GPT-based large ~6B) ※ As needed
  - Classifier/Detector
    - `yolox-s-intel-v1` (YOLOX small / detector)
    - `fasterrcnn-res50-v1` (Faster-RCNN Res50)
  - Spiking-LM
    - `spiking-lm-core-v1` (Spiking generation model)
  - Ensemble / Multimodal
    - `multimodal-ensemble-v1` (Multimodal integration layer)
  - RAG-support
    - `rag-lite-v1` (retriever + generation wrapper)

- Decision Nodes (x2)
  - Planner
    - `planner-rl-ppo-v1` (PPO-based planner)
  - Controller
    - `motor-controller-dnn-v1` (Controller model)

- Memory Nodes (x3)
  - Vector DB (separate from operational artifacts: config/index templates)
    - `milvus-schema-v1` (Vector DB schema definition)
  - Episodic storage
    - `minio-log-schema-v1`

- Learning Nodes (x1)
  - `trainer-ddp-manager-v1` (Distributed learning job management)

- Aggregation/Mediation Nodes (x2)
  - `federator-agg-v1` (Secure aggregation protocol)
  - `result-aggregator-v1` (Output aggregation, confidence evaluation)

- Management/Utility Nodes (x2)
  - `auth-service-v1` (API key/RBAC service)
  - `monitoring-stack-v1` (Prometheus/Grafana/ELK configuration)

---

## 2. Artifact Naming Convention (Recommended Meta)
- Format example: `<component>-<base-model>-<purpose>-v<major>`
  - Example: `vision-vit-base16-embed-v1` → component=vision, base-model=vit-base16, purpose=embed, version=v1
- Metadata to record (artifact manifest):
  - `artifact_name`, `model_version`, `base_model`, `task`, `embedding_dim`, `quantized` (bool), `precision` (fp32/fp16/int8), `training_config_hash`, `train_data_tags`, `license`, `created_at`, `node_type`, `privacy_level`

Implementation Notes:
- Generation scripts/training scripts create `artifact_manifest.json` in the run save directory and include it in the upload ZIP.
- CLI/frontend flags to use (existing implementation): `--artifact-name`, `--precision`, `--quantize`(store_true), `--privacy-level`, `--node-type`. These are reflected in the manifest.
- If `artifact_name` is not specified, it is auto-generated following the recommended format prefix (`{node_type}.{model_category}.{model_variant}.{run_name}.{timestamp}`).

---

## 3. Frontend Learning Form Parameters → Artifact Generation Mapping

When triggering training from the frontend, this shows which fields/settings of the finally generated artifact correspond to the main parameters input by the user.

- Input parameters (example):
  - `component` (selection): Corresponds to `artifact_name` prefix (e.g., `vision`, `audio`, `text`, `spiking`)
  - `base_model` (selection/text): Pre-trained base (e.g., `vit-base16`, `wav2vec2-base`, `gpt-small-v1`) → `base_model` meta
  - `task` (selection): `embed` / `classification` / `lm-finetune` / `detection` → `task` meta
  - `embedding_dim` (number): Embedding dimension → `embedding_dim`
  - `hidden_size`, `num_layers`, `num_heads` (number): Architecture changes → stored in `model_config`
  - `max_seq_length` / `sample_rate` / `input_size`: Model I/O specs → `input_spec`
  - `batch_size`, `learning_rate`, `optimizer`, `epochs`, `weight_decay`: Training settings → `training_config` (and generate `training_config_hash`)
  - `precision` (selection): `fp32`/`fp16`/`int8` → `precision`, `quantized` flag
  - `quantize` (bool): If True, execute quantization post-processing in job → set `quantized=true` and append to artifact name (e.g., `-int8`)
  - `checkpoint_interval` (number): Checkpoint save frequency → `checkpoint_policy`
  - `augmentations` / `preprocessing_profile`: Data preprocessing → `data_prep_profile`
  - `train_data_tags` (tag list): Which datasets were used → `train_data_tags` meta
  - `privacy_level` (selection): `none`/`dp`/`secure-agg` → Apply differential privacy or secure aggregation to training job

- Mapping examples (frontend input → generated artifact manifest):
  - `component=vision`, `base_model=vit-base16`, `task=embed`, `embedding_dim=768`, `precision=fp16`, `quantize=false`, `batch_size=256`, `epochs=10` →
    - artifact_name: `vision-vit-base16-embed-v1`
    - manifest: {"base_model":"vit-base16","task":"embed","embedding_dim":768,"precision":"fp16","training_config_hash":"<sha256>"}

  - `component=inference`, `base_model=gpt-small-v1`, `task=lm-finetune`, `max_seq_length=1024`, `learning_rate=2e-5`, `epochs=3`, `quantize=int8` →
    - artifact_name: `gpt-small-v1-lm-finetune-int8-v1`
    - manifest includes `quantized:true`, `precision:int8`, `input_spec:{max_seq_length:1024}`

---

## 4. Frontend Implementation Notes (Brief)
- When launching training jobs, always calculate `training_config_hash` (JSON normalization → SHA256) and link to artifact. This enables reproducibility and comparison.
- Quantization option should allow selection in UI between executing `post-training-quantize` step in job or training-time quantization aware (QAT).
- Privacy settings (differential privacy or secure aggregation) should be included in `privacy_level` in training job definition and propagated to Trainer/Aggregator.

---

## 5. Proposed Next Actions
1. Decide priority list of artifacts above (first 3 to create), and automate learning pipeline in CI. Recommended first 3: `vit-base16-embed-v1`, `wav2vec2-base-embed-v1`, `gpt-small-v1-lm-finetune-v1`.
2. Add above parameters to frontend learning form (`frontend/pages/settings.py` etc.), and design API to send training jobs via `api_client`.

---

File save location: `docs/DISTRIBUTED_BRAIN_MODEL_ARTIFACTS.md`