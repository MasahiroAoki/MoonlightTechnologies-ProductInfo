<!-- Reviewed against source: 2025-12-21. English translation pending. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# モデルアーティファクト一覧とフロントエンド学習パラメータマッピング

> 実装ノート（アーティファクト）: トレーニングスクリプトが出力する `artifact_manifest.json` と推奨CLIフラグについては `docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。

作成日: 2025-12-21

このドキュメントは、前に定義した24ノード構成に対して「実装可能なモデルアーティファクト名」を決定し一覧化したものです。併せて、フロントエンドの学習フォーム（LLM／エンコーダ学習）で指定する主要パラメータが、どのアーティファクト設定に対応するかを明確にします。

---

## 1. ノード別モデルアーティファクト候補

 - 観測ノード（Sensing x4）
  - `sensing-camera-preproc-v1` (画像前処理パイプライン)
  - `sensing-audio-preproc-v1` (音声前処理パイプライン)
  - `sensing-iot-normalizer-v1` (センサ正規化)

- エンコードノード（Encoders x4）
  - Vision
    - `vit-base16-embed-v1` (ViT-base/16 → 768d embedding)
    - `resnet50-proj-v1` (ResNet50 + projector → 512d)
  - Audio
    - `wav2vec2-base-embed-v1` (wav2vec2-base → 512d)
    - `hubert-large-embed-v1` (HuBERT-large → 1024d)
  - Text
    - `sbert-all-mpnet-v1` (SBERT / mpnet-base-v2 → 768d)
  - Spiking / Event
    - `snn-dvs-embed-v1` (SNN / DVS向け埋め込み)

 - 推論ノード（Inference x6）
  - LM (短文/対話)
    - `gpt-small-v1` (GPT系 小型 ~300M)
    - `gpt-medium-v1` (GPT系 中型 ~1.5B)
    - `gpt-large-v1` (GPT系 大型 ~6B)  ※需要に応じ
  - Classifier/Detector
    - `yolox-s-intel-v1` (YOLOX small / detector)
    - `fasterrcnn-res50-v1` (Faster-RCNN Res50)
  - Spiking-LM
    - `spiking-lm-core-v1` (スパイキング生成モデル)
  - Ensemble / Multimodal
    - `multimodal-ensemble-v1` (マルチモーダル統合レイヤ)
  - RAG-support
    - `rag-lite-v1` (retriever + generation wrapper)

- 意思決定ノード（Decision x2）
  - Planner
    - `planner-rl-ppo-v1` (PPOベースプランナー)
  - Controller
    - `motor-controller-dnn-v1` (制御器モデル)

 - 記憶ノード（Memory x3）
  - Vector DB (運用アーティファクトとは別に: config / index templates)
    - `milvus-schema-v1` (ベクトルDBスキーマ定義)
  - Episodic storage
    - `minio-log-schema-v1`

- 学習ノード（Trainer x1）
  - `trainer-ddp-manager-v1` (分散学習ジョブ管理)

- 集約／調停ノード（Aggregator x2）
  - `federator-agg-v1` (安全集約プロトコル)
  - `result-aggregator-v1` (出力集約、信頼度評価)

- 管理／ユーティリティ（Management x2）
  - `auth-service-v1` (APIキー・RBACサービス)
  - `monitoring-stack-v1` (Prometheus/Grafana/ELK設定)

---

## 2. アーティファクト命名規則（推奨メタ）
- フォーマット例: `<component>-<base-model>-<purpose>-v<major>`
  - 例: `vision-vit-base16-embed-v1` → component=vision, base-model=vit-base16, purpose=embed, version=v1
- 記録するメタデータ（artifact manifest）:
  - `artifact_name`, `model_version`, `base_model`, `task`, `embedding_dim`, `quantized` (bool), `precision` (fp32/fp16/int8), `training_config_hash`, `train_data_tags`, `license`, `created_at`, `node_type`, `privacy_level`

注意事項（実装上の仕様）:
- 生成スクリプト/トレーニングスクリプトは `artifact_manifest.json` を run の保存ディレクトリに作成し、アップロードZIPに含めます。
- CLI/フロントエンドで利用するフラグ名（既存実装）: `--artifact-name`, `--precision`, `--quantize`（store_true）, `--privacy-level`, `--node-type`。これらは manifest に反映されます。
- `artifact_name` を未指定にした場合は自動生成され、推奨形式のプレフィックス（`{node_type}.{model_category}.{model_variant}.{run_name}.{timestamp}`）に従います。

---

## 3. フロントエンド学習フォームのパラメータ → アーティファクト生成マッピング

フロントエンドでトレーニングをトリガーする際、ユーザーが入力する主要パラメータとそれが最終的に生成されるアーティファクトのどのフィールド／設定に反映されるかを示します。

- 入力パラメータ（例）:
  - `component` (選択): 対応アーティファクトの `artifact_name` プレフィックス（例: `vision`, `audio`, `text`, `spiking`）
  - `base_model` (選択/テキスト): 事前学習済みベース（例: `vit-base16`, `wav2vec2-base`, `gpt-small-v1`）→ `base_model` メタ
  - `task` (選択): `embed` / `classification` / `lm-finetune` / `detection` → `task` メタ
  - `embedding_dim` (数値): 埋め込み次元 → `embedding_dim`
  - `hidden_size`, `num_layers`, `num_heads` (数値): アーキテクチャ上の変更 → `model_config` に格納
  - `max_seq_length` / `sample_rate` / `input_size`: モデルの入出力仕様 → `input_spec`
  - `batch_size`, `learning_rate`, `optimizer`, `epochs`, `weight_decay`: トレーニング設定 → `training_config`（および `training_config_hash` を生成）
  - `precision` (選択): `fp32`/`fp16`/`int8` → `precision`、`quantized` フラグ
  - `quantize` (bool): Trueなら量子化ポスト処理をジョブ内で実行 → `quantized=true` としてアーティファクト名に付記（例: `-int8`）
  - `checkpoint_interval` (数値): チェックポイント保存頻度 → `checkpoint_policy`
  - `augmentations` / `preprocessing_profile`: データ前処理→ `data_prep_profile`
  - `train_data_tags` (タグリスト): どのデータセットを使ったか → `train_data_tags` メタ
  - `privacy_level` (選択): `none`/`dp`/`secure-agg` → 学習ジョブに差分プライバシーやセキュア集約を適用

- マッピング例（フロントエンド入力 → 生成されるartifact manifest）:
  - `component=vision`, `base_model=vit-base16`, `task=embed`, `embedding_dim=768`, `precision=fp16`, `quantize=false`, `batch_size=256`, `epochs=10` →
    - artifact_name: `vision-vit-base16-embed-v1`
    - manifest: {"base_model":"vit-base16","task":"embed","embedding_dim":768,"precision":"fp16","training_config_hash":"<sha256>"}

  - `component=inference`, `base_model=gpt-small-v1`, `task=lm-finetune`, `max_seq_length=1024`, `learning_rate=2e-5`, `epochs=3`, `quantize=int8` →
    - artifact_name: `gpt-small-v1-lm-finetune-int8-v1`
    - manifest includes `quantized:true`, `precision:int8`, `input_spec:{max_seq_length:1024}`

---

## 4. フロントエンド側での実装注意点（短く）
- 学習ジョブ起動時には必ず `training_config_hash` を計算（JSON正規化→SHA256）し、アーティファクトに紐づける。これにより再現性と比較が可能。 
- 量子化オプションはジョブ内で `post-training-quantize` ステップを実行するか、トレーニング中の量子化対応（QAT）を選択できるようUIで選べるようにする。
- プライバシー設定（差分プライバシーやセキュア集約）は、`privacy_level` をトレーニングジョブ定義に入れ、Trainer/Aggregatorに伝搬する。

---

## 5. 次のアクション提案
1. 上記アーティファクトの優先リスト（最初に作る3つ）を決め、学習パイプラインをCIで自動化する。推奨最初の3つ: `vit-base16-embed-v1`, `wav2vec2-base-embed-v1`, `gpt-small-v1-lm-finetune-v1`。
2. フロントエンドの学習フォーム（`frontend/pages/settings.py` など）に上のパラメータを追加し、`api_client` 経由で学習ジョブを送信するAPIを設計。

---

ファイル保存先: `docs/DISTRIBUTED_BRAIN_MODEL_ARTIFACTS.md`
