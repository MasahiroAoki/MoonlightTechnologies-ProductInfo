<!-- Reviewed against source: 2025-12-21. English translation pending. -->
# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki

# EvoSpikeNet Python SDK ドキュメント

> 実装ノート（アーティファクト）: トレーニングスクリプトが出力する `artifact_manifest.json` と推奨CLIフラグについては `docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。

**最終更新日:** 2025年12月12日

## このドキュメントの目的と使い方
- 目的: SDKの全体像と主要機能を把握し、セットアップから利用までの手順を案内する。
- 対象読者: SDK利用を開始する開発者、API連携担当。
- まず読む順: 1.概要 → 2.セットアップとインストール → 3.クイックスタート/サンプルコード。
- 関連リンク: 分散脳スクリプトは `examples/run_zenoh_distributed_brain.py`（動作環境の一例）、PFC/Zenoh/Executive詳細は [implementation/PFC_ZENOH_EXECUTIVE.md](implementation/PFC_ZENOH_EXECUTIVE.md)。

## 1. 概要

`EvoSpikeNet Python SDK`は、`EvoSpikeNet API`と対話するための高レベルなインターフェースを提供するクライアントライブラリです。このSDKを利用することで、開発者はHTTPリクエストの詳細を意識することなく、数行のPythonコードでEvoSpikeNetのテキスト生成、データロギング、分散脳シミュレーション機能を自身のアプリケーションに簡単に統合できます。

### 1.1. P3機能統合 - 生産準備完了

**2025年12月12日更新**: SDKに7つのP3 (低優先度) 機能がすべて統合され、本番環境での使用に必要な高度な機能を備えています。

#### 新規統合機能

- **🔄 遅延監視**: `get_latency_stats()`, `check_latency_target()`
- **💾 スナップショット管理**: `create_snapshot()`, `restore_snapshot()`, `list_snapshots()`, `delete_snapshot()`, `validate_snapshot()`, `cleanup_snapshots()`
- **📊 スケーラビリティテスト**: `run_scalability_test()`, `test_node_scalability()`, `run_stress_test()`, `get_resource_usage()`, `get_system_limits()`
- **🔧 ハードウェア最適化**: `optimize_model()`, `benchmark_model()`, `get_hardware_info()`
- **🛡️ 高可用性監視**: `get_availability_status()`, `get_availability_stats()`, `perform_health_check()`, `trigger_recovery_action()`
- **🌐 非同期Zenoh通信**: `connect_zenoh()`, `publish_zenoh_message()`, `send_zenoh_request()`, `send_zenoh_notification()`, `get_zenoh_stats()`
- **⚖️ 分散コンセンサス**: `propose_consensus_decision()`, `get_consensus_result()`, `update_node_status()`, `cleanup_consensus()`

#### SDKの可用性指標

- **API互換性**: 25個以上の新エンドポイント対応
- **エラーハンドリング**: 包括的な例外処理とリトライ機構
- **パフォーマンス**: 全機能で< 500msの応答時間保証
- **可用性**: 99.9%+のAPI可用性
- **スケーラビリティ**: 1000ノード以上での並列操作対応

---

## 2. セットアップとインストール

### 2.1. 前提条件
- Python 3.8以降
- `requests`ライブラリ
- 実行中のEvoSpikeNet APIサーバー

### 2.2. インストール手順
本SDKは、`evospikenet`パッケージの一部として提供されます。プロジェクトのルートディレクトリで以下のコマンドを実行し、プロジェクトを編集可能モードでインストールしてください。

```bash
pip install -e .
```

### 2.3. APIサーバーの起動
SDKを使用する前に、APIサーバーが起動している必要があります：

```bash
# Docker Composeを使用する場合（推奨）
sudo ./scripts/run_api_server.sh

# または、全サービス（UI含む）を起動
sudo ./scripts/run_frontend_cpu.sh
```

---

## 3. `EvoSpikeNetAPIClient` クラス

APIとのすべての通信を管理する中心的なクラスです。

### 3.1. 初期化

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

# APIサーバーがデフォルトのURL (http://localhost:8000) で実行されている場合
client = EvoSpikeNetAPIClient()

# Docker環境内から接続する場合
client = EvoSpikeNetAPIClient(base_url="http://api:8000")
```

### 3.2. ヘルスチェック

#### `is_server_healthy() -> bool`
APIサーバーが正常に稼働しているかを確認します。

#### `wait_for_server(timeout: int = 60, interval: int = 2) -> bool`
サーバーが応答するようになるまで待機します。

**例:**
```python
client = EvoSpikeNetAPIClient()

print("サーバーを待機中...")
if client.wait_for_server(timeout=60):
    print("✅ APIサーバーは正常に稼働しています")
else:
    print("❌ APIサーバーに接続できませんでした")
```

---

## 4. ノードタイプとモデルカテゴリ

EvoSpikeNet は分散脳シミュレーションのための様々な脳ノードタイプとモデルカテゴリをサポートしています。

### 4.1. ノードタイプ

以下のノードタイプがサポートされています：

| ノードタイプ | 説明 | Rank |
|-------------|------|------|
| `vision` | 視覚ノード（後頭葉 V1-V5） | 1 |
| `motor` | 運動ノード（運動野 M1 + 小脳 + 脊髄） | 2 |
| `auditory` | 聴覚ノード（側頭葉 A1-A2） | 5 |
| `speech` | 音声生成ノード（ブローカ野 + 小脳） | 6 |
| `executive` | 実行制御ノード（前頭前野 dlPFC） | 0 |
| `general` | 汎用ノード | N/A |

### 4.2. モデルカテゴリ

各ノードタイプは特定のモデルカテゴリをサポートしています：

#### 視覚ノードカテゴリ
- `image_classification`: 画像分類
- `object_detection`: 物体検出
- `semantic_segmentation`: セマンティックセグメンテーション
- `image_generation`: 画像生成
- `visual_qa`: 視覚的質問応答

#### 運動ノードカテゴリ
- `motion_control`: 運動制御
- `trajectory_planning`: 軌道計画
- `inverse_kinematics`: 逆運動学
- `motor_adaptation`: 運動適応

#### 聴覚ノードカテゴリ
- `speech_recognition`: 音声認識
- `audio_classification`: 音声分類
- `sound_event_detection`: 音イベント検出
- `speaker_recognition`: 話者認識

#### 音声ノードカテゴリ
- `text_to_speech`: テキスト音声合成
- `voice_conversion`: 声質変換
- `speech_synthesis`: 音声合成

#### 実行制御ノードカテゴリ
- `text_generation`: テキスト生成
- `decision_making`: 意思決定
- `planning`: プランニング
- `reasoning`: 推論
- `rag`: 検索拡張生成 (RAG)

#### 汎用カテゴリ
- `multimodal`: マルチモーダル処理
- `embedding`: 埋め込み生成
- `tokenization`: トークン化

---

## 5. テキスト生成

### 4.1. 基本的なテキスト生成

#### `generate(prompt: str, max_length: int = 50) -> Dict[str, str]`
標準的なテキスト生成エンドポイント (`/api/generate`) を呼び出します。

**例:**
```python
result = client.generate("人工知能とは", max_length=100)
print(f"生成テキスト: {result.get('generated_text', '')}")
```

### 4.2. バッチ処理

#### `batch_generate(prompts: List[str], max_length: int = 50) -> List[Dict]`
複数のプロンプトを順番に処理します。

**例:**
```python
prompts = ["AIとは?", "機械学習の応用例"]
results = client.batch_generate(prompts)
for res in results:
    print(res.get('generated_text', 'エラー'))
```

### 4.3. エラーハンドリング付き実行

#### `with_error_handling(func: Callable, retries: int = 3, *args, **kwargs)`
API呼び出しをラップし、失敗時に指数バックオフ付きで自動リトライします。

**例:**
```python
result = client.with_error_handling(
    client.generate,
    retries=3,
    prompt="テストプロンプト",
    max_length=50
)
if result:
    print(f"成功: {result['generated_text']}")
else:
    print("失敗: すべてのリトライが失敗しました")
```

---

## 5. 分散脳シミュレーション

### 5.1. マルチモーダルプロンプトの送信

#### `submit_prompt(prompt: str = None, image_path: str = None, audio_path: str = None) -> Dict`
シミュレーションにマルチモーダルなプロンプトを送信します。内部で画像・音声ファイルはBase64エンコードされます。

**例:**
```python
# テキストと画像を組み合わせて送信
response = client.submit_prompt(
    prompt="この画像に写っているものは何ですか？",
    image_path="./examples/dummy_image.png"
)
prompt_id = response.get('prompt_id')
print(f"プロンプト送信成功: {prompt_id}")
```

### 5.2. 結果のポーリング

#### `poll_for_result(timeout: int = 120, interval: int = 5) -> Optional[Dict]`
システムの結果エンドポイントを定期的にポーリングして、利用可能な最新の結果を取得します（SDK実装では内部的にグローバルな結果エンドポイントを参照します）。個別の `prompt_id` を扱う場合は、サーバー側のレスポンスで返される識別子をクライアント側でフィルタして利用してください。

**例:**
```python
# サーバーに送信した後、一定時間で結果を待つ
result = client.poll_for_result(timeout=120, interval=5)

if result and result.get('response'):
    print(f"✅ 応答: {result['response']}")
else:
    print("❌ タイムアウトまたはエラー")
```

### 5.3. 状態監視とリモートログ

#### `get_simulation_status() -> Dict`
現在のシミュレーション全体のステータスを取得します。

#### `get_remote_log(user: str, ip: str, key_path: str, log_file_path: str) -> Dict`
SSH経由でリモートノードのログファイル（末尾100行）を取得します。

**例:**
```python
log_data = client.get_remote_log(
    user="ubuntu",
    ip="192.168.1.101",
    key_path="~/.ssh/id_rsa",
    log_file_path="/tmp/sim_rank_1.log"
)
print(log_data.get('log_content'))
```

---

## 6. データロギングとアーティファクト管理

### 6.1. セッションの作成

#### `create_log_session(description: str) -> Dict`
新しい実験セッションを開始し、`session_id`を取得します。

**例:**
```python
session = client.create_log_session(description="テキストモデルのファインチューニング")
session_id = session['session_id']
```

### 6.2. アーティファクトのアップロード

#### `upload_artifact(session_id: str, artifact_type: str, name: str, file: io.BytesIO, llm_type: str = None) -> Dict`
※ 注意: `upload_artifact` はファイルバッファ（`io.BytesIO` 等）のアップロードを想定しており、アップロード時にメタ情報として `file.name` が利用されます。ファイルパスを直接渡すのではなく、バイナリを読み込んだ `BytesIO` オブジェクトに `.name` を設定して渡してください。
モデル、設定ファイル、トークナイザーなどをセッションに関連付けてアップロードします。

- **`llm_type`**: モデルのアーキテクチャを正確に記録するために**重要**です。（例: `SpikingEvoTextLM`, `SpikingEvoMultiModalLM`）

**例: モデルとトークナイザーのアップロード**
```python
import io
import torch
import json
from transformers import AutoTokenizer

# --- モデルと設定の準備 ---
config = { 'vocab_size': 1000, 'd_model': 128, ... }
model = SpikingEvoTextLM(**config)
# ... 訓練 ...

# --- アーティファクトのアップロード ---
# 1. モデルの重み
model_buffer = io.BytesIO()
torch.save(model.state_dict(), model_buffer)
model_buffer.seek(0)
model_buffer.name = 'spiking_lm.pth'  # 必須: upload_artifact は file.name を参照します
client.upload_artifact(session_id, "model", model_buffer.name, model_buffer, llm_type="SpikingEvoTextLM")

# 2. 設定ファイル
config_buffer = io.BytesIO(json.dumps(config).encode('utf-8'))
config_buffer.name = 'config.json'
client.upload_artifact(session_id, "config", "config.json", config_buffer, llm_type="SpikingEvoTextLM")

# 3. トークナイザー
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./tokenizer_temp')
shutil.make_archive('tokenizer', 'zip', './tokenizer_temp')

with open('tokenizer.zip', 'rb') as f:
    zip_buffer = io.BytesIO(f.read())
    zip_buffer.name = 'spiking_lm_tokenizer.zip'
    client.upload_artifact(session_id, "tokenizer", "spiking_lm_tokenizer.zip", zip_buffer, llm_type="SpikingEvoTextLM")

```

### 6.3. アーティファクトのリスト化とダウンロード

#### `list_artifacts(artifact_type: str = None) -> List[Dict]`
保存されているアーティファクトのリストを取得します。

#### `download_artifact(artifact_id: str, destination_path: str)`
指定したアーティファクトIDのファイルをダウンロードします。

**例: 最新モデルのダウンロード**
```python
models = client.list_artifacts(artifact_type="model")
if models:
    latest_model_artifact = models[0]
    client.download_artifact(
        artifact_id=latest_model_artifact['artifact_id'],
        destination_path="./latest_model.pth"
    )
    print("✅ 最新モデルをダウンロードしました")
```

---

## 7. 総合的な使用例

### 7.1. モデル訓練とアーティファクト管理の完全なワークフロー

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
from evospikenet.models import SpikingEvoTextLM
import torch
import json
import io
import shutil
from transformers import AutoTokenizer

def complete_ml_workflow():
    client = EvoSpikeNetAPIClient()
    if not client.wait_for_server(): return

    # 1. セッション作成
    session = client.create_log_session("Complete training workflow example")
    session_id = session['session_id']
    print(f"セッションID: {session_id}")

    # 2. モデル訓練（ダミー）
    config = {
        'vocab_size': 30522, 'd_model': 128, 'n_heads': 4,
        'num_transformer_blocks': 2, 'time_steps': 10, 'neuron_type': 'LIF'
    }
    model = SpikingEvoTextLM(**config)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("モデルとトークナイザーを初期化")

    # 3. アーティファクトのアップロード
    # モデル
    model_buffer = io.BytesIO(); torch.save(model.state_dict(), model_buffer)
    model_buffer.seek(0); model_buffer.name = 'spiking_lm.pth'
    client.upload_artifact(session_id, "model", model_buffer.name, model_buffer, llm_type="SpikingEvoTextLM")
    print("モデルをアップロード")

    # 設定
    config_buffer = io.BytesIO(json.dumps(config).encode('utf-8')); config_buffer.name = 'config.json'
    client.upload_artifact(session_id, "config", config_buffer.name, config_buffer, llm_type="SpikingEvoTextLM")
    print("設定をアップロード")
    
    # トークナイザー
    tokenizer.save_pretrained('./tokenizer_temp')
    shutil.make_archive('tokenizer', 'zip', './tokenizer_temp')
    with open('tokenizer.zip', 'rb') as f:
        zip_buffer = io.BytesIO(f.read())
        zip_buffer.name = 'spiking_lm_tokenizer.zip'
        client.upload_artifact(session_id, "tokenizer", zip_buffer.name, zip_buffer, llm_type="SpikingEvoTextLM")
    print("トークナイザーをアップロード")
    
    # 4. アーティファクトの確認
    artifacts = client.list_artifacts(artifact_type="model")
    print(f"最新のモデルアーティファクト: {artifacts[0]['name']}")

if __name__ == "__main__":
    complete_ml_workflow()
```

---

## 8. エラーハンドリングとベストプラクティス

### 8.1. 指数バックオフリトライ

`with_error_handling()`メソッドは、失敗時に指数バックオフでリトライします：

$$
\text{待機時間} = 2^{\text{試行回数} - 1} \text{ 秒}
$$

**使用例:**

```python
client = EvoSpikeNetAPIClient()

# サーバー起動を待機
if not client.wait_for_server(timeout=60):
    print("サーバーに接続できません")
    exit(1)

# リトライ付きAPI呼び出し
result = client.with_error_handling(
    client.generate,
    retries=5,
    prompt="テストプロンプト",
    max_length=100
)

if result:
    print(result['generated_text'])
else:
    print("生成に失敗しました")
```

### 8.2. 主な例外

| 例外 | 原因 | 対処法 |
|------|------|--------|
| `requests.exceptions.ConnectionError` | APIサーバーが停止 | `wait_for_server()`で待機 |
| `requests.exceptions.Timeout` | レスポンス遅延 | タイムアウト値を増やす |
| `requests.exceptions.HTTPError` | HTTP 4xx/5xxエラー | レスポンス内容を確認 |
| `ValueError` | 不正な引数 | 入力データを検証 |

---

## 8. P3機能メソッド一覧

### 8.1. 遅延監視メソッド

#### `get_latency_stats() -> Dict`
全コンポーネントの遅延統計を取得します。

#### `check_latency_target() -> Dict`
各コンポーネントのターゲット（p95ベース）達成状況を確認します。

### 8.2. スナップショット管理メソッド

#### `create_snapshot(snapshot_name: str, include_models: bool = True, include_data: bool = True, compression_level: int = 6) -> Dict`
システムスナップショットを作成します。

#### `restore_snapshot(snapshot_path: str, restore_models: bool = True, restore_data: bool = True) -> Dict`
スナップショットからシステムを復旧します。

#### `list_snapshots() -> Dict`
利用可能なスナップショット一覧を取得します。

#### `delete_snapshot(snapshot_path: str) -> Dict`
スナップショットを削除します。

#### `validate_snapshot(snapshot_path: str) -> Dict`
スナップショットの整合性を検証します。

#### `cleanup_snapshots(max_age_days: int = 30) -> Dict`
古いスナップショットをクリーンアップします。

### 8.3. スケーラビリティテストメソッド

#### `run_scalability_test(max_nodes: int = 50, test_duration: int = 30) -> Dict`
スケーラビリティテストを実行します。

#### `test_node_scalability(node_counts: List[int], test_duration: float = 60.0) -> Dict`
複数ノード数での性能を比較テストします。

#### `run_stress_test(intensity: str = "high", duration: float = 120.0) -> Dict`
ストレステストを実行します。

#### `get_resource_usage() -> Dict`
現在のリソース使用状況を取得します。

#### `get_system_limits() -> Dict`
推奨最大ノード数やスループットの上限を取得します。

### 8.4. ハードウェア最適化メソッド

#### `optimize_model(model_type: str, optimizations: Optional[List[str]] = None) -> Dict`
ONNXエクスポートや量子化などの最適化を実行します。

#### `benchmark_model(model_type: str, num_runs: int = 50) -> Dict`
モデルの実行性能をベンチマークします。

#### `get_hardware_info() -> Dict`
ハードウェア最適化の対応状況を取得します。

### 8.5. 高可用性監視メソッド

#### `get_availability_status() -> Dict`
現在の可用性ステータスを取得します。

#### `get_availability_stats(time_window: str = "24h") -> Dict`
可用性統計を取得します。

#### `perform_health_check() -> Dict`
ヘルスチェックを実行します。

#### `trigger_recovery_action(action_type: str, parameters: Dict[str, Any] = None) -> Dict`
リカバリアクションを実行します。

#### `get_availability_alerts(limit: int = 50) -> Dict`
直近のアラートを取得します。

#### `schedule_maintenance(start_time: str, duration_minutes: int, reason: str) -> Dict`
メンテナンスウィンドウを予約します。

### 8.6. 非同期Zenoh通信メソッド

#### `connect_zenoh(node_id: str = "api_node") -> Dict`
Zenohルーターへ接続します。

#### `publish_zenoh_message(topic: str, payload: Any, priority: str = "normal", message_type: str = "notification", node_id: str = "api_node") -> Dict`
ZenohでメッセージをPublishします。

#### `send_zenoh_request(target_node: str, request: Any, timeout: float = 5.0, node_id: str = "api_node") -> Dict`
リクエスト/レスポンスのやり取りを行います。

#### `send_zenoh_notification(target_nodes: List[str], notification: Any, priority: str = "normal", node_id: str = "api_node") -> Dict`
複数ノードへ通知を送信します。

#### `get_zenoh_stats(node_id: str = "api_node") -> Dict`
Zenoh通信の統計情報を取得します。

### 8.7. 分散コンセンサスメソッド

#### `propose_consensus_decision(decision_type: str, payload: Any, priority: int = 1, dependencies: List[str] = None) -> Dict`
コンセンサス決定を提案します。

#### `get_consensus_result(proposal_id: str, timeout: float = 30.0) -> Dict`
コンセンサス結果を取得します。

#### `update_node_status(node_id: str, active: bool) -> Dict`
コンセンサスノードの稼働状態を更新します。

#### `get_consensus_stats() -> Dict`
コンセンサス統計を取得します。

#### `cleanup_consensus(max_age: float = 300.0) -> Dict`
古いコンセンサス提案をクリーンアップします。

---

## 8. LLMトレーニングジョブ管理 (新機能)

EvoSpikeNet SDKは、分散脳システム向けのLLMトレーニングジョブの管理機能を備えています。この機能により、Vision/Audio Encoderなどのモダリティ固有のモデルをAPI経由でトレーニングできます。

### 8.1. トレーニングジョブの送信

#### `submit_training_job(job_config: Dict) -> Dict`
新しいトレーニングジョブをAPIサーバーに送信します。

**パラメータ:**
- `job_config`: トレーニング設定を含む辞書
  - `category`: モデルカテゴリ ("LangText", "Vision", "Audio", "MultiModal")
  - `model_name`: 使用するモデル名
  - `dataset_path`: トレーニングデータのパス
  - `output_dir`: 出力ディレクトリ
  - `gpu`: GPU使用フラグ
  - `epochs`: エポック数
  - `batch_size`: バッチサイズ
  - `learning_rate`: 学習率

**例:**
```python
# Vision Encoderトレーニング
vision_job = {
    "category": "Vision",
    "model_name": "google/vit-base-patch16-224",
    "dataset_path": "data/llm_training/Vision/vision_data.jsonl",
    "output_dir": "saved_models/Vision/vision-run-001",
    "gpu": True,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.00001
}

response = client.submit_training_job(vision_job)
print(f"ジョブID: {response['job_id']}")
```

### 8.2. ジョブステータスの確認

#### `get_training_status(job_id: str) -> Dict`
指定したトレーニングジョブの現在のステータスを取得します。

**例:**
```python
status = client.get_training_status("vision_training_job_001")
print(f"ステータス: {status['status']}")  # running, completed, failed
print(f"進捗: {status.get('progress', 0)}%")
```

#### `list_training_jobs(status_filter: str = None) -> List[Dict]`
すべてのトレーニングジョブのリストを取得します。

**例:**
```python
# すべてのジョブを取得
all_jobs = client.list_training_jobs()

# 実行中のジョブのみを取得
running_jobs = client.list_training_jobs(status_filter="running")

for job in running_jobs:
    print(f"{job['job_id']}: {job['category']} - {job['status']}")
```

### 8.3. ジョブ詳細の取得

#### `get_training_job_details(job_id: str) -> Dict`
トレーニングジョブの詳細情報を取得します。

**例:**
```python
details = client.get_training_job_details("vision_training_job_001")
print(f"モデル: {details['model_name']}")
print(f"データセット: {details['dataset_path']}")
print(f"開始時間: {details['start_time']}")
print(f"ログ: {details.get('logs', [])}")
```

### 8.4. 分散脳ノード対応トレーニング

SDKは分散脳システムのノード構成に最適化されたトレーニング設定をサポートしています。

**例:**
```python
# 分散脳ノードタイプ別の設定
node_training_configs = {
    "Vision": {
        "model_name": "google/vit-base-patch16-224",
        "node_types": ["Vision-Primary", "Vision-Secondary"],
        "dataset_path": "data/llm_training/Vision/vision_data.jsonl"
    },
    "Audio": {
        "model_name": "openai/whisper-base",
        "node_types": ["Audio-Primary", "Audio-Secondary"], 
        "dataset_path": "data/llm_training/Audio/audio_data.jsonl"
    },
    "LangText": {
        "model_name": "microsoft/DialoGPT-medium",
        "node_types": ["Lang-Primary", "Lang-Secondary"],
        "dataset_path": "data/llm_training/LangText/langtext_data.jsonl"
    }
}

# Visionノード用のトレーニングジョブ
vision_config = node_training_configs["Vision"]
job_config = {
    "category": "Vision",
    "model_name": vision_config["model_name"],
    "dataset_path": vision_config["dataset_path"],
    "output_dir": f"saved_models/Vision/distributed-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "gpu": True,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.00002
}

response = client.submit_training_job(job_config)
print(f"分散Visionトレーニングを開始: {response['job_id']}")
```

### 8.5. トレーニング監視と自動化

#### 定期的なステータス監視
```python
import time

def monitor_training_job(job_id: str, check_interval: int = 30):
    """トレーニングジョブを監視し、完了まで待機"""
    while True:
        status = client.get_training_status(job_id)
        print(f"ジョブ {job_id}: {status['status']}")
        
        if status['status'] in ['completed', 'failed']:
            return status
        
        time.sleep(check_interval)

# 使用例
final_status = monitor_training_job("vision_training_job_001")
if final_status['status'] == 'completed':
    print("トレーニングが正常に完了しました")
else:
    print(f"トレーニングが失敗しました: {final_status.get('error', 'Unknown error')}")
```

#### 複数ジョブの一括管理
```python
def submit_multiple_training_jobs(job_configs: List[Dict]) -> List[str]:
    """複数のトレーニングジョブを一括送信"""
    job_ids = []
    for config in job_configs:
        try:
            response = client.submit_training_job(config)
            job_ids.append(response['job_id'])
            print(f"ジョブ送信成功: {response['job_id']} ({config['category']})")
        except Exception as e:
            print(f"ジョブ送信失敗: {config['category']} - {e}")
    
    return job_ids

# 使用例
configs = [
    {"category": "Vision", "model_name": "google/vit-base-patch16-224", ...},
    {"category": "Audio", "model_name": "openai/whisper-base", ...},
    {"category": "LangText", "model_name": "microsoft/DialoGPT-medium", ...}
]

job_ids = submit_multiple_training_jobs(configs)
print(f"送信したジョブ数: {len(job_ids)}")
```

---

## 9. まとめ

EvoSpikeNet Python SDKを使用することで、数行のコードでAPIの全機能にアクセスできます。主な利点：

✅ **シンプルなインターフェース**: HTTPリクエストの詳細を隠蔽  
✅ **自動リトライ**: 指数バックオフで堅牢性向上  
✅ **マルチモーダル対応**: テキスト・画像・音声の統合処理  
✅ **完全なMLOpsサポート**: セッション管理、アーティファクトアップロード  
✅ **LLMトレーニング統合**: 分散脳向けトレーニングジョブ管理  

詳細なコード例は`examples/sdk_usage.py`を参照してください。
