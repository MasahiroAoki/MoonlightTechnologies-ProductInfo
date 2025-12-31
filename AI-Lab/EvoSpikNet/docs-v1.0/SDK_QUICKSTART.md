<!-- Reviewed against source: 2025-12-21. English translation pending. -->
# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet SDK クイックスタートガイド

**最終更新日:** 2025年12月15日
**30秒で始めるEvoSpikeNet SDK**

## このドキュメントの目的と使い方
- 目的: SDKを最短でセットアップし、APIクライアントを動かす手順を示す。
- 対象読者: SDK利用を開始する開発者。
- まず読む順: インストール → APIサーバー起動 → 最小限の使用例。
- 関連リンク: 分散脳スクリプトは `examples/run_zenoh_distributed_brain.py`（動作確認環境として）、PFC/Zenoh/Executive詳細は [implementation/PFC_ZENOH_EXECUTIVE.md](implementation/PFC_ZENOH_EXECUTIVE.md)。
- 実装ノート（アーティファクト）: `docs/implementation/ARTIFACT_MANIFESTS.md` — `artifact_manifest.json` と CLI フラグの仕様（`--artifact-name` / `--precision` / `--quantize` / `--privacy-level` / `--node-type`）を参照してください。

## インストール

```bash
pip install -e .
```

## APIサーバーの起動

```bash
sudo ./scripts/run_api_server.sh
```

## 最小限の使用例

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

# クライアント初期化
client = EvoSpikeNetAPIClient()

# サーバーの確認
if client.wait_for_server():
    # テキスト生成
    result = client.generate("人工知能とは")
    print(result['generated_text'])
```

---

## よく使うパターン

### 1️⃣ シンプルなテキスト生成

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()
result = client.generate("機械学習の応用例を5つ列挙してください")
print(result['generated_text'])
```

### 2️⃣ 複数プロンプトの処理

```python
prompts = ["What is AI?", "Explain machine learning", "Deep learning basics"]
results = client.batch_generate(prompts, max_length=100)

for prompt, result in zip(prompts, results):
    print(f"{prompt}: {result.get('generated_text', 'Failed')}")
```

### 3️⃣ 画像を含むマルチモーダル処理

```python
response = client.submit_prompt(
    prompt="この画像に写っているものは何ですか？",
    image_path="./image.jpg"
)
result = client.poll_for_result(timeout=60)
print(result['response'])
```

### 4️⃣ エラーハンドリング付き実行

```python
# プロンプト検証
if client.validate_prompt("テストプロンプト"):
    # 自動リトライ付きで実行
    result = client.with_error_handling(
        client.generate,
        prompt="テストプロンプト",
        max_length=100,
        retries=3
    )
    if result:
        print("成功:", result['generated_text'])
```

### 5️⃣ 非同期タスクの監視

```python
# タスク送信
client.submit_prompt(prompt="複雑なタスク")

# 結果をポーリング
result = client.poll_for_result(timeout=120, interval=5)

if result:
    print("結果:", result['response'])
else:
    print("タイムアウト")
```

### 6️⃣ モデル保存と復元

```python
import torch
import io

# セッション作成
session = client.create_log_session("モデル訓練実験")
session_id = session['session_id']

# モデル保存
model_buffer = io.BytesIO()
torch.save(model.state_dict(), model_buffer)
model_buffer.seek(0)

# アップロード
artifact = client.upload_artifact(
    session_id=session_id,
    artifact_type="model",
    name="model.pth",
    file=model_buffer
)

# ダウンロード
client.download_artifact(
    artifact_id=artifact['artifact_id'],
    destination_path="./downloaded_model.pth"
)
```

### 7️⃣ データセットアップロード

```python
import zipfile
import os

# トレーニングデータディレクトリの準備
data_dir = "./training_data"
os.makedirs(f"{data_dir}/images", exist_ok=True)

# データセットをZIP圧縮
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
    # CSVファイル追加
    zf.write(f"{data_dir}/captions.csv", arcname='captions.csv')
    # 画像ファイル追加
    for root, _, files in os.walk(f"{data_dir}/images"):
        for file in files:
            full_path = os.path.join(root, file)
            archive_name = os.path.join('images', os.path.relpath(full_path, f"{data_dir}/images"))
            zf.write(full_path, arcname=archive_name)

zip_buffer.seek(0)
zip_buffer.name = "training_dataset.zip"

# データセットアップロード
dataset_artifact = client.upload_artifact(
    session_id=session_id,
    artifact_type="dataset",
    name="vision_training_data",
    file=zip_buffer,
    llm_type="SpikingEvoMultiModalLM"
)

print(f"データセットアップロード完了: {dataset_artifact['artifact_id']}")
```

---

## サーバー情報の確認

```python
# サーバーヘルスチェック
is_healthy = client.is_server_healthy()
print(f"サーバーは正常ですか？: {'はい' if is_healthy else 'いいえ'}")

# ステータス監視
status = client.get_simulation_status()
print(f"現在のプロンプトステータス: {status.get('last_prompt_status', 'N/A')}")
print(f"アクティブノード数: {len(status.get('nodes', []))}")
```

---

## よくあるエラーと解決方法

| エラー | 原因 | 解決策 |
|-------|------|------|
| `ConnectionError` | APIサーバーが起動していない | `sudo ./scripts/run_api_server.sh` で起動 |
| `Timeout` | 処理が遅い | `timeout`パラメータを増やす |
| `Invalid prompt` | プロンプトが条件を満たさない | `validate_prompt()`で事前確認 |

---

## 次のステップ

- 📖 [完全なSDKドキュメント](./EvoSpikeNet_SDK.md)を読む
- 📁 [サンプルコード](./docs/sdk/)を確認する
- 🔧 [トラブルシューティング](./EvoSpikeNet_SDK.md#11-トラブルシューティング)を参照する
- 💬 GitHub Issuesで質問する

---

## 高度な機能 (P3実装完了)

### 🔄 遅延監視と最適化

```python
# 遅延統計の取得
latency_stats = client.get_latency_stats()
print(f"平均遅延: {latency_stats['mean']:.2f}ms")
print(f"95パーセンタイル: {latency_stats['p95']:.2f}ms")

# 遅延ターゲットの確認
target_met = client.check_latency_target(500.0)  # 500ms目標
print(f"遅延目標達成: {target_met}")
```

### 💾 スナップショット/復旧

```python
# システムスナップショット作成
snapshot_result = client.create_snapshot(
    snapshot_name="backup_20251212",
    include_models=True,
    include_data=True
)

# スナップショット一覧
snapshots = client.list_snapshots()

# システム復旧
restore_result = client.restore_snapshot(
    snapshot_path="/path/to/snapshot.gz",
    restore_models=True,
    restore_data=True
)
```

### 📊 スケーラビリティテスト

```python
# スケーラビリティテスト実行
test_result = client.run_scalability_test(
    max_nodes=1000,
    test_duration=300.0,
    load_pattern="linear"
)

# リソース使用状況取得
resources = client.get_resource_usage()
print(f"CPU使用率: {resources['cpu_usage']}%")
print(f"メモリ使用量: {resources['memory_usage']}MB")
```

### 🔧 ハードウェア最適化

```python
# ハードウェア最適化（ONNX/量子化など）
optimization_result = client.optimize_model(
    model_type="vision",              # "vision" | "audio"
    optimizations=["onnx", "quantize"]
)

# モデルベンチマーク
benchmark_result = client.benchmark_model(
    model_type="vision",
    num_runs=50
)
```

### 🛡️ 高可用性監視

```python
# 可用性ステータス取得
availability = client.get_availability_status()
print(f"全体可用性: {availability['overall_availability']}%")
print(f"アップタイム: {availability['uptime_percentage']}%")

# ヘルスチェック実行
health_result = client.perform_health_check()

# 可用性統計取得
stats = client.get_availability_stats(time_window="24h")
```

### 🌐 非同期Zenoh通信

```python
# Zenoh通信統計取得
zenoh_stats = client.get_zenoh_stats()
print(f"メッセージ数: {zenoh_stats['messages_sent']}")
print(f"平均遅延: {zenoh_stats['avg_latency']}ms")
```

### ⚖️ 分散コンセンサス

```python
# コンセンサス提案
proposal_result = client.propose_consensus_decision(
    decision_type="resource_allocation",
    payload={"resource": "gpu", "amount": 50},
    priority=1
)

# コンセンサス結果取得
result = client.get_consensus_result(proposal_result['proposal_id'])

# コンセンサス統計
consensus_stats = client.get_consensus_stats()
```

---

## チートシート

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()

# サーバー確認
client.wait_for_server()           # 起動待機
client.is_server_healthy()         # ヘルスチェック

# テキスト生成
client.generate(prompt)            # シンプル生成
client.batch_generate(prompts)     # バッチ処理
client.submit_prompt(prompt)       # 非同期送信
client.poll_for_result()           # 結果待機

# 検証・制御
client.validate_prompt(prompt)     # プロンプト検証
client.with_error_handling(func)   # リトライ付き実行

# ステータス・ログ
client.get_simulation_status()     # ステータス取得
client.get_simulation_result()     # 結果取得
client.get_remote_log()            # ログ取得

# アーティファクト管理
client.create_log_session()        # セッション作成
client.upload_artifact()           # アップロード
client.download_artifact()         # ダウンロード
client.list_artifacts()            # リスト表示
```

## 6️⃣ LLMトレーニングジョブの実行 (新機能)

### Vision Encoderトレーニング

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

client = EvoSpikeNetAPIClient()

# Vision Encoderトレーニングジョブの送信
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
print(f"トレーニングジョブを開始しました: {response['job_id']}")

# ジョブステータスの確認
status = client.get_training_status(response['job_id'])
print(f"ジョブステータス: {status['status']}")
```

### Audio Encoderトレーニング

```python
# Audio Encoderトレーニングジョブの送信
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
print(f"Audioトレーニングジョブを開始しました: {response['job_id']}")
```

### トレーニングジョブの監視

```python
# すべてのトレーニングジョブのリストを取得
jobs = client.list_training_jobs()
for job in jobs:
    print(f"ジョブID: {job['job_id']}, ステータス: {job['status']}, カテゴリ: {job['category']}")

# 特定のジョブの詳細を取得
job_details = client.get_training_job_details("vision_training_job_001")
print(f"ジョブ詳細: {job_details}")
```

### 分散脳ノード対応トレーニング

```python
# 分散脳ノードタイプに応じたトレーニング
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

# Visionノード用のトレーニングジョブ
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
