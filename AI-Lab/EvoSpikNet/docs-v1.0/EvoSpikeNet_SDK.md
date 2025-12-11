# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki

# EvoSpikeNet Python SDK ドキュメント

**最終更新日:** 2025年12月7日

## 1. 概要

`EvoSpikeNet Python SDK`は、`EvoSpikeNet API`と対話するための高レベルなインターフェースを提供するクライアントライブラリです。このSDKを利用することで、開発者はHTTPリクエストの詳細を意識することなく、数行のPythonコードでEvoSpikeNetのテキスト生成、データロギング、分散脳シミュレーション機能を自身のアプリケーションに簡単に統合できます。

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

## 4. テキスト生成

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

#### `poll_for_result(prompt_id: str, timeout: int = 120, interval: int = 5) -> Optional[Dict]`
`prompt_id`を指定して、特定のタスクの結果が利用可能になるまで定期的にポーリングします。

**例:**
```python
# 上記で受け取ったprompt_idを使用
result = client.poll_for_result(prompt_id=prompt_id, timeout=120)

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
model_buffer.seek(0); model_buffer.name = 'spiking_lm.pth'
client.upload_artifact(session_id, "model", "spiking_lm.pth", model_buffer, llm_type="SpikingEvoTextLM")

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

## 9. まとめ

EvoSpikeNet Python SDKを使用することで、数行のコードでAPIの全機能にアクセスできます。主な利点：

✅ **シンプルなインターフェース**: HTTPリクエストの詳細を隠蔽  
✅ **自動リトライ**: 指数バックオフで堅牢性向上  
✅ **マルチモーダル対応**: テキスト・画像・音声の統合処理  
✅ **完全なMLOpsサポート**: セッション管理、アーティファクトアップロード  

詳細なコード例は`examples/sdk_usage.py`を参照してください。
