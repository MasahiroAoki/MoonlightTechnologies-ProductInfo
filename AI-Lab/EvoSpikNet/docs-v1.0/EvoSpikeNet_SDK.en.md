# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki

# EvoSpikeNet Python SDK ドキュメント

**最終更新日:** 2025年12月2日

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

# カスタムURL
client = EvoSpikeNetAPIClient(base_url="http://your-api-server:8000")
```

### 3.2. ヘルスチェック

#### `is_server_healthy() -> bool`
APIサーバーが正常に稼働しているかを確認します。

**例:**
```python
client = EvoSpikeNetAPIClient()

if client.is_server_healthy():
    print("✅ APIサーバーは正常に稼働しています")
else:
    print("❌ APIサーバーに接続できません")
```

---

## 4. テキスト生成

### 4.1. 基本的なテキスト生成

#### `generate(prompt: str, max_length: int = 50) -> Dict[str, str]`
標準的なテキスト生成エンドポイントを呼び出します。

**パラメータ:**
- `prompt` (str): テキストプロンプト
- `max_length` (int): 生成する最大トークン数（デフォルト: 50）

**戻り値:** 
生成されたテキストを含む辞書

**例:**
```python
client = EvoSpikeNetAPIClient()

# シンプルなテキスト生成
result = client.generate("人工知能とは", max_length=100)
print(f"生成されたテキスト: {result.get('generated_text', '')}")
```

---

## 5. 分散脳シミュレーション

分散脳シミュレーションと対話するための包括的な機能を提供します。

### 5.1. マルチモーダルプロンプトの送信

#### `submit_prompt(prompt: str = None, image_path: str = None, audio_path: str = None) -> Dict`
シミュレーションにマルチモーダルなプロンプトを送信します。

**パラメータ:**
- `prompt` (str, optional): テキストプロンプト
- `image_path` (str, optional): 画像ファイルへのパス
- `audio_path` (str, optional): 音声ファイルへのパス

**注意:** 少なくとも1つのモダリティを提供する必要があります。

**例1: テキストのみ**
```python
client = EvoSpikeNetAPIClient()

# テキストプロンプトを送信
response = client.submit_prompt(prompt="日本の首都はどこですか？")
print(f"プロンプト送信結果: {response}")
```

**例2: テキスト + 画像**
```python
# テキストと画像を組み合わせて送信
response = client.submit_prompt(
    prompt="この画像に写っているものは何ですか？",
    image_path="./examples/sample_image.jpg"
)
print(f"マルチモーダルプロンプト送信成功: {response}")
```

**例3: すべてのモダリティ**
```python
# テキスト、画像、音声を組み合わせて送信
response = client.submit_prompt(
    prompt="音声と画像を分析してください",
    image_path="./data/image.png",
    audio_path="./data/audio.wav"
)
```

### 5.2. シミュレーション状態の監視

#### `get_simulation_status() -> Dict`
現在のシミュレーションのステータスを取得します。

**戻り値:**
全ノードのステータス情報を含む辞書
- `nodes`: 各ノードの情報（ID、ラベル、状態など）
- `last_prompt_status`: 最後のプロンプトの処理状態
- その他のメタ情報

**例:**
```python
status = client.get_simulation_status()

print(f"プロンプトステータス: {status.get('last_prompt_status', 'N/A')}")
print(f"アクティブノード数: {len(status.get('nodes', []))}")

# 各ノードの詳細を表示
for node in status.get('nodes', []):
    print(f"  - {node.get('label')}: {node.get('status', 'unknown')}")
```

### 5.3. シミュレーション結果の取得

#### `get_simulation_result() -> Dict`
完了したクエリの最新の結果を取得します。

**戻り値:**
```python
{
    "response": "生成されたテキスト応答",
    "timestamp": 1234567890.123
}
```

結果が利用可能でない場合は `{"response": None}` を返します。

**例:**
```python
result = client.get_simulation_result()

if result.get("response"):
    print(f"シミュレーション応答: {result['response']}")
    print(f"タイムスタンプ: {result.get('timestamp', '')}")
else:
    print("まだ結果がありません")
```

### 5.4. 結果のポーリング

#### `poll_for_result(timeout: int = 120, interval: int = 5) -> Optional[Dict]`
結果が利用可能になるまで、結果エンドポイントを定期的にポーリングします。

**パラメータ:**
- `timeout` (int): タイムアウトまでの最大秒数（デフォルト: 120）
- `interval` (int): ポーリング間隔（秒）（デフォルト: 5）

**戻り値:**
結果が見つかった場合はその内容、タイムアウトした場合は`None`

**例:**
```python
client = EvoSpikeNetAPIClient()

# プロンプトを送信
client.submit_prompt(prompt="AIの未来について教えてください")

# 結果を待機（最大2分）
print("結果を待っています...")
result = client.poll_for_result(timeout=120, interval=5)

if result:
    print(f"✅ 応答: {result['response']}")
else:
    print("❌ タイムアウト: 結果を取得できませんでした")
```

---

## 6. データロギングとアーティファクト管理

実験の再現性とデータ管理を容易にするための堅牢な機能を提供します。

### 6.1. セッションの作成

#### `create_log_session(description: str) -> Dict`
新しい実験セッションを開始し、一意なセッションIDを取得します。

**パラメータ:**
- `description` (str): このセッションの目的や内容に関する説明

**戻り値:**
セッション情報を含む辞書（`session_id`を含む）

**例:**
```python
session = client.create_log_session(
    description="SNNモデルのハイパーパラメータ調整実験"
)
session_id = session['session_id']
print(f"✅ セッションID: {session_id}")
```

### 6.2. アーティファクトのアップロード

#### `upload_artifact(session_id: str, artifact_type: str, name: str, file: io.BytesIO) -> Dict`
モデル、データセット、設定ファイルなどのデータアーティファクトを、指定したセッションに関連付けてアップロードします。

**パラメータ:**
- `session_id` (str): アーティファクトを関連付けるセッションのID
- `artifact_type` (str): アーティファクトの種類（例: `model`, `config`, `simulation_data`）
- `name` (str): アーティファクトのファイル名
- `file` (io.BytesIO): アップロードするファイルオブジェクト

**例:**
```python
import io
import torch

# モデルの保存
model_buffer = io.BytesIO()
torch.save(model.state_dict(), model_buffer)
model_buffer.seek(0)
model_buffer.name = 'model.pth'

# アップロード
result = client.upload_artifact(
    session_id=session_id,
    artifact_type="model",
    name="spiking_lm_v1.pth",
    file=model_buffer
)
print(f"✅ アーティファクトID: {result['artifact_id']}")
```

### 6.3. アーティファクトのリスト化

#### `list_artifacts(artifact_type: str = None) -> List[Dict]`
データベースに保存されているすべてのアーティファクトのリストを取得します。

**パラメータ:**
- `artifact_type` (str, optional): フィルタリングするアーティファクトの種類

**例:**
```python
# すべてのアーティファクトを取得
all_artifacts = client.list_artifacts()

# モデルのみを取得
models = client.list_artifacts(artifact_type="model")

for artifact in models:
    print(f"ID: {artifact['artifact_id']}")
    print(f"名前: {artifact['name']}")
    print(f"作成日時: {artifact['created_at']}")
    print("---")
```

### 6.4. アーティファクトのダウンロード

#### `download_artifact(artifact_id: str, destination_path: str)`
指定したアーティファクトIDのファイルをダウンロードします。

**パラメータ:**
- `artifact_id` (str): ダウンロードするアーティファクトの一意なID
- `destination_path` (str): ファイルを保存するローカルパス

**例:**
```python
# 最新のモデルをダウンロード
models = client.list_artifacts(artifact_type="model")
if models:
    latest_model = models[0]
    client.download_artifact(
        artifact_id=latest_model['artifact_id'],
        destination_path="./downloaded_model.pth"
    )
    print("✅ モデルをダウンロードしました")
```

---

## 7. リモートログの取得

マルチPC分散シミュレーションにおいて、リモートマシンのログファイルを取得します。

#### `get_remote_log(user: str, ip: str, key_path: str, log_file_path: str) -> Dict`

**パラメータ:**
- `user` (str): SSH username
- `ip` (str): リモートホストのIPアドレス
- `key_path` (str): SSH秘密鍵へのローカルパス
- `log_file_path` (str): リモートホスト上のログファイルの絶対パス

**例:**
```python
log_content = client.get_remote_log(
    user="ubuntu",
    ip="192.168.1.100",
    key_path="/home/user/.ssh/id_rsa",
    log_file_path="/home/appuser/app/simulation_rank1.log"
)
print(f"ログ内容:\n{log_content.get('log_content', '')}")
```

---

## 8. 総合的な使用例

### 8.1. テキストクエリの実行

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
import time

def simple_text_query():
    """シンプルなテキストクエリの例"""
    client = EvoSpikeNetAPIClient()
    
    # 1. ヘルスチェック
    if not client.is_server_healthy():
        print("❌ APIサーバーが応答しません")
        return
    
    # 2. プロンプト送信
    prompt = "人工知能の未来について教えてください"
    print(f"📤 プロンプト送信: {prompt}")
    client.submit_prompt(prompt=prompt)
    
    # 3. 結果を待機
    print("⏳ 処理を待っています...")
    result = client.poll_for_result(timeout=60, interval=3)
    
    # 4. 結果表示
    if result and result.get('response'):
        print(f"\n✅ 応答:\n{result['response']}\n")
    else:
        print("❌ 応答を取得できませんでした")

if __name__ == "__main__":
    simple_text_query()
```

### 8.2. マルチモーダルクエリの実行

```python
from evospikenet.sdk import EvoSpikeNetAPIClient

def multimodal_query():
    """画像を含むマルチモーダルクエリの例"""
    client = EvoSpikeNetAPIClient()
    
    # 画像とテキストを組み合わせたクエリ
    response = client.submit_prompt(
        prompt="この画像に写っている物体を説明してください",
        image_path="./data/sample_image.png"
    )
    
    print("プロンプト送信完了")
    
    # 結果を待機
    result = client.poll_for_result(timeout=120)
    
    if result:
        print(f"視覚処理結果: {result['response']}")

if __name__ == "__main__":
    multimodal_query()
```

### 8.3. モデル訓練とアーティファクト管理の完全ワークフロー

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
from evospikenet.models import SpikingEvoSpikeNetLM
import torch
import json
import io

def complete_ml_workflow():
    """モデル訓練からアーティファクト保存までの完全なワークフロー"""
    
    client = EvoSpikeNetAPIClient()
    
    # --- ステップ1: セッション作成 ---
    print("\n=== ステップ1: 新しいセッションを作成 ===")
    session = client.create_log_session(
        description="SpikingEvoSpikeNetLMの訓練実験"
    )
    session_id = session['session_id']
    print(f"✅ セッションID: {session_id}")
    
    # --- ステップ2: モデル訓練（ダミー） ---
    print("\n=== ステップ2: モデル訓練 ===")
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 4,
        'num_transformer_blocks': 2,
        'time_steps': 10
    }
    
    model = SpikingEvoSpikeNetLM(**config)
    print("✅ モデル初期化完了")
    
    # 実際にはここで訓練を実行
    # model.train()...
    
    # --- ステップ3: モデルと設定を保存 ---
    print("\n=== ステップ3: アーティファクトのアップロード ===")
    
    # モデルの保存
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
    print(f"✅ モデルアップロード完了: {model_artifact['artifact_id']}")
    
    # 設定ファイルの保存
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
    print(f"✅ 設定ファイルアップロード完了: {config_artifact['artifact_id']}")
    
    # --- ステップ4: アーティファクトの確認 ---
    print("\n=== ステップ4: アーティファクトリストの確認 ===")
    artifacts = client.list_artifacts()
    print(f"保存されているアーティファクト数: {len(artifacts)}")
    for artifact in artifacts[-5:]:  # 最新5件を表示
        print(f"  - {artifact['name']} ({artifact['artifact_type']})")
    
    # --- ステップ5: モデルのダウンロードと復元 ---
    print("\n=== ステップ5: モデルのダウンロードと復元 ===")
    models = client.list_artifacts(artifact_type="model")
    if models:
        latest_model_id = models[0]['artifact_id']
        client.download_artifact(
            artifact_id=latest_model_id,
            destination_path="./restored_model.pth"
        )
        
        # モデルの復元
        restored_model = SpikingEvoSpikeNetLM(**config)
        restored_model.load_state_dict(torch.load("./restored_model.pth"))
        print("✅ モデルを復元しました")
        
        # クリーンアップ
        import os
        os.remove("./restored_model.pth")
    
    print("\n" + "="*50)
    print("ワークフロー完了！")

if __name__ == "__main__":
    complete_ml_workflow()
```

### 8.4. 状態監視とリアルタイムフィードバック

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
import time

def monitor_simulation():
    """シミュレーション状態をリアルタイムで監視"""
    
    client = EvoSpikeNetAPIClient()
    
    # プロンプト送信
    client.submit_prompt(prompt="複雑な計算タスク")
    
    print("シミュレーション監視を開始...")
    print("="*60)
    
    # 最大60秒間、3秒ごとに状態をチェック
    max_time = 60
    interval = 3
    elapsed = 0
    
    while elapsed < max_time:
        status = client.get_simulation_status()
        prompt_status = status.get('last_prompt_status', 'Unknown')
        
        print(f"[{elapsed}s] ステータス: {prompt_status}")
        
        # 各ノードの情報を表示
        for node in status.get('nodes', [])[:5]:  # 最初の5ノードのみ
            label = node.get('label', 'N/A')
            node_status = node.get('status', 'N/A')
            print(f"  → {label}: {node_status}")
        
        # 完了チェック
        if 'completed' in prompt_status.lower() or 'idle' in prompt_status.lower():
            print("\n✅ シミュレーション完了！")
            
            # 結果を取得
            result = client.get_simulation_result()
            if result and result.get('response'):
                print(f"\n応答: {result['response']}")
            break
        
        time.sleep(interval)
        elapsed += interval
        print("-"*60)
    
    if elapsed >= max_time:
        print("\n⚠️ タイムアウト")

if __name__ == "__main__":
    monitor_simulation()
```

---

## 9. 便利なヘルパーメソッド

### 9.1. バッチテキスト生成

#### `batch_generate(prompts: List[str], max_length: int = 50) -> List[Dict]`
複数のプロンプトを順序どおりに処理し、結果をリストで返します。各プロンプトのエラーは個別に処理されます。

**パラメータ:**
- `prompts` (List[str]): 処理するプロンプトのリスト
- `max_length` (int): 生成する最大トークン数（デフォルト: 50）

**戻り値:**
各プロンプトの結果を含むリスト

**例:**
```python
client = EvoSpikeNetAPIClient()

prompts = [
    "人工知能とは",
    "機械学習の応用例",
    "ニューラルネットワークの仕組み"
]

results = client.batch_generate(prompts, max_length=100)

for prompt, result in zip(prompts, results):
    if result.get('generated_text'):
        print(f"✓ {prompt}: {result['generated_text'][:50]}...")
    else:
        print(f"✗ {prompt}: 生成失敗")
```

### 9.2. サーバー情報の取得

#### `get_server_info() -> Optional[Dict]`
サーバーとモデルの情報を取得します。バージョン、利用可能なモデル、サーバーの状態などが含まれます。

**戻り値:**
サーバー情報を含む辞書、またはサーバーが応答しない場合は`None`

**例:**
```python
client = EvoSpikeNetAPIClient()

info = client.get_server_info()
if info:
    print(f"サーバーバージョン: {info.get('version', 'N/A')}")
    print(f"モデル: {info.get('model', 'N/A')}")
    print(f"利用可能: {info.get('available', False)}")
```

### 9.3. サーバーの健全性確認（タイムアウト付き）

#### `wait_for_server(timeout: int = 30, interval: int = 2) -> bool`
サーバーが応答するようになるまで待機します。定期的にヘルスチェックを実行し、タイムアウトまたは成功で戻ります。

**パラメータ:**
- `timeout` (int): 最大待機秒数（デフォルト: 30）
- `interval` (int): チェック間隔（秒）（デフォルト: 2）

**戻り値:**
サーバーが応答する場合は`True`、タイムアウトした場合は`False`

**例:**
```python
client = EvoSpikeNetAPIClient()

print("サーバーを待機中...")
if client.wait_for_server(timeout=60, interval=3):
    print("✓ サーバーが利用可能になりました")
else:
    print("✗ タイムアウト: サーバーが応答しません")
```

### 9.4. プロンプトの検証

#### `validate_prompt(prompt: str) -> bool`
プロンプトが有効か確認します。空でなく、最大長を超えていないことを確認します。

**パラメータ:**
- `prompt` (str): 検証するプロンプト

**戻り値:**
プロンプトが有効な場合は`True`、無効な場合は`False`

**例:**
```python
client = EvoSpikeNetAPIClient()

test_prompts = [
    "有効なプロンプト",
    "",  # 無効: 空
    "x" * 15000  # 無効: 長すぎる
]

for prompt in test_prompts:
    if client.validate_prompt(prompt):
        print(f"✓ 有効: {prompt[:30]}")
    else:
        print(f"✗ 無効: {prompt[:30]}")
```

### 9.5. エラーハンドリングとリトライ

#### `with_error_handling(func: Callable, retries: int = 3, *args, **kwargs) -> Optional[Any]`
関数呼び出しにエラーハンドリングとリトライロジックを追加します。指数バックオフを使用して待機します。

**パラメータ:**
- `func` (Callable): 実行する関数
- `retries` (int): リトライ回数（デフォルト: 3）
- `*args, **kwargs`: 関数に渡す引数とキーワード引数

**戻り値:**
関数の戻り値、またはすべてのリトライが失敗した場合は`None`

**例:**
```python
client = EvoSpikeNetAPIClient()

result = client.with_error_handling(
    client.generate,
    retries=3,
    prompt="テストプロンプト",
    max_length=100
)

if result:
    print(f"✓ 成功: {result['generated_text']}")
else:
    print("✗ すべてのリトライが失敗しました")
```

---

## 10. エラーハンドリングとベストプラクティス

### 10.1. 基本的なエラーハンドリング

```python
from evospikenet.sdk import EvoSpikeNetAPIClient
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_api_call():
    """エラーハンドリングを含む堅牢なAPI呼び出し"""
    
    client = EvoSpikeNetAPIClient()
    
    try:
        # APIサーバーのヘルスチェック
        if not client.is_server_healthy():
            raise ConnectionError("APIサーバーが応答しません")
        
        # プロンプト送信
        response = client.submit_prompt(prompt="テストクエリ")
        
        # 結果待機（タイムアウト付き）
        result = client.poll_for_result(timeout=30)
        
        if not result:
            print("⚠️ 結果を取得できませんでした（タイムアウト）")
            return None
        
        return result
        
    except ConnectionError as e:
        print(f"❌ 接続エラー: {e}")
        print("APIサーバーが起動しているか確認してください")
        
    except Timeout:
        print("❌ タイムアウト: APIサーバーの応答が遅すぎます")
        
    except RequestException as e:
        print(f"❌ APIリクエストエラー: {e}")
        
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
    
    return None

if __name__ == "__main__":
    result = robust_api_call()
```

### 10.2. ベストプラクティス

1. **常にヘルスチェックを実行**: API呼び出しの前に`is_server_healthy()`でサーバーの状態を確認

2. **適切なタイムアウトを設定**: `poll_for_result()`では処理の複雑さに応じてタイムアウトを調整

3. **エラーハンドリング**: すべてのAPI呼び出しを`try-except`ブロックで囲む

4. **リソースのクリーンアップ**: ファイルを使用した後は適切に削除

5. **セッション管理**: 関連する実験は同じセッションIDでグループ化

6. **ログ記録**: 重要な操作とエラーは適切にログに記録

---

## 11. トラブルシューティング

### よくある問題と解決方法

**問題1: `is_server_healthy()`が`False`を返す**
```
解決策:
1. APIサーバーが起動しているか確認: docker ps | grep api
2. 正しいURLを指定しているか確認
3. ファイアウォールやネットワーク設定を確認
```

**問題2: `poll_for_result()`がタイムアウトする**
```
解決策:
1. タイムアウト時間を増やす
2. シミュレーションのログを確認してエラーがないかチェック
3. 分散脳シミュレーションが正しく起動しているか確認
```

**問題3: アーティファクトのアップロードが失敗する**
```
解決策:
1. ファイルサイズが大きすぎないか確認
2. セッションIDが有効か確認
3. APIサーバーのディスク容量を確認
```

---

## 12. まとめ

このSDKは、EvoSpikeNetの強力な機能をPythonから簡単に利用できるようにします。基本的なテキスト生成から、複雑なマルチモーダル分散脳シミュレーション、実験管理まで、包括的な機能を提供します。

### 利用可能なサンプルコード

以下のサンプルファイルが`examples/sdk/`ディレクトリに提供されています：

- **simple_generation.py** - 基本的なテキスト生成
- **batch_generation.py** - 複数プロンプトのバッチ処理
- **robust_error_handling.py** - エラーハンドリングとリトライロジック
- **multimodal_generation.py** - テキスト、画像、音声を使用したマルチモーダル処理
- **async_patterns.py** - 非同期タスク管理とポーリングパターン

詳細な情報や最新のアップデートについては、プロジェクトの[GitHub リポジトリ](https://github.com/MasahiroAoki/EvoSpikeNet)を参照してください。
