# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet SDK クイックスタートガイド

**30秒で始めるEvoSpikeNet SDK**

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
- 📁 [サンプルコード](./examples/sdk/)を確認する
- 🔧 [トラブルシューティング](./EvoSpikeNet_SDK.md#11-トラブルシューティング)を参照する
- 💬 GitHub Issuesで質問する

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

Happy coding! 🚀
