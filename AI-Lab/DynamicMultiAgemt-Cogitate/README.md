# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# Cogitate
セキュアで協調的なマルチタスク・マルチエージェントAIシステム

Copyright (c) 2025 Moonlight Technologies Inc. All rights reserved.

# 概要
LangGraphによるオーケストレーション、FastAPIバックエンド、Reactフロントエンド、リアルタイム更新のためのRedis Pub/Sub、状態永続化のためのPostgreSQL、トレーシングのためのJaegerを特徴とする、スケーラブルなAIシステムです。重要なアクションに対する人間参加型ループ（Human-in-the-Loop）とOWASP LLMセキュリティプラクティスを実装しています。

本システムは、管理者とオペレーターの2つの役割を持つユーザーロールシステムを導入しています。管理者はユーザーとシステム設定を管理し、オペレーターは個人のAPIキーを使用してマルチエージェントカンファレンスを実行します。

# 主な実装機能

*   **動的なLLMプロバイダサポート:**
    *   OpenAIへのハードコードされた依存を排除し、Gemini、Grok、Claudeなど複数の主要プロバイダをサポートするようになりました。
    *   各プロバイダの`langchain`インテグレーション（`langchain-google-genai`, `langchain-anthropic`など）は、`requirements.txt`で管理されます。
    *   現在サポートしているプロバイダは、"ChatGPT"、"Gemini"、"Grok"、"Claude"、そして"ローカルモデル" (Ollama経由)です。

*   **ローカルLLMのサポート:**
    *   `HuggingFaceEmbeddings` を使用したローカルRAG（Retrieval-Augmented Generation）機能を実装しました。
    *   Ollama (`ChatOllama`) を介したローカルモデル推論をサポートします。

*   **UI/UXの改善:**
    *   **リアルタイムストリーミング:** LLMからの応答がリアルタイムでストリーミングされ、タイプライターのように表示されるため、体感速度が大幅に向上しました。
    *   エージェント設定とクエリが `localStorage` に保存され、ページをリロードしても状態が維持されます。
    *   エージェントカードが縦並びで表示される問題を修正し、横並びで折り返すレスポンシブなグリッドレイアウトに変更しました。
    *   クエリ入力欄を複数行入力可能な `<textarea>` に変更しました。
    *   「会議を開始」ボタンのロジックを改善し、より直感的に利用できるようになりました。

*   **セキュリティ:**
    *   プロンプトインジェクションを検出するセキュリティ機能は、`INSPECTOR_LLM_API_KEY` 環境変数を設定することで有効化できるオプション機能になりました。
    *   ローカルの `multiagent.db` データベースファイルが `.gitignore` によってバージョン管理から除外されるようになりました。

*   **高度な協調ワークフロー:**
    *   エージェントの協調プロセスを多段階に拡張しました。
    *   **ステージ1（ブレインストーミング）:** 全エージェントが初期クエリに対して回答を生成します。
    *   **ステージ2（深掘り）:** 管理者は、初期回答をレビューした後、各エージェントに「リサーチ」「反証」「文献検索」などの専門的な役割を割り当て、さらに深掘りのための第二のクエリを投入します。これにより、より構造化された議論が可能になります。

*   **バグ修正:**
    *   バックエンドでPydanticモデルをJSONにシリアライズする際の `TypeError` を修正し、WebSocket通信が正しく行われるようになりました。
    *   開発環境におけるWebSocketプロキシ設定を修正しました。
    *   エージェントの権限チェックがエージェント名ではなく、LLMプロバイダに基づいて正しく行われるようになりました。

# セットアップ手順

## 1. 前提条件

**必要なツール：**

*   Docker（24.0.5以上）
*   Docker Compose（2.20.2以上）
*   Node.js（18.x）
*   Python（3.11）
*   Kubernetes（kubectl 1.28以上、minikubeまたはクラウドプロバイダのクラスタ）

**APIキー：**

*   xAI APIキー：xAI Console（[https://console.x.ai/team/default/api-keys]）から取得。
*   Google APIキー：Google Cloud Consoleから取得。

**環境：**
Linux/macOS/Windows（WSL2推奨）。

## 2. 環境変数の設定

`backend/.env`ファイルを作成します。`ENCRYPTION_KEY`は、オペレーターのAPIキーを暗号化するために使用される32バイトのキーです。

**`ENCRYPTION_KEY`の生成方法:**
```python
# Pythonインタープリタで以下を実行してキーを生成
from cryptography.fernet import Fernet
Fernet.generate_key()
```

**`.env`ファイルの内容:**
```
# backend/.env
# OPENAI_API_KEY=your_openai_api_key  # 不要になりました
XAI_API_KEY=your_xai_api_key
GOOGLE_API_KEY=your_google_api_key
REDIS_URL=redis://redis:6379/0
JWT_SECRET_KEY=a_very_secret_key_that_is_long_and_secure
ENCRYPTION_KEY=your_generated_32_byte_encryption_key
JAEGER_HOST=jaeger:4317
# ローカル開発用にホストポートマッピング経由でデータベースに接続
DB_URL=postgresql://user:password@localhost:5433/cogitatedb
# オプションのセキュリティ機能用
# INSPECTOR_LLM_API_KEY=your_openai_api_key
```

**環境変数をエクスポート（オプション）：**
```bash
export $(cat backend/.env | xargs)
```

### LLMプロバイダの設定

本アプリケーションは複数のLLMプロバイダをサポートしています。各プロバイダの要件は以下の通りです。

*   **ChatGPT (`chatgpt`)**:
    *   `langchain-openai` パッケージが必要です。`pip install langchain-openai`でインストールできます。
    *   UIから追加できる有効なOpenAIのAPIキーが必要です。

*   **Gemini (`gemini`)**:
    *   `langchain-google-genai` パッケージが必要です。
    *   UIから有効なGoogle APIキーを追加する必要があります。

*   **Grok (`grok`)**:
    *   `xai-sdk` パッケージが必要です。
    *   UIから有効なxAI APIキーを追加する必要があります。

*   **Claude (`claude`)**:
    *   `langchain-anthropic` パッケージが必要です。
    *   UIから有効なAnthropic APIキーを追加する必要があります。

*   **ローカルモデル (`local`)**:
    *   このオプションは [Ollama](https://ollama.com/) によって提供されるローカルモデルを使用します。
    *   `langchain-community` と `ollama` パッケージが必要です。
    *   お使いのマシンでOllamaサービスが実行されている必要があります。アプリケーションは `http://ollama:11434` で接続を試みます。
    *   APIキーは不要です。

*   **セキュリティインスペクター (オプション)**:
    *   オプションのプロンプトインジェクションセキュリティ機能にはOpenAIのAPIキーが必要です。
    *   有効にするには、`.env`ファイルで `INSPECTOR_LLM_API_KEY` 環境変数を設定してください。

## 3. 依存関係のインストール

**バックエンド：**
```bash
cd backend
pip install -r requirements.txt
```

**フロントエンド：**
```bash
cd frontend
npm install
```

## 4. ローカルでの実行

Docker Composeで全てのサービスを起動します。`--profile "*"`フラグは、`backend`と`frontend`を含む全ての定義済みサービスを起動するために必要です。
```bash
sudo docker compose --profile "*" up --build
```
**サービスの確認：**

*   フロントエンド： `http://localhost:3000`
*   バックエンド： `http://localhost:8000/docs` （FastAPI Swagger UI）
*   Jaeger： `http://localhost:16686` （トレース確認）

**アプリケーションの利用:**

1.  フロントエンド（`http://localhost:3000`）にアクセスします。
2.  **Register** ページで新しいユーザーアカウントを作成します。最初のユーザーは自動的に **管理者(administrator)** ロールになります。
3.  作成したアカウントで **Login** します。
4.  **オペレーター(operator)** としてログインした場合、UIからLLMのAPIキーを設定・保存できます。
5.  メインの **Meeting View** で、エージェントを追加・設定し、クエリを入力してカンファレンスを開始します。

## 5. 動作検証

**ローカルテスト：**

`pytest`を使用して、コアロジックの単体テストと統合テストを実行します。
```bash
# backend ディレクトリから実行
python -m pytest
```

**LLMプロバイダ接続テスト:**

サポートされている各LLMプロバイダへの接続を確認するためのスクリプトが用意されています。

*   **前提条件:** テストしたいプロバイダのAPIキーを `backend/.env` ファイルに設定する必要があります。
    *   `OPENAI_API_KEY`: ChatGPT用
    *   `GOOGLE_API_KEY`: Gemini用
    *   `XAI_API_KEY`: Grok用
    *   `ANTHROPIC_API_KEY`: Claude用
*   **実行:**
    ```bash
    # backend ディレクトリから実行
    python scripts/test_llm_providers.py
    ```
```bash
# frontend ディレクトリから実行
npm test
```

## 6. 本番環境への移行

**Kubernetesデプロイメント**

シークレットを作成します。
```bash
kubectl create secret generic app-secrets \
--from-literal=XAI_API_KEY=your_xai_api_key \
--from-literal=GOOGLE_API_KEY=your_google_api_key \
--from-literal=JWT_SECRET_KEY=your_kubernetes_secret_key \
--from-literal=ENCRYPTION_KEY=your_kubernetes_encryption_key
```

## 7. セキュリティ対策
*   **OWASP LLM01 (プロンプトインジェクション):** LLMを使用してプロンプトインジェクションを検出する `inspect_input` 機能は、オプション機能として利用可能です。有効にするには `INSPECTOR_LLM_API_KEY` を設定してください。
*   **OWASP LLM02 (不正な出力の処理):** `sanitize_output`関数により、XSS/RCEを防ぎます。
*   **OWASP LLM06 (機密情報の漏洩):** `human_approval`ノードにより、重要なステップで人間の介入を必須とします。
*   **APIキー管理:** ユーザー固有のAPIキーは、AES-256-GCM暗号化（`cryptography.fernet`経由）を使用してデータベースに保存されます。キーの暗号化/復号は`ENCRYPTION_KEY`に依存します。本番環境では、このキーをKubernetes Secretsなどで安全に管理してください。
