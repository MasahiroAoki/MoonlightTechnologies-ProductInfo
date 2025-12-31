<!-- 
Reviewed against source: 2025-12-31
Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved.
Author: Masahiro Aoki
-->

# ⚠️ 商用利用される企業様への重要なお知らせ

本リポジトリのコードは MIT License で公開されていますが、**企業による商用製品・商用サービスへの組み込み利用**には、および**当社が保有する特許を実施**して収益を得る場合には、別途「企業向け商用ライセンス契約」の締結が必要です。

対象となる主なケース：
- 自社SaaS・アプリ・サービスに本フレームワークを組み込んで有料提供する場合
- 本フレームワークを使って顧客にAI機能を提供し対価を得る場合
- 社内システムであっても収益に直結する形で大規模利用する場合

→ 該当する企業様は必ず以下までお問い合わせください  
✉️ maoki@moonlight-tech.biz  

個人利用・研究利用・PoC・スタートアップのプロトタイプなどは完全に無料でMITライセンスのままご利用いただけます。

# EvoSpikeNet - 分散脳シミュレーションフレームワーク

**最終更新日:** 2025年12月31日  
**バージョン:** v0.1.2  
**ステータス:** 🟢 Production Ready (Plan C+ Phase 1開始予定)

## 📊 プロジェクトステータス

- ✅ **コア機能**: 完全実装済み（SNN、進化、分散処理、マルチモダリティ）
- ✅ **P3本番機能**: 8機能完全実装（遅延監視、スナップショット、スケーラビリティ、ハードウェア最適化、可用性監視、非同期通信、分散合意形成、データアップロード）
- ✅ **テストインフラ**: 完全実装済み（統合テスト、E2Eテスト、パフォーマンスベンチマーク、90%カバレッジ目標）
- ✅ **型安全性**: 完全実装済み（mypy統合、モダンPythonタイピング、Evolution v2技術的問題解決）
- ✅ **CI/CDパイプライン**: 完全実装済み（GitHub Actions、Dockerビルド、セキュリティスキャン、自動デプロイ）
- ✅ **APIドキュメント**: 自動生成実装済み（OpenAPI 3.0、Swagger UI、ReDoc、Postmanコレクション）
- ✅ **長期記憶（エピソード記憶）**: 実装完了（学習効率30%向上、適応性強化）
- ✅ **長期記憶ノード** ⭐ NEW (2025年12月31日): FAISSベクトル検索とZenoh分散通信を統合した新しい長期記憶システム実装（エピソディック/セマンティック記憶、記憶統合ノード）
- 📋 **実装計画**: Plan C+（現システム改善、2025年12月-2026年6月）→ Plan D（Brain Language、2026年7月-2027年6月）
- 📖 **完全ドキュメント**: [REMAINING_FEATURES.md](docs/REMAINING_FEATURES.md) に統合実装ロードマップを記載
- 🧪 **テストガイド**: [TESTING.md](docs/TESTING.md) に包括的なテストスイートガイドを記載

詳細な実装状況、技術的問題、今後18ヶ月の詳細ロードマップは [docs/REMAINING_FEATURES.md](docs/REMAINING_FEATURES.md) を参照してください。

---

## 1. プロジェクト概要

EvoSpikeNetは、生物学的な脳の機能的専門化と統合の原理に着想を得た、スケーラブルな**分散脳シミュレーションフレームワーク**です。専門化されたニューラルモジュール（視覚、言語、運動など）が個別のプロセスとして動作し、それらを中央の**前頭前野（PFC）モジュール**が動的に調整・統合します。

本フレームワークの最大の特徴は、PFCに実装された**Q-PFCフィードバックループ**です。これは、PFCが自身の意思決定の不確実性（認知エントロピー）を測定し、その値を用いて量子インスパイアード回路をシミュレート、その結果を自身のニューロン活動にフィードバックするという、高度な自己言及的制御メカニズムです。

`torch.distributed`を基盤とし、マルチプロセス/マルチノードでの実行をサポートすることで、単一デバイスの制約を超えた大規模なニューロモーフィックシステムの構築と研究を可能にします。

## 2. Web UIの起動

- **分散脳シミュレーション (Zenohベース)**:
    - **非同期通信**: `Zenoh` publish/subscribeモデルを採用し、`torch.distributed`ベースの旧アーキテクチャから脱却。これにより、堅牢性とスケーラビリティが大幅に向上。
    - **PFCによる認知制御**: `ChronoSpikeAttention`を用いたタスクルーティングと、Q-PFCフィードバックループによる自己変調機能を備えた中央制御ハブ。
    - **階層的機能モジュール**: 視覚、聴覚、言語、運動などの各機能が、親ノードと複数の子ノードからなる階層的な処理パイプラインとして実装。
    - **UIによる対話**: Web UIからテキスト、画像、音声を含むマルチモーダルなプロンプトを送信し、シミュレーションの実行、リアルタイムな状態監視、結果の取得が可能。

- **Q-PFCフィードバックループ**:
    - PFCが自身の認知負荷（エントロピー）に応じて、`QuantumModulationSimulator`を介して自身のワーキングメモリのダイナミクスを動的に調整する、本フレームワークの最も独創的な機能。

- **本格的なSNN言語モデル (`SpikingEvoTextLM` / 旧`SpikingTextLM`)**:
    - `snnTorch`ベースのスパイクTransformer。`TAS-Encoding`や`ChronoSpikeAttention`を備え、`SpikingTextLM`は後方互換のため残存（v2.0で削除予定）。

- **トライモーダル処理能力 (`SpikingEvoMultiModalLM` / 旧`SpikingMultiModalLM`)**:
    - テキスト、画像、音声を統合処理。`SpikingMultiModalLM`は互換レイヤーとして残存（v2.0で削除予定）。

- **ハイブリッド検索RAG**:
    - Milvus（ベクトル検索）とElasticsearch（キーワード検索）を並列で実行し、Reciprocal Rank Fusion (RRF) アルゴリズムで結果を融合することで、高精度な検索拡張生成を実現。
    - **長文対応**: Milvusのスキーマ定義に基づき、最大65,535文字のドキュメントを保存可能。
    - **自動プロンプト切り詰め**: Hugging Faceモデルなどの制約に合わせて、プロンプトを自動的に最適な長さに調整。
    - **対話型データ管理**: チェックボックスによる行選択、インライン編集、リアルタイム文字数カウンターなど、使いやすいUI。
    - **デバッグ可視化** ⭐ NEW (2025年12月17日): クエリ処理の内部プロセス（言語検出、キーワード抽出、ベクトル/キーワード検索結果、RRF融合、生成詳細）をUI上で可視化。
- **プラグインアーキテクチャ** ⭐ NEW (2025年12月20日):
    - **動的プラグインシステム**: 7種類のプラグインタイプ（Neuron、Encoder、Plasticity、Functional、Learning、Monitoring、Communication）をサポート。
    - **entry_points対応**: setuptools entry_pointsによる自動検出とロード機能。
    - **ライフサイクル管理**: initialize → activate → execute → deactivate の明確なライフサイクル。
    - **新機能追加時間70%短縮**: モノリシック構造から脱却し、プラグインとして独立実装可能。
    - **ビルトインプラグイン**: LIF/Izhikevich/EntangledSynchrony（Neuron）、Rate/TAS/Latency（Encoder）、STDP/MetaPlasticity/Homeostasis（Plasticity）を提供。

- **マイクロサービス化** ⭐ NEW (2025年12月20日):
    - **疎結合アーキテクチャ**: Training、Inference、Model Registry、Monitoringの4サービスに分離。
    - **API Gateway**: 統一的なエントリーポイントとルーティング機能（Port 8000）。
    - **スケーラビリティ80%向上**: 各サービスを独立してスケール可能。
    - **障害分離**: 個別サービスの障害が全体に波及しない設計。
    - **Docker Compose対応**: `docker-compose.microservices.yml`で簡単デプロイ。
- **静的解析統合** ⭐ NEW (2025年12月20日):
    - **自動コード品質チェック**: Black、isort、Flake8、Pylint、mypy、Bandit、interrogateの7ツールを統合。
    - **Pre-commitフック**: コミット前に10種類以上の自動品質チェックを実行。
    - **CI/CD自動化**: GitHub ActionsでセキュリティスキャンやDocstring coverage検証を実施。
    - **品質ダッシュボード**: Pylint/Bandit/Flake8の結果をHTML可視化。
    - **開発効率化**: Makefile、セットアップスクリプト、包括的なガイドを提供。
    - **品質目標達成**: Pylint ≥7.0、セキュリティ問題 ≤5、Docstring coverage ≥60%。

- **負荷分散の精緻化** ⭐ NEW (2025年12月20日):
    - **同一モジュールタイプ動的負荷分散**: 5種類の分散戦略(最小応答時間、重み付けラウンドロビン、一貫性ハッシュ、動的容量、キュー長)。
    - **インスタンスプーリング**: モジュールタイプごとに複数インスタンスを管理。
    - **リアルタイムメトリクス監視**: 応答時間、スループット、エラー率の継続的追跡。
    - **適応的容量管理**: 負荷に応じた自動調整と再バランス。
    - **ヘルスベースルーティング**: 健全性チェックと自動フェイルオーバー。
    - **スループット25%向上**: ベンチマークで100→125 req/s、応答時間24%短縮を達成。

- **設定外部化（Configuration Management）** ⭐ NEW (2025年12月20日):
    - **型安全な設定管理**: Pydanticベースの自動バリデーションとIDE補完対応。
    - **多層設定読み込み**: 環境変数 > 環境別YAML > デフォルトYAML > ビルトインデフォルトの優先順位。
    - **ホットリロード**: サーバー再起動不要で設定変更を即座に反映。
    - **6カテゴリ設定**: Database、API、Model、Zenoh、Hardware、Monitoringの包括的設定管理。
    - **環境別設定**: Development/Staging/Production環境の明確な分離。
    - **7APIエンドポイント**: 設定取得、更新、検証、リロード、エクスポート、スキーマ取得。
    - **運用柔軟性90%向上**: 環境構築80%短縮、設定変更95%短縮、設定ミス90%削減。

- **多様なSNNコアエンジン**:
    - 計算効率に優れた`LIFNeuronLayer`、生物学的妥当性の高い`IzhikevichNeuronLayer`、量子インスパイアードの`EntangledSynchronyLayer`など、複数のニューロンモデルをサポート。
    - 視覚・音声エンコーダは`SpikingEvoVisionEncoder`/`SpikingEvoAudioEncoder`が推奨（旧`SpikingVisionEncoder`/`SpikingAudioEncoder`は非推奨、v2.0で削除予定）。

- **フェデレーテッド学習 (Flower)**:
    - `Flower`フレームワークを統合し、プライバシーを保護しながら分散環境でモデルを協調学習させる機能をサポート。

- **RESTful APIとPython SDK**:
    - `FastAPI`ベースのAPIが、テキスト生成、データロギング、分散脳シミュレーションの制御など、フレームワークの全機能へのプログラムアクセスを提供。
    - **型安全なPython SDK** ⭐ NEW (2025年12月17日): `EvoSpikeNetAPIClient`の完全な型ヒント対応、`Enum`による定数定義、`dataclass`によるレスポンス構造化、詳細なエラー情報（`ErrorInfo`）、自動リトライ機能（指数バックオフ）、接続プーリング、統計トラッキング機能を実装。開発時エラー80%削減。
    - **Jupyter統合** ⭐ NEW (2025年12月17日): `JupyterAPIClient`によるインタラクティブなAPI操作、リッチHTML出力、マジックコマンド（`%evospikenet_connect`, `%%evospikenet_generate`）、複数表示モード（HTML/JSON/テキスト）、統計可視化機能。開発効率40%向上。
    - **バリデーション・テストツール** ⭐ NEW (2025年12月17日): `APIValidator`によるエンドポイント検証、パフォーマンスベンチマーク（`PerformanceMetrics`）、ロードテスト、結果エクスポート機能を提供。
    - **コンテナ間通信の最適化**: ファイルベースからAPI経由の通信に変更し、Dockerコンテナ間での信頼性を向上。

- **統合Web UI**:
    - データ生成、モデル訓練、推論、結果分析、システム管理など、フレームワークの全機能をブラウザからインタラクティブに操作できるDashベースのマルチページアプリケーション。
    - **リアルタイム状態監視**: 分散脳シミュレーションの各ノードの状態、エネルギー、スパイク活動をリアルタイムで可視化。
    - **マルチモーダルクエリ**: テキスト、画像、音声を組み合わせた複雑なクエリに対応。
    - **モデル分類管理** ⭐ NEW (2025年12月17日): モデルアップロード時に脳ノードタイプ（Vision, Motor, Auditory, Speech, Executive, General）、モデルカテゴリ（20+種類）、モデルバリアント（Lightweight, Standard, High Accuracy, Realtime, Experimental）を選択可能。分散脳アーキテクチャに沿った体系的なモデル管理を実現。

- **Embodied AI機能 (P2実装)**:
    - **知覚の言語化**: 視覚・聴覚特徴を自然言語に変換 (`PerceptionToTextConverter`)
    - **運動指示生成**: 言語コマンドから運動ゴールを生成 (`TextToMotorConverter`)
    - **クローズドループ制御**: センサー入力に基づくリアルタイム適応 (`ClosedLoopController`)
    - **自動フェイルオーバー**: Raftコンセンサスによる5秒以内の高可用性フェイルオーバー
    - **負荷分散**: タスクと読み取り操作のインテリジェント分散 (`LoadBalancer`)

- **長期記憶ノード** ⭐ NEW (2025年12月31日):
    - **エピソディック記憶（`EpisodicMemoryNode`）**: 時系列イベントの保存と検索。体験や出来事を時間順に管理し、コンテキストに基づく想起が可能。
    - **セマンティック記憶（`SemanticMemoryNode`）**: 概念・知識の保存。関連概念とのリンク付け、重要度に基づく自動管理。
    - **記憶統合ノード（`MemoryIntegratorNode`）**: エピソディック/セマンティック記憶のクロスモーダル連想と統合。高次認知機能の実現。
    - **FAISS高速ベクトル検索**: コサイン類似度による類似記憶検索（数ミリ秒）、1000+ queries/sec のスループット。
    - **Zenoh分散通信統合**: ノード間リアルタイム記憶共有、PTP同期タイムスタンプ、自動整理機能。
    - **パフォーマンス**: 保存レイテンシ <10ms、検索レイテンシ <5ms、最大10,000エントリ対応。

## 2. Web UIの起動

EvoSpikeNetのWeb UIを起動するには、複数の方法があります。用途に応じて適切な起動オプションを選択してください。すべての方法で、起動後ブラウザで `http://localhost:8050` にアクセスできます。

### 2.1. 推奨起動方法（Docker Compose）

Docker Composeを使用した起動が最も推奨されます。必要なサービスのみを起動し、リソースを効率的に使用できます。

#### デフォルト起動（Frontend + 基本サービス）
Frontendとその依存関係（API, PostgreSQL, Milvus, Elasticsearch, Zenoh Router）を起動します。開発・デモ用途に最適です。

```bash
docker-compose up
```

**起動されるサービス:**
- `frontend`: Web UI (ポート8050)
- `api`: FastAPIバックエンド (ポート8000)
- `postgres`: データベース
- `milvus-standalone`: ベクトルデータベース
- `elasticsearch`: 検索エンジン
- `zenoh-router`: 分散通信ルーター

#### 全サービス起動
開発、テスト、本番環境向けにすべてのサービスを起動します。

```bash
docker-compose --profile full up
```

**追加で起動されるサービス:**
- `dev`: 開発用コンテナ (ポート8052)
- `test`: テスト実行環境
- `mkdocs`: ドキュメントサーバー (ポート8001)
- `llm-training-api`: LLMトレーニングAPI (ポート8000 ※要設定)

#### 個別サービス起動
特定のサービスのみを起動する場合：

```bash
# Frontendのみ
docker-compose up frontend

# APIのみ
docker-compose up api

# ドキュメントのみ
docker-compose --profile full up mkdocs

# LLMトレーニングAPIのみ
docker-compose --profile full up llm-training-api
```

### 2.2. シェルスクリプト起動（レガシー）

従来のシェルスクリプトを使用した起動方法です。GPU/CPU環境に応じて自動設定されます。

#### GPU環境
```bash
sudo ./scripts/run_frontend_gpu.sh
```

## **APIキー設定（トレーニングスクリプト・アーティファクトアップロード用）**

- トレーニングスクリプトは `EVOSPIKENET_API_KEY` を優先して使用します。
    - 明示的に単一キーを渡す場合は `EVOSPIKENET_API_KEY` を設定してください。
    - 複数キーをカンマ区切りで渡す既存の環境変数 `EVOSPIKENET_API_KEYS` がある場合は、先頭のキーをフォールバックとして使用します（互換性のため）。

例（`.env`）:

```env
# 単一キー推奨
EVOSPIKENET_API_KEY=4b6f2a1c8e9d4f3a9b0c7d6e5f1a2b3c

# 互換: 複数キー（カンマ区切り）
EVOSPIKENET_API_KEYS=4b6f2a1c8e9d4f3a9b0c7d6e5f1a2b3c,9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f
```

例（`docker-compose.yml` の `dev` / `api` 両コンテナに同じキーを設定）:

```yaml
services:
    api:
        environment:
            - EVOSPIKENET_API_KEY=${EVOSPIKENET_API_KEY:-test-api-key}
    dev:
        environment:
            - EVOSPIKENET_API_KEY=${EVOSPIKENET_API_KEY:-test-api-key}
```

トレーニングスクリプト実行例（`--upload-to-db` を付けると学習後にアーティファクトをAPIへアップロードします）:

```bash
docker exec evospikenet-dev python examples/train_spiking_evospikenet_lm.py --upload-to-db
```

注意:
- 本番環境では `EVOSPIKENET_API_KEY` を秘密管理（Vault, AWS Secrets Manager など）で扱ってください。
- SDK はアップロード時にクライアントのセッションヘッダ（`X-API-Key`）を用いて認証します。環境変数が一致していることを確認してください。


#### CPU環境
```bash
sudo ./scripts/run_frontend_cpu.sh
```

**注意:** これらのスクリプトは全サービスを起動するため、リソース消費が大きいです。開発時はDocker Composeのデフォルト起動を推奨します。

### 2.3. 環境変数設定

起動時の動作をカスタマイズするための環境変数です。

#### 基本設定
```bash
# デバイス設定 (cpu/gpu)
export DEVICE=cpu

# データベース設定
export DATABASE_URL=postgresql://user:password@host:port/db

# API設定
export EVOSPIKENET_API_KEYS=your_api_key
export EVOSPIKENET_ALLOW_NO_AUTH=false

# 分散脳設定
export ACTIVE_RANKS=0,1,2,3,4,5,6
```

#### Docker Composeでの使用例
```bash
DEVICE=gpu ACTIVE_RANKS=0,1,2,3,4,5,6,7,8 docker-compose up
```

### 2.4. マイクロサービス起動

大規模展開向けのマイクロサービスアーキテクチャを使用する場合：

```bash
docker-compose -f docker-compose.microservices.yml up
```

**起動されるサービス:**
- `gateway`: API Gateway (ポート8000)
- `training`: トレーニングサービス (ポート8001)
- `inference`: 推論サービス (ポート8002)
- `model-registry`: モデルレジストリ (ポート8003)
- `monitoring`: 監視サービス (ポート8004)

### 2.5. トレーニング専用起動

LLMトレーニングに特化した環境を起動する場合：

```bash
# CPUトレーニング
docker-compose -f docker-compose.train.yml up

# GPUトレーニング
docker-compose -f docker-compose.gpu.yml up
```

### 2.6. トラブルシューティング

#### ポート競合
他のサービスがポートを使用している場合：
```bash
# ポート変更例
docker-compose up --scale frontend=1 -p 8051:8050
```

#### ビルドキャッシュクリア
イメージの再ビルドが必要な場合：
```bash
docker-compose build --no-cache
docker-compose up
```

#### ログ確認
起動時の詳細ログを確認：
```bash
docker-compose up --verbose
```

#### サービス状態確認
```bash
docker-compose ps
```

#### ログ表示
```bash
# 全サービスのログ
docker-compose logs

# 特定サービスのログ
docker-compose logs frontend
```

### 2.7. 停止方法

```bash
# サービス停止
docker-compose down

# ボリュームも削除
docker-compose down -v

# マイクロサービス停止
docker-compose -f docker-compose.microservices.yml down
```

起動後、ブラウザで `http://localhost:8050` にアクセスしてください。

## 2.5. 統合テストシステムの実行 ⭐ NEW

EvoSpikeNetの全機能を統合的にテストできるメニューシステムを提供しています。8つのテストカテゴリから選択して実行可能です。

### インタラクティブテストメニュー起動

```bash
cd /path/to/EvoSpikeNet
python3 tests/unit/test_menu.py
```

### 統合テストオプション

メニューから以下の統合テストオプションを選択できます：

- **🚀 統合テスト実行 (オプション10)**: 全テストスイートを並列実行
- **📊 テストレポート表示 (オプション11)**: 最新のテスト結果を表示
- **⚙️ 設定変更 (オプション12)**: 並列実行数などの設定を変更

### テストカテゴリ

- 🚀 **パフォーマンス最適化**: バッチ処理、圧縮、負荷分散、非同期パイプライン
- 🧠 **AI/ML コア**: モデル、進化、アテンション、符号化
- 🌐 **分散システム**: Zenoh通信、分散脳、連合学習
- 🔒 **セキュリティ・安全性**: セキュリティフレームワーク、安全ウォッチドッグ
- 🔗 **統合・E2E**: 完全統合テスト、エンドツーエンドワークフロー
- 🔌 **API・サービス**: REST API、SDK統合
- 📊 **データ処理**: テキスト、ビジョン、音声、RAG
- 👁️ **可視化・UI**: 3D可視化、脳UI

### 並列実行機能

- 最大並列実行数を設定可能（デフォルト: 3）
- 優先度順にテストスイートを実行
- 高優先度テストを先に実行

### Web UI 総合テストページ ⭐ NEW

ブラウザから直接テストを実行できる専用ページを提供しています。

**アクセス方法:**
1. Web UI起動後、ナビゲーションバーから **🧪 総合テスト** アイコンをクリック
2. テストカテゴリを選択
3. 並列実行数を設定（1-10）
4. **🚀 テスト実行** ボタンをクリック

**主な機能:**
- **リアルタイム進捗表示**: 実行中のテスト状況をリアルタイムで確認
- **詳細な結果レポート**: 成功/失敗数、実行時間、カテゴリ別結果
- **レポートダウンロード**: テスト結果をテキストファイルとして保存
- **統合実行モード**: 全テストスイートを並列実行

詳細は [docs/INTEGRATED_TEST_UI.md](docs/INTEGRATED_TEST_UI.md) を参照してください。

## 3. Infrastructure as Code (IaC) による環境管理 ⭐ NEW (2025年12月20日)

EvoSpikeNetは、**Infrastructure as Code (IaC)** を採用し、環境再現性100%を実現しました。Terraform、Ansible、Kubernetes、Docker Composeを統合した包括的な環境管理システムを提供しています。

### IaCの主な特徴

- **環境再現性100%**: どの環境でも同じ構成を自動的に再現可能
- **マルチ環境対応**: Dev/Staging/Production環境の明確な分離
- **ワンコマンドセットアップ**: 環境検証からデプロイまで完全自動化
- **Kubernetes対応**: 本番環境での大規模展開とオートスケーリング
- **ヘルスチェック統合**: 全サービスの自動健全性確認

### クイックスタート (IaC)

```bash
# 1. 環境検証と自動セットアップ
make env-setup

# 2. Terraformで環境構築
make terraform-init
make terraform-apply

# 3. Dockerサービス起動
make docker-up

# 4. ヘルスチェック
make health-check
```

### IaCツール統合

| ツール                 | 用途                 | 主な機能                                                        |
| ---------------------- | -------------------- | --------------------------------------------------------------- |
| **Terraform**          | インフラ構築         | Docker network/volume管理、環境変数自動生成、ヘルスチェック生成 |
| **Ansible**            | システムセットアップ | Docker/Python/依存関係の自動インストール（20+タスク）           |
| **Kubernetes**         | 本番デプロイ         | StatefulSet、HPA（3-10レプリカ）、Ingress、オートスケーリング   |
| **Environment Script** | 環境検証             | Python≥3.9、Docker、ディスク容量≥10GB、ポート利用可否チェック   |

### IaC関連コマンド

```bash
# 環境管理
make env-setup         # 環境セットアップと検証
make env-validate      # 検証のみ

# Terraform
make terraform-init    # 初期化
make terraform-plan    # 実行計画表示
make terraform-apply   # インフラ適用
make terraform-destroy # インフラ削除

# Ansible
make ansible-setup     # Ansibleでシステムセットアップ

# Kubernetes
make k8s-deploy        # Kubernetesへデプロイ

# Docker
make docker-build      # イメージビルド
make docker-up         # サービス起動
make docker-down       # サービス停止
make docker-logs       # ログ表示
```

### 環境タイプ

| 環境        | GPU | APIポート | UIポート | 用途         |
| ----------- | --- | --------- | -------- | ------------ |
| Development | CPU | 8000      | 8050     | ローカル開発 |
| Staging     | GPU | 8100      | 8150     | 検証環境     |
| Production  | GPU | 8200      | 8250     | 本番運用     |

詳細は [docs/INFRASTRUCTURE_AS_CODE.md](docs/INFRASTRUCTURE_AS_CODE.md) を参照してください。

## 4. Dockerを使用した環境設定 (推奨)

本プロジェクトはDocker Composeを全面的に採用しており、数コマンドで完全な開発・実行環境を構築できます。

### 前提条件
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2以降, `docker compose` コマンド)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (GPUモード利用時)

### 環境のビルド
初めて実行する際や、`Dockerfile`に変更があった場合は、以下のコマンドでDockerイメージをビルドしてください。（`sudo`が必要な場合があります）
```bash
docker compose build
```

### その他のコマンド
- **APIサーバーのみを起動:** `sudo ./scripts/run_api_server.sh`
- **テストスイートの実行:** `sudo ./scripts/run_tests_cpu.sh`

## 4. プロジェクト構成

| パス                 | 説明                                                      |
| :------------------- | :-------------------------------------------------------- |
| `evospikenet/`       | フレームワークの主要なソースコード。                      |
| `frontend/`          | DashベースのWeb UIアプリケーションのソースコード。        |
| `tests/`             | `pytest`を使用したユニットテスト。                        |
| `scripts/`           | 開発、テスト、実行を簡略化するシェルスクリプト群。        |
| `examples/`          | フレームワークの特定用途を示すサンプルプログラム。        |
| `docker-compose.yml` | 全サービス（UI, API, DB等）を定義するDocker Compose設定。 |
| `pyproject.toml`     | プロジェクトのメタデータとPythonの依存関係を定義。        |
| `README.md`          | このファイル。                                            |

## 5. ドキュメント

より詳細な技術情報や使用方法については、`docs/` ディレクトリ内の以下のドキュメントを参照してください。

### 📚 主要ドキュメント一覧

| ドキュメント名                 | 日本語版                                                                              | 英語版                                                                                      | 説明                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ | :--------------------------------------------------- |
| **コンセプト**                 | [EVOSPIKENET_CONCEPTS.md](docs/EVOSPIKENET_CONCEPTS.md)                               | [EVOSPIKENET_CONCEPTS.en.md](docs/EVOSPIKENET_CONCEPTS.en.md)                               | フレームワークの基本概念と設計思想                   |
| **ユーザーマニュアル**         | [UserManual.md](docs/UserManual.md)                                                   | [UserManual.en.md](docs/UserManual.en.md)                                                   | Web UIの操作ガイド                                   |
| **SDK**                        | [EvoSpikeNet_SDK.md](docs/EvoSpikeNet_SDK.md)                                         | [EvoSpikeNet_SDK.en.md](docs/EvoSpikeNet_SDK.en.md)                                         | Python SDKの詳細ガイド                               |
| **SDKクイックスタート**        | [SDK_QUICKSTART.md](docs/SDK_QUICKSTART.md)                                           | [SDK_QUICKSTART.en.md](docs/SDK_QUICKSTART.en.md)                                           | SDKの簡易スタートガイド                              |
| **データハンドリング**         | [DATA_HANDLING.md](docs/DATA_HANDLING.md)                                             | [DATA_HANDLING.en.md](docs/DATA_HANDLING.en.md)                                             | データ形式と処理方法                                 |
| **分散脳システム**             | [DISTRIBUTED_BRAIN_SYSTEM.md](docs/DISTRIBUTED_BRAIN_SYSTEM.md)                       | [DISTRIBUTED_BRAIN_SYSTEM.en.md](docs/DISTRIBUTED_BRAIN_SYSTEM.en.md)                       | 分散脳シミュレーションの詳細                         |
| **RAGシステム**                | [RAG_SYSTEM_DETAILED.md](docs/RAG_SYSTEM_DETAILED.md)                                 | [RAG_SYSTEM_DETAILED.en.md](docs/RAG_SYSTEM_DETAILED.en.md)                                 | ハイブリッド検索RAGの実装詳細                        |
| **実装状況とロードマップ**     | [REMAINING_FEATURES.md](docs/REMAINING_FEATURES.md)                                   | [REMAINING_FEATURES.en.md](docs/REMAINING_FEATURES.en.md)                                   | 実装済み機能と今後の計画                             |
| **実装インデックス**           | [IMPLEMENTATION_INDEX.md](docs/IMPLEMENTATION_INDEX.md)                               |                                                                                             | 全実装コンポーネントの詳細なインデックスとステータス |
| **テストスイートガイド**       | [TESTING.md](docs/TESTING.md)                                                         | [TESTING.en.md](docs/TESTING.en.md)                                                         | 包括的なテスト実行方法とベストプラクティス           |
| **統合テストメニュー**         | [TEST_MENU_README.md](TEST_MENU_README.md)                                            |                                                                                             | 統合テストシステムの使用方法と操作ガイド ⭐ NEW       |
| **APIドキュメント**            | [API Documentation](docs/api/)                                                        | [API Documentation](docs/api/)                                                              | 自動生成APIドキュメント（Swagger UI、OpenAPI）       |
| **L5自己進化実装計画**         | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.md)     | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md)     | L5レベルの自己進化機能の詳細計画 ⭐                   |
| **L5機能洗い出し**             | [L5_FEATURE_BREAKDOWN.md](docs/L5_FEATURE_BREAKDOWN.md)                               | [L5_FEATURE_BREAKDOWN.en.md](docs/L5_FEATURE_BREAKDOWN.en.md)                               | L5機能の詳細な分解と実装方針 ⭐                       |
| **LLM統合戦略**                | [LLM_INTEGRATION_STRATEGY.md](docs/LLM_INTEGRATION_STRATEGY.md)                       | [LLM_INTEGRATION_STRATEGY.en.md](docs/LLM_INTEGRATION_STRATEGY.en.md)                       | 大規模言語モデル統合の戦略                           |
| **AEG-Comm実装計画**           | [AEG_COMM_IMPLEMENTATION_PLAN.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.md)               | [AEG_COMM_IMPLEMENTATION_PLAN.en.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.en.md)               | インテリジェント通信制御の実装計画 ⭐ NEW             |
| **シミュレーション記録ガイド** | [SIMULATION_RECORDING_GUIDE.md](docs/SIMULATION_RECORDING_GUIDE.md)                   | [SIMULATION_RECORDING_GUIDE.en.md](docs/SIMULATION_RECORDING_GUIDE.en.md)                   | データ記録・解析機能の使用方法 ⭐                     |
| **シミュレーション記録README** | [SIMULATION_RECORDING_README.md](docs/SIMULATION_RECORDING_README.md)                 | [SIMULATION_RECORDING_README.en.md](docs/SIMULATION_RECORDING_README.en.md)                 | 記録機能の概要                                       |
| **スパイク通信解析**           | [SPIKE_COMMUNICATION_ANALYSIS.md](docs/SPIKE_COMMUNICATION_ANALYSIS.md)               | [SPIKE_COMMUNICATION_ANALYSIS.en.md](docs/SPIKE_COMMUNICATION_ANALYSIS.en.md)               | スパイク通信の解析手法                               |
| **パイプライン解析**           | [distributed_brain_pipeline_analysis.md](docs/distributed_brain_pipeline_analysis.md) | [distributed_brain_pipeline_analysis_en.md](docs/distributed_brain_pipeline_analysis_en.md) | 分散脳パイプラインの詳細解析                         |
| **ドキュメント更新サマリー**   | [DOCUMENTATION_UPDATE_SUMMARY.md](docs/DOCUMENTATION_UPDATE_SUMMARY.md)               | [DOCUMENTATION_UPDATE_SUMMARY.en.md](docs/DOCUMENTATION_UPDATE_SUMMARY.en.md)               | ドキュメント更新履歴                                 |

### 📁 その他のドキュメント

- **アーキテクチャ図**: `docs/architecture/` ディレクトリ
- **SDK詳細**: `docs/sdk/` ディレクトリ

---

## 🤖 LLMトレーニング統合システム

EvoSpikeNetは、大規模言語モデル（LLM）のトレーニングと管理のための包括的な統合システムを提供します。

### サポートされるLLMカテゴリ

- **🗣️ LangText**: テキストベース言語モデル（GPT, BERT, DialoGPT）
- **👁️ Vision**: 画像認識・分類モデル（ViT, ResNet, CLIP）
- **🎵 Audio**: 音声処理モデル（Whisper, Wav2Vec2）
- **🔗 MultiModal**: マルチモーダルモデル（CLIP, LLaVA）

### クイックスタート

1. **データ収集**:
   ```bash
   make collect-data
   ```

2. **トレーニングサーバー起動**:
   ```bash
   # GPUトレーニング
   make train-gpu
   
   # CPUトレーニング
   make train-cpu
   
   # 両方 + ロードバランサー
   make train-all
   ```

3. **API経由でトレーニング開始**:
   ```bash
   curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "category": "LangText",
       "model_name": "microsoft/DialoGPT-medium",
       "dataset_path": "data/llm_training/LangText/langtext_data.jsonl",
       "output_dir": "saved_models/LangText/my-run",
       "gpu": true,
       "epochs": 3
     }'
   ```

### ダッシュボード統合

フロントエンドダッシュボード（http://localhost:8050）から直接LLMトレーニングジョブを管理できます：

- **TransformerLMページ**: LangTextモデルのトレーニング
- **MultiModal LMページ**: Audioモデルのトレーニング
- リアルタイムジョブステータス監視
- トレーニングログの表示

### Docker統合

```bash
# トレーニングAPI + フロントエンド起動
docker-compose up llm-training-api frontend

# ステータス確認
docker-compose ps
```

詳細は [docs/LLM_TRAINING_SYSTEM.md](docs/LLM_TRAINING_SYSTEM.md) を参照してください。

