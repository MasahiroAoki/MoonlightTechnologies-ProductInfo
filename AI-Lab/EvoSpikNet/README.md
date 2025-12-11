
# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# ⚠️ 商用利用される企業様への重要なお知らせ

本リポジトリのコードは MIT License で公開されていますが、**企業による商用製品・商用サービスへの組み込み利用**には、および**当社が保有する特許を実施**して収益を得る場合には、別途「企業向け商用ライセンス契約」の締結が必要です。

対象となる主なケース：
- 自社SaaS・アプリ・サービスに本フレームワークを組み込んで有料提供する場合
- 本フレームワークを使って顧客にAI機能を提供し対価を得る場合
- 社内システムであっても収益に直結する形で大規模利用する場合

→ 該当する企業様は必ず以下までお問い合わせください  
✉️ maoki@moonlight-tech.biz  
Web: https://www.moonlight-tech.biz/commercial-license

個人利用・研究利用・PoC・スタートアップのプロトタイプなどは完全に無料でMITライセンスのままご利用いただけます。

# EvoSpikeNet - 分散脳シミュレーションフレームワーク

**最終更新日:** 2025年12月10日

## 1. プロジェクト概要

EvoSpikeNetは、生物学的な脳の機能的専門化と統合の原理に着想を得た、スケーラブルな**分散脳シミュレーションフレームワーク**です。専門化されたニューラルモジュール（視覚、言語、運動など）が個別のプロセスとして動作し、それらを中央の**前頭前野（PFC）モジュール**が動的に調整・統合します。

本フレームワークの最大の特徴は、PFCに実装された**Q-PFCフィードバックループ**です。これは、PFCが自身の意思決定の不確実性（認知エントロピー）を測定し、その値を用いて量子インスパイアード回路をシミュレート、その結果を自身のニューロン活動にフィードバックするという、高度な自己言及的制御メカニズムです。

`torch.distributed`を基盤とし、マルチプロセス/マルチノードでの実行をサポートすることで、単一デバイスの制約を超えた大規模なニューロモーフィックシステムの構築と研究を可能にします。

## 2. 主な実装済み機能

- **分散脳シミュレーション (Zenohベース)**:
    - **非同期通信**: `Zenoh` publish/subscribeモデルを採用し、`torch.distributed`ベースの旧アーキテクチャから脱却。これにより、堅牢性とスケーラビリティが大幅に向上。
    - **PFCによる認知制御**: `ChronoSpikeAttention`を用いたタスクルーティングと、Q-PFCフィードバックループによる自己変調機能を備えた中央制御ハブ。
    - **階層的機能モジュール**: 視覚、聴覚、言語、運動などの各機能が、親ノードと複数の子ノードからなる階層的な処理パイプラインとして実装。
    - **UIによる対話**: Web UIからテキスト、画像、音声を含むマルチモーダルなプロンプトを送信し、シミュレーションの実行、リアルタイムな状態監視、結果の取得が可能。

- **Q-PFCフィードバックループ**:
    - PFCが自身の認知負荷（エントロピー）に応じて、`QuantumModulationSimulator`を介して自身のワーキングメモリのダイナミクスを動的に調整する、本フレームワークの最も独創的な機能。

- **本格的なSNN言語モデル (`SpikingTextLM`)**:
    - `snnTorch`ベースの、スパイクで動作するTransformerモデル。`TAS-Encoding`や`ChronoSpikeAttention`などのカスタムコンポーネントを含む。

- **トライモーダル処理能力 (`SpikingMultiModalLM`)**:
    - テキスト、画像 (`SpikingVisionEncoder`)、音声 (`SpikingAudioEncoder`) の3つのモダリティを統合的に処理。

- **ハイブリッド検索RAG**:
    - Milvus（ベクトル検索）とElasticsearch（キーワード検索）を並列で実行し、Reciprocal Rank Fusion (RRF) アルゴリズムで結果を融合することで、高精度な検索拡張生成を実現。
    - **長文対応**: Milvusのスキーマ定義に基づき、最大65,535文字のドキュメントを保存可能。
    - **自動プロンプト切り詰め**: Hugging Faceモデルなどの制約に合わせて、プロンプトを自動的に最適な長さに調整。
    - **対話型データ管理**: チェックボックスによる行選択、インライン編集、リアルタイム文字数カウンターなど、使いやすいUI。

- **多様なSNNコアエンジン**:
    - 計算効率に優れた`LIFNeuronLayer`、生物学的妥当性の高い`IzhikevichNeuronLayer`、量子インスパイアードの`EntangledSynchronyLayer`など、複数のニューロンモデルをサポート。

- **フェデレーテッド学習 (Flower)**:
    - `Flower`フレームワークを統合し、プライバシーを保護しながら分散環境でモデルを協調学習させる機能をサポート。

- **RESTful APIとPython SDK**:
    - `FastAPI`ベースのAPIが、テキスト生成、データロギング、分散脳シミュレーションの制御など、フレームワークの全機能へのプログラムアクセスを提供。
    - APIを容易に利用するためのPython SDK (`EvoSpikeNetAPIClient`)も完備。
    - **コンテナ間通信の最適化**: ファイルベースからAPI経由の通信に変更し、Dockerコンテナ間での信頼性を向上。

- **統合Web UI**:
    - データ生成、モデル訓練、推論、結果分析、システム管理など、フレームワークの全機能をブラウザからインタラクティブに操作できるDashベースのマルチページアプリケーション。
    - **リアルタイム状態監視**: 分散脳シミュレーションの各ノードの状態、エネルギー、スパイク活動をリアルタイムで可視化。
    - **マルチモーダルクエリ**: テキスト、画像、音声を組み合わせた複雑なクエリに対応。

- **シミュレーションデータ記録・解析**:
    - オプション機能として、スパイク、膜電位、重み、制御状態を記録可能。
    - HDF5形式での効率的なデータ保存、自動解析、可視化ツール (`sim_recorder.py`, `sim_analyzer.py`)。
    - 発火率計算、スパイクラスタープロット、発火率時系列プロット、サマリーレポート自動生成。

## 3. Web UIの起動

シミュレーションの実行、パラメータ調整、結果の可視化は、すべてWeb UIから行えます。以下のコマンドで、UIとそれに必要なバックエンドサービス（API, Milvus, Elasticsearchなど）をすべて起動できます。

```bash
# GPUを利用可能な環境の場合
sudo ./scripts/run_frontend_gpu.sh

# CPUのみの環境の場合
sudo ./scripts/run_frontend_cpu.sh
```
起動後、ブラウザで `http://localhost:8050` にアクセスしてください。

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

## 5. プロジェクト構成

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

## 6. ドキュメント

より詳細な技術情報や使用方法については、`docs/` ディレクトリ内の以下のドキュメントを参照してください。

### 📚 主要ドキュメント一覧

| ドキュメント名                 | 日本語版                                                                              | 英語版                                                                                      | 説明                                     |
| :----------------------------- | :------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ | :--------------------------------------- |
| **コンセプト**                 | [EVOSPIKENET_CONCEPTS.md](docs/EVOSPIKENET_CONCEPTS.md)                               | [EVOSPIKENET_CONCEPTS.en.md](docs/EVOSPIKENET_CONCEPTS.en.md)                               | フレームワークの基本概念と設計思想       |
| **ユーザーマニュアル**         | [UserManual.md](docs/UserManual.md)                                                   | [UserManual.en.md](docs/UserManual.en.md)                                                   | Web UIの操作ガイド                       |
| **SDK**                        | [EvoSpikeNet_SDK.md](docs/EvoSpikeNet_SDK.md)                                         | [EvoSpikeNet_SDK.en.md](docs/EvoSpikeNet_SDK.en.md)                                         | Python SDKの詳細ガイド                   |
| **SDKクイックスタート**        | [SDK_QUICKSTART.md](docs/SDK_QUICKSTART.md)                                           | [SDK_QUICKSTART.en.md](docs/SDK_QUICKSTART.en.md)                                           | SDKの簡易スタートガイド                  |
| **データハンドリング**         | [DATA_HANDLING.md](docs/DATA_HANDLING.md)                                             | [DATA_HANDLING.en.md](docs/DATA_HANDLING.en.md)                                             | データ形式と処理方法                     |
| **分散脳システム**             | [DISTRIBUTED_BRAIN_SYSTEM.md](docs/DISTRIBUTED_BRAIN_SYSTEM.md)                       | [DISTRIBUTED_BRAIN_SYSTEM.en.md](docs/DISTRIBUTED_BRAIN_SYSTEM.en.md)                       | 分散脳シミュレーションの詳細             |
| **RAGシステム**                | [RAG_SYSTEM_DETAILED.md](docs/RAG_SYSTEM_DETAILED.md)                                 | [RAG_SYSTEM_DETAILED.en.md](docs/RAG_SYSTEM_DETAILED.en.md)                                 | ハイブリッド検索RAGの実装詳細            |
| **実装状況とロードマップ**     | [REMAINING_FEATURES.md](docs/REMAINING_FEATURES.md)                                   | [REMAINING_FEATURES.en.md](docs/REMAINING_FEATURES.en.md)                                   | 実装済み機能と今後の計画                 |
| **L5自己進化実装計画**         | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.md)     | [L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md](docs/L5_EVO_GENOME_IMPLEMENTATION_PLAN.en.md) | L5レベルの自己進化機能の詳細計画 ⭐       |
| **L5機能洗い出し**             | [L5_FEATURE_BREAKDOWN.md](docs/L5_FEATURE_BREAKDOWN.md)                               | [L5_FEATURE_BREAKDOWN.en.md](docs/L5_FEATURE_BREAKDOWN.en.md)                           | L5機能の詳細な分解と実装方針 ⭐           |
| **LLM統合戦略**                | [LLM_INTEGRATION_STRATEGY.md](docs/LLM_INTEGRATION_STRATEGY.md)                       | [LLM_INTEGRATION_STRATEGY.en.md](docs/LLM_INTEGRATION_STRATEGY.en.md)                   | 大規模言語モデル統合の戦略               |
| **AEG-Comm実装計画**           | [AEG_COMM_IMPLEMENTATION_PLAN.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.md)               | [AEG_COMM_IMPLEMENTATION_PLAN.en.md](docs/AEG_COMM_IMPLEMENTATION_PLAN.en.md)           | インテリジェント通信制御の実装計画 ⭐ NEW |
| **シミュレーション記録ガイド** | [SIMULATION_RECORDING_GUIDE.md](docs/SIMULATION_RECORDING_GUIDE.md)                   | [SIMULATION_RECORDING_GUIDE.en.md](docs/SIMULATION_RECORDING_GUIDE.en.md)                   | データ記録・解析機能の使用方法 ⭐         |
| **シミュレーション記録README** | [SIMULATION_RECORDING_README.md](docs/SIMULATION_RECORDING_README.md)                 | [SIMULATION_RECORDING_README.en.md](docs/SIMULATION_RECORDING_README.en.md)                 | 記録機能の概要                           |
| **スパイク通信解析**           | [SPIKE_COMMUNICATION_ANALYSIS.md](docs/SPIKE_COMMUNICATION_ANALYSIS.md)               | [SPIKE_COMMUNICATION_ANALYSIS.en.md](docs/SPIKE_COMMUNICATION_ANALYSIS.en.md)               | スパイク通信の解析手法                   |
| **パイプライン解析**           | [distributed_brain_pipeline_analysis.md](docs/distributed_brain_pipeline_analysis.md) | [distributed_brain_pipeline_analysis_en.md](docs/distributed_brain_pipeline_analysis_en.md) | 分散脳パイプラインの詳細解析             |
| **ドキュメント更新サマリー**   | [DOCUMENTATION_UPDATE_SUMMARY.md](docs/DOCUMENTATION_UPDATE_SUMMARY.md)               | [DOCUMENTATION_UPDATE_SUMMARY.en.md](docs/DOCUMENTATION_UPDATE_SUMMARY.en.md)               | ドキュメント更新履歴                     |

### 📁 その他のドキュメント

- **アーキテクチャ図**: `docs/architecture/` ディレクトリ
- **SDK詳細**: `docs/sdk/` ディレクトリ

