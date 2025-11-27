# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet - 分散脳シミュレーションフレームワーク

**最終更新日:** 2025年11月24日

## 1. プロジェクト概要

EvoSpikeNetは、生物学的な脳の機能的専門化と統合の原理に着想を得た、スケーラブルな**分散脳シミュレーションフレームワーク**です。専門化されたニューラルモジュール（視覚、言語、運動など）が個別のプロセスとして動作し、それらを中央の**前頭前野（PFC）モジュール**が動的に調整・統合します。

本フレームワークの最大の特徴は、PFCに実装された**Q-PFCフィードバックループ**です。これは、PFCが自身の意思決定の不確実性（認知エントロピー）を測定し、その値を用いて量子インスパイアード回路をシミュレート、その結果を自身のニューロン活動にフィードバックするという、高度な自己言及的制御メカニズムです。

`torch.distributed`を基盤とし、マルチプロセス/マルチノードでの実行をサポートすることで、単一デバイスの制約を超えた大規模なニューロモーフィックシステムの構築と研究を可能にします。

## 2. 主な実装済み機能

- **分散脳シミュレーション**:
    - **PFCによる認知制御**: `ChronoSpikeAttention`を用いたタスクルーティングと、Q-PFCフィードバックループによる自己変調機能を備えた中央制御ハブ。
    - **階層的機能モジュール**: 視覚、聴覚、言語、運動などの各機能が、親ノードと複数の子ノードからなる階層的な処理パイプラインとして実装されており、段階的な情報処理を実現。
    - **UIによる対話**: Web UIからテキスト、画像、音声を含むマルチモーダルなプロンプトを送信し、シミュレーションの実行、リアルタイムな状態監視、結果の取得が可能。

- **Q-PFCフィードバックループ**:
    - PFCが自身の認知負荷（エントロピー）に応じて、`QuantumFeedbackSimulator`を介して自身のワーキングメモリのダイナミクスを動的に調整する、本フレームワークの最も独創的な機能。

- **本格的なSNN言語モデル (`SpikingEvoSpikeNetLM`)**:
    - `snnTorch`ベースの、スパイクで動作するTransformerモデル。`TAS-Encoding`や`ChronoSpikeAttention`などのカスタムコンポーネントを含む。

- **トライモーダル処理能力 (`MultiModalEvoSpikeNetLM`)**:
    - テキスト、画像 (`SpikingVisionEncoder`)、音声 (`SpikingAudioEncoder`) の3つのモダリティを統合的に処理。

- **ハイブリッド検索RAG**:
    - Milvus（ベクトル検索）とElasticsearch（キーワード検索）を並列で実行し、Reciprocal Rank Fusion (RRF) アルゴリズムで結果を融合することで、高精度な検索拡張生成を実現。
    - **長文対応**: 最大32,000文字のドキュメントをサポート（Milvusの上限は65,535文字）。
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

より詳細な技術情報や使用方法については、以下のドキュメントを参照してください。
- **技術コンセプト**: `EVOSPIKENET_CONCEPTS.md`
- **機能分析**: `FEATURE_ANALYSIS.md`
- **UI操作ガイド**: `UserManual.md`
- **SDKガイド**: `EvoSpikeNet_SDK.md`
- **データ形式**: `DATA_HANDLING.md`
- **実装状況とロードマップ**: `REMAINING_FEATURES.md`
