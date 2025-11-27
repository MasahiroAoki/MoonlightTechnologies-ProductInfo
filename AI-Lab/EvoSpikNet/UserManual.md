# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet Dashboard ユーザーマニュアル

**最終更新日:** 2025年11月26日

## 1. はじめに

このドキュメントは、`EvoSpikeNet Dashboard`の各機能の使用方法について解説します。このダッシュボードは、EvoSpikeNetフレームワークの持つ複雑な機能を、ブラウザから直感的に操作し、可視化するための統一されたインターフェースです。

## 2. セットアップと起動

本プロジェクトはDocker Composeを全面的に採用しており、アプリケーションと必要なバックエンドサービス（API, データベース, Milvusなど）を一度に起動できます。

### 2.1. 前提条件
- Docker
- Docker Compose (v2以降)
- (GPU利用の場合) NVIDIA Container Toolkit

### 2.2. 起動方法
プロジェクトのルートディレクトリで、まず以下のコマンドを実行してDockerイメージをビルドします（初回または更新時）。`sudo`が必要な場合があります。
```bash
docker compose build
```
その後、以下のスクリプトを実行してWeb UIと関連サービスを起動します。

```bash
# GPUを利用可能な環境の場合
sudo ./scripts/run_frontend_gpu.sh

# CPUのみの環境の場合
sudo ./scripts/run_frontend_cpu.sh
```
これにより、`http://localhost:8050`でダッシュボードが利用可能になります。

---

## 3. UIナビゲーションと各ページの機能

ダッシュボードは、画面上部のナビゲーションメニューに基づいた階層構造になっています。

### 3.1. Home
このユーザーマニュアルやプロジェクトのREADMEなど、基本的なドキュメントが表示されるデフォルトページです。

### 3.2. Data Generation （データ生成）
モデルの訓練や分析に使用する様々なデータを生成・管理するためのページ群です。

- **SNN Models**:
    - 4層の基本的なSNNモデルのシミュレーションを実行し、その内部状態（スパイク活動、膜電位）を可視化・保存 (`.pt`ファイル) します。ニューロンの種類（LIF/Izhikevich）や各種パラメータをUIから直接設定できます。
- **Knowledge Base**:
    - RAGシステムが利用する知識ベースを管理します。テーブル内のドキュメントを直接、作成(Create)、読み取り(Read)、更新(Update)、削除(Delete)できます。ここでの変更は、MilvusとElasticsearchの両方にリアルタイムで同期されます。
- **Automated Learning/Optimization**:
    - **Hyperparameter Tuning**: `Optuna`を利用したハイパーパラメータチューニングを実行します。探索したいパラメータと範囲を設定し、チューニングを開始します。
    - **Auto Learning**: `run_auto_learning.py`スクリプトを介して、継続的な自動学習サイクルを実行します。

### 3.3. Prompt （対話型プロンプト）
訓練済みのモデルと対話するためのインターフェースです。

- **RAG Chat**:
    - RAGシステムとチャット形式で対話します。ドロップダウンから複数のLLMバックエンド（OpenAI, Hugging Face, SNNなど）を選択できます。SNNバックエンドを選択した場合、「Save Neuron Data」を有効にすることで、推論時のニューロン活動をファイルに保存できます。
- **Spiking LM Chat**:
    - `SpikingEvoSpikeNetLM`と直接対話します。訓練済みのモデルを選択し、プロンプトを入力してテキストを生成します。こちらもニューロン活動の保存が可能です。

### 3.4. Data Analysis （データ分析）
生成または保存されたデータを可視化・分析するためのページ群です。

- **Generic Visualization**:
    - フレームワークの各機能（RAG Chat, Spiking LM Chat, SNN Modelsなど）で生成されたニューロンデータ (`.pt`ファイル) をアップロードし、その内容（スパイク、膜電位、アテンションなど）を可視化します。これにより、異なるモデルの内部状態を同じインターフェースで横断的に分析できます。
- **Tuning Results**:
    - 「Automated Learning」ページで実行したハイパーパラメータチューニングの結果 (`.db`ファイル) を選択し、最適化の履歴や各パラメータの重要度などをインタラクティブなグラフで分析します。

### 3.5. System Settings （システム設定）
システム全体の管理やユーティリティ機能を提供します。

- **System Utilities**:
    - テストスイートの実行、LLMによる合成データ生成（データ蒸留）、モデルの性能評価など、システムレベルのコマンドを実行します。
- **Model Management**:
    - `saved_models`ディレクトリに保存されている訓練済みモデルを一覧表示し、不要になったモデルを削除することができます。

---

## 4. 主要機能の詳細解説: Distributed Brain

「Prompt」メニューにある「Distributed Brain」ページは、本フレームワークの最も高度な機能であり、`run_distributed_brain_simulation.py`スクリプトを介して、複数のプロセス（ノード）からなる分散脳シミュレーションを管理・実行します。

### 4.1. Node Configuration （ノード設定）タブ

**目的**: シミュレーションの構成を定義し、各`torch.distributed`プロセスを起動・停止します。

**ステップバイステップガイド**:
1.  **リモートホストの設定 (マルチPCの場合)**:
    - シミュレーションを複数のマシンにまたがって実行する場合、「Remote Host Configuration」テーブルに、SSH接続に必要な情報（IPアドレス、ユーザー名、SSHキーのパス）を追加し、「Save Hosts」をクリックします。
    - **技術的注意**: `docker-compose.yml`の設定により、ホストマシンの`~/.ssh`ディレクトリがコンテナにマウントされるため、SSHキーのパスはホストマシン上のパス（例: `~/.ssh/id_rsa`）をそのまま使用できます。

2.  **シミュレーションの構成**:
    - **Select Simulation Type**: `Language Focus`や`Image Focus`など、実行したい定義済みの脳アーキテクチャを選択します。選択に応じて、必要なノードリストが下のテーブルに表示されます。
    - **Select Model for Node**: 各ノード（特に言語や視覚などの主要モジュール）に対して、データベースに保存されている訓練済みモデルを選択できます。ここで`MultiModalEvoSpikeNetLM`のようなマルチモーダルモデルを選択すると、そのノードは複数の種類のデータ（例: 視覚ノードが画像とテキストの両方）を処理できるようになります。

3.  **ノードのホスト割り当て**:
    - 「Node Status & Host Assignment」テーブルに表示された各ノード（プロセス）に対し、「Host」ドロップダウンから実行させたいマシン（`localhost`または設定したリモートホスト）を割り当てます。Rank 0 (PFC) はマスタープロセスとして機能するため、通常`localhost`で実行します。

4.  **シミュレーションの開始・停止**:
    - **Start Nodes**: このボタンをクリックすると、各ノードが`run_distributed_brain_simulation.py`のインスタンスとして、割り当てられたホスト上でプロセスとして起動します（リモートホストへはSSH経由）。テーブルの「Status」列が「Starting...」に変わります。
    - **Stop Nodes**: 実行中のすべてのシミュレーションプロセスを安全に停止します。バックエンドではファイルフラグ (`/tmp/stop_evospikenet_simulation.flag`) が作成され、各プロセスがそれを検知してクリーンに終了します。

### 4.2. Brain Simulation （脳シミュレーション）タブ

**目的**: 実行中のシミュレーションと対話し、その内部状態をリアルタイムで監視します。

**操作方法**:
1.  **プロンプトの送信**:
    - テキストエリアにクエリを入力し、必要に応じて画像や音声ファイルをアップロードします。
    - **マルチモーダルな入力**: ノードにマルチモーダルモデルを設定した場合、ここでの入力は組み合わせて解釈されます。例えば、視覚ノードに`MultiModalEvoSpikeNetLM`を設定し、画像と言語プロンプトを両方提供すると、その両方が視覚ノードに送信され、総合的に処理されます。
    - 「Execute Query」ボタンをクリックすると、プロンプトがAPIエンドポイント (`/api/distributed_brain/prompt`) に送信され、実行中のシミュレーションのマスタープロセス (Rank 0, PFC) によってポーリングされます。

2.  **状態の監視と結果の確認**:
    - **Query Status & Live Simulation Graph**: これらのUIコンポーネントは、`/api/distributed_brain/status`エンドポイントを定期的にポーリングします。マスタープロセス (PFC) はシミュレーションの現在の状態（各ノードのステータス、アクティブな通信を示すエッジなど）をこのエンドポイントにPOSTするため、UIがリアルタイムで更新されます。
    - **PFC Charts**: PFCの内部状態（エネルギー、エントロピー、スパイク活動）が時系列グラフで表示されます。
    - **Query Response**: シミュレーションが完了すると、PFCは最終的な応答を`/api/distributed_brain/result`エンドポイントにPOSTし、このエリアに表示されます。

3.  **ログの確認**:
    - 「Node Logs」セクションのドロップダウンからノード（例: `Node 4 - Lang-Main`）を選択すると、そのノードが生成したログファイル (`/tmp/sim_rank_{rank}.log`) の内容が表示されます。この機能は**リモートホストにも対応**しており、API経由でリモートマシンのログファイルを取得して表示します。
