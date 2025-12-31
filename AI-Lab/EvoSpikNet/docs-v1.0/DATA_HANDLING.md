<!-- Reviewed against source: 2025-12-21. English translation pending. -->
# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet: データ取扱ガイド

**最終更新日:** 2025年12月15日

## このドキュメントの目的と使い方
- 目的: 各種データ（スパイク/テキスト/RAG/マルチモーダル）の作成・フォーマット・検証手順を一望できるようにする。
- 対象読者: データパイプライン担当、研究/学習オペレーション担当。
- まず読む順: データアップロード → スパイクデータ → テキストコーパス → 知識ベース → マルチモーダルデータセット。
- 関連リンク: 分散脳スクリプトは `examples/run_zenoh_distributed_brain.py`、PFC/Zenoh/Executive詳細は [docs/implementation/PFC_ZENOH_EXECUTIVE.md](docs/implementation/PFC_ZENOH_EXECUTIVE.md)。

- 実装ノート（アーティファクト）: トレーニングジョブが生成する `artifact_manifest.json` とアップロード時のルールについては `docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。

このドキュメントでは、スパイクデータ、テキストコーパス、RAG知識ベース、マルチモーダルデータセットなど、`EvoSpikeNet`フレームワークで使用される様々なAIデータの作成、フォーマット、検証方法について詳述します。

---

## 0. データアップロード機能

EvoSpikeNetは、トレーニングデータをAPIサーバーにアップロードし、分散環境での共有を可能にする機能を提供します。これにより、ローカルファイルだけでなく、アップロードされたデータセットを使用してLLMトレーニングを実行できます。

### アップロード可能なデータ形式

- **マルチモーダルデータ**: 画像とキャプションのペア（`captions.csv` + `images/`ディレクトリ）
- **将来拡張予定**: 音声データ、テキストコーパスなどのサポート

### アップロード手順

1. **データ準備**: ローカルにトレーニングデータを作成
2. **スクリプト実行**: `--upload-data`フラグを使用してトレーニングを開始
3. **APIアップロード**: データが自動的にZIP圧縮され、APIサーバーにアップロード
4. **共有利用**: アップロードされたデータは他のユーザーやシステムで再利用可能

### 使用例

```bash
# データアップロード付きトレーニング
python examples/train_multi_modal_lm.py \
  --mode train \
  --dataset custom \
  --data-dir your_data_dir \
  --run-name your_run \
  --upload-data \
  --data-name your_dataset_name \
  --epochs 10 \
  --batch-size 4
```

### APIサーバーが利用できない場合

APIサーバーが利用できない場合は、自動的にローカルデータでのトレーニングにフォールバックします。データアップロードはスキップされ、ローカルファイルを使用してトレーニングが継続されます。

---

## 1. スパイクデータの生成とフォーマット

SNNモデルへの直接入力となるスパイクデータは、`torch.Tensor`として表現されます。

- **フォーマット**: `torch.Tensor`
- **形状**: `(time_steps, num_input_neurons)`の2Dテンソル。
- **dtype**: `torch.int8`
- **値**: `0` (スパイクなし) または `1` (スパイクあり)。

テスト用の人工的なスパイクデータは、UIの「Data Generation」メニューにある「SNN Models」ページで生成できます。

---

## 2. テキストコーパス

言語モデル（`EvoSpikeNetLM`, `SpikingEvoSpikeNetLM`）の訓練には、`evospikenet/dataloaders.py`モジュールがサポートする様々なデータソースが利用可能です。

- **`WikipediaLoader`**: Wikipediaの記事を動的に読み込みます。
- **`AozoraBunkoLoader`**: 青空文庫のHTMLからテキストを抽出します。
- **`LocalFileLoader`**: ローカルのテキストファイルを読み込みます。

これらのローダーは、`examples/train_spiking_evospikenet_lm.py`などの訓練スクリプトで活用されます。

---

## 3. RAG知識ベースの管理

Retrieval-Augmented Generation (RAG)機能は、MilvusとElasticsearchに外部知識を保存します。データ管理は主にUIから行います。

- **データ構造**: 各ドキュメントは、`id`（一意）、`embedding`（ベクトル）、`text`（本文）、`source`（出所）のフィールドで構成されます。
- **UIによるCRUD操作**: 「RAG System」ページの「Data Management」タブは、知識ベースを直接管理するための強力なインターフェースです。
    - **作成 (Create)**: `add row`ボタンで行を追加し、`text`と`source`を入力します。`embedding`は自動で生成・保存されます。
    - **読み取り (Read)**: Milvus内の全データがテーブルに表示されます。
    - **更新 (Update)**: テーブルのセルを直接編集すると、リアルタイムでデータベースが更新されます。
    - **削除 (Delete)**: 行を選択し、`delete row`ボタンで削除します。

---

## 4. マルチモーダルデータセット

`MultiModalEvoSpikeNetLM`は、画像とキャプションのペアで訓練されます。

- **ディレクトリ構造**:
  ```
  data/multi_modal_dataset/
  ├── images/ (画像ファイルを格納)
  └── captions.csv (画像パスとキャプションの対応を記述)
  ```
- **`captions.csv`のフォーマット**:
  ```csv
  image_path,caption
  images/image_0.png,"キャプション1"
  images/image_1.jpg,"キャプション2"
  ```
このデータセットは、`examples/train_multi_modal_lm.py`スクリプトによるモデル訓練に使用されます。

---

## 5. 可視化用データ

UIでのインタラクティブな分析やオフラインでの詳細な可視化のため、フレームワークは`.pt`形式でニューロン活動データを保存します。

- **データ構造**: すべてのファイルは`spikes`（スパイク）、`membrane_potential`（膜電位）などのキーを持つ辞書として保存されます。
- **生成される場所**:
    - **RAG Chat**: SNNバックエンド選択時にニューロンデータを保存可能。
    - **Spiking LM Chat**: テキスト生成時にニューロンデータを保存可能。
    - **SNN Models**: 4層SNNのシミュレーション実行時に生成（例: `4_layer_snn_data_lif.pt`）。
- **利用方法**: 生成された`.pt`ファイルは、「Data Analysis」メニューの「Generic Visualization」ページにアップロードして再可視化したり、`examples/visualize_*.py`スクリプトでオフライン分析したりできます。

---

## 6. 合成データ生成 (`Data Distillation`)

`evospikenet/distillation.py`モジュールは、LLM（例: OpenAI）を用いて高品質な合成データを生成する機能を提供します。これは、特定のタスク（感情分析、QAペア生成など）のデータセットを効率的に作成するのに役立ちます。

「System Settings」メニューの「System Utilities」ページにある「Distill Data」コマンドから、タスクタイプ、サンプル数、プロンプトを指定して実行できます。

---

## 7. 音声データ

マルチモーダルモデルは、音声入力をサポートしています。

- **フォーマット**: `.wav`, `.mp3`, `.flac`など、`torchaudio`がサポートする標準的な音声ファイル形式。
- **UIからの利用**: 「Distributed Brain」ページの「Brain Simulation」タブから音声ファイルをアップロードし、テキストプロンプトや画像と共にシミュレーションへの入力として使用できます。
- **データ処理**: バックエンドでは、アップロードされたファイルは`torchaudio.load`で波形とサンプルレートに変換され、モデルの`SpikingAudioEncoder`が処理できる形式に前処理されます。

---

## 8. フェデレーテッド学習データセット

フェデレーテッド学習では、各クライアントが独立したローカルデータセットを保持します。

- **フォーマット**: CSV (`.csv`) ファイル。
- **データ構造**: 現在の実装では、テキスト分類タスクを想定しており、各行が`text`と`label`の2つの列で構成されている必要があります。
- **利用方法**: `examples/run_fl_client.py`スクリプトを実行する際に、`--data-path`引数でローカルのCSVファイルへのパスを指定します。

---

## 9. 分散脳シミュレーションのデータフロー

分散脳シミュレーションは、UI、API、シミュレーションプロセス間で複数のデータをやり取りします。

- **入力データ (UI → API → Simulation)**:
    1. ユーザーは「Distributed Brain」UIでテキストプロンプトを入力し、画像や音声ファイルをアップロードします。
    2. 「Execute Query」ボタンを押すと、メディアファイルはBase64エンコードされ、テキストと共にAPIエンドポイント `/api/distributed_brain/prompt` に送信されます。
    3. APIは、受け取ったプロンプトデータとメディアファイルを、一意のIDを持つJSONファイルおよび関連ファイルとしてサーバーの `/tmp` ディレクトリに書き出します。
    4. シミュレーションプロセス（特にRank 0のPFC）は、この `/tmp` ディレクトリを定期的にスキャン（ポーリング）して新しいプロンプトファイルを検出し、処理を開始します。

- **出力データ (Simulation → API → UI)**:
    - **状態 (Status)**: Rank 0プロセスは、シミュレーションの現在の状態（各ノードのステータス、エッジのアクティビティ、PFCのエントロピーなど）を定期的にAPIエンドポイント `/api/distributed_brain/status` にPOSTします。UIはこのエンドポイントをポーリングして表示をリアルタイムに更新します。
    - **結果 (Result)**: シミュレーションがタスクを完了すると、最終的なテキスト結果を `/tmp` ディレクトリに結果ファイルとして書き出します。UIは、APIエンドポイント `/api/distributed_brain/result` をポーリングします。このエンドポイントは対応する結果ファイルを読み取り、その内容をUIに返し、ファイルを削除します。
    - **ログ (Logs)**: 各シミュレーションプロセス（Rank 0, 1, ...）は、自身のログを `/tmp/sim_rank_{rank}.log` というファイルに書き出します。UIは、選択されたノードのログを（必要に応じてAPIを介して）読み込み、表示します。
    - **アーティファクト (Artifacts)**: シミュレーション中、各プロセスは内部状態（スパイクや膜電位など）のテンソルを `.pt` ファイルとしてデータベースにアーティファクトとしてアップロードできます。これは `EvoSpikeNetAPIClient` の `upload_artifact` メソッドを介して行われます。
