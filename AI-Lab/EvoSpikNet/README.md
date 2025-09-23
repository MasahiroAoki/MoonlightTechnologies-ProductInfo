# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# EvoSpikeNet - 分散型進化ニューロモーフィックフレームワーク

## 1. プロジェクト概要

EvoSpikeNetは、GPUの並列計算能力またはCPUを活用し、脳の可塑性を模倣した、スケーラブルでエネルギー効率の高い「動的スパイクニューラルネットワーク（SNN）」のシミュレーションフレームワークです。浮動小数点演算を排除し、整数演算とスパース計算に特化することで、リアルタイムの時系列データ処理やエッジAI応用を目指します。

## 2. 主な実装済み機能

`REMAINING_FEATURES.md`で定義されているフェーズに基づき、本フレームワークは以下の主要な機能セットを実装しています。

- **コアSNNエンジン (フェーズ1):**
    - `LIFNeuronLayer`: 整数演算ベースのLIFニューロン層。
    - `SynapseMatrixCSR`: メモリ効率の高いスパースなシナプス接続。
    - `SNNModel`: 複数の層とシナプスを統合し、シミュレーションを実行するコアモデル。
- **動的進化と可視化 (フェーズ2):**
    - `STDP`: スパイクタイミング依存可塑性（STDP）によるオンライン学習。
    - `DynamicGraphEvolutionEngine`: シナプスの生成・削除を行う動的グラフ進化。
    - `InsightEngine`: 発火活動のラスタプロットや接続構造のグラフ化など、ネットワーク内部を可視化。
- **エネルギー駆動型コンピューティング (フェーズ3):**
    - `EnergyManager`: ニューロンの発火をエネルギーレベルに基づいて制限し、より生物学的に現実的なシミュレーションを実現。
- **テキスト処理 (フェーズ4):**
    - `WordEmbeddingLayer`, `PositionalEncoding`, `RateEncoder` によるテキストデータのスパイク列への変換。
- **量子インスパイアード機能 (フェーズ3):**
    - `EntangledSynchronyLayer`: 文脈に応じてニューロン群の同期を動的に制御する特殊レイヤー。
    - `HardwareFitnessEvaluator`: ハードウェアの性能指標を考慮した進化的アルゴリズムの適応度評価関数。
- **実験的な勾配ベース学習 (フェーズ6):**
    - `examples/run_gradient_training_demo.py`: コアライブラリの安定性を損なうことなく、代理勾配（Surrogate Gradients）を用いてSNNを訓練する方法を示す自己完結型のデモ。
- **LLMによるデータ蒸留:**
    - `evospikenet/distillation.py`: 大規模言語モデル（LLM）を利用して、テストや訓練用の高品質な合成データを生成する機能。柔軟なバックエンドに対応する共通インターフェース設計を採用。
- **本格的なSNN言語モデル (`SpikingEvoSpikeNetLM`):**
    - `snnTorch`をベースとした、実際にスパイクで動作するTransformerベースの言語モデル。
    - `TAS-Encoding`: テキストを時間適応型のスパイク列に変換するエンコーディング層。
    - `ChronoSpikeAttention`: スパイク領域で動作するハイブリッドなアテンション機構。
    - `MetaSTDP` / `AEG`: メタ学習とエネルギー効率を考慮した高度な学習・制御機構。
    - **訓練、評価、チューニング:** モデルの訓練、パープレキシティ評価、ハイパーパラメータ調整までの一連のワークフローをスクリプトで提供 (`examples/train_spiking_evospikenet_lm.py`, `scripts/run_hp_tuning.sh`)。
    - **UIによる可視化:** Web UIから、テキスト生成時の内部スパイク活動とアテンションの重みを可視化可能。

## 3. Web UI
シミュレーションの実行、パラメータ調整、結果の可視化をブラウザからインタラクティブに行うためのWebフロントエンドが利用可能です。特に、**SNN言語モデルの訓練、評価、内部動作の可視化**といった高度な機能もUIから実行・確認できます。以下のスクリプトで起動し、ブラウザで `http://localhost:8050` にアクセスしてください。

```bash
# CPUモードでWeb UIを起動
./scripts/run_frontend_cpu.sh
```

## 4. Dockerを使用した環境構築・実行 (推奨)

本プロジェクトは、Dockerを使用して、CPUまたはGPUの実行環境を簡単に構築できます。

### 前提条件
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (GPUモードを使用する場合)

### 環境のビルド
まず、プロジェクトのルートディレクトリで以下のコマンドを実行し、Dockerイメージをビルドします。
```bash
docker compose build
```

### 実行モードの選択 (CPU / GPU)
`scripts` ディレクトリ内の実行スクリプトは、CPUモード用 (`*_cpu.sh`) とGPUモード用 (`*_gpu.sh`) に分かれています。目的に応じて適切なスクリプトを選択してください。

---

### 開発環境
開発用のコンテナを起動し、インタラクティブなシェルに接続します。

- **GPUモード:**
  ```bash
  ./scripts/run_dev_gpu.sh
  ```
- **CPUモード:**
  ```bash
  ./scripts/run_dev_cpu.sh
  ```
コンテナから出るには `exit` を実行してください。コンテナはバックグラウンドで実行され続けます。完全に停止するには `docker stop evospikenet-dev-gpu` (または `...-cpu`) を実行してください。

---

### テストの実行
`pytest`によるテストスイートを実行します。

- **GPUモード:**
  ```bash
  ./scripts/run_tests_gpu.sh
  ```
- **CPUモード:**
  ```bash
  ./scripts/run_tests_cpu.sh
  ```

---

### サンプルプログラムの実行
`example.py`のサンプルプログラムを実行します。

- **GPUモード:**
  ```bash
  ./scripts/run_prod_gpu.sh
  ```
- **CPUモード:**
  ```bash
  ./scripts/run_prod_cpu.sh
  ```
---

### Web UIの実行
インタラクティブな操作が可能なWeb UIを起動します。

- **GPUモード:**
  ```bash
  ./scripts/run_frontend_gpu.sh
  ```
- **CPUモード:**
  ```bash
  ./scripts/run_frontend_cpu.sh
  ```
起動後、ブラウザで `http://localhost:8050` にアクセスしてください。

## 5. ローカルでの開発環境設定 (上級者向け)

**注意:** この方法は、ローカルマシンに適切なバージョンのPythonとCUDAがインストールされており、パスが正しく設定されていることを前提とします。多くのユーザーにとっては、環境の差異による問題を回避できる**Dockerを使用した実行（セクション4）が推奨されます。**

本フレームワークは、CPUまたはGPU（NVIDIA製）で実行可能です。性能を最大限に引き出すためには、**NVIDIA GPUとCUDA** のセットアップが推奨されます。

1.  **リポジトリのクローンと移動**
2.  **Python仮想環境の作成と有効化**
3.  **依存ライブラリのインストール:**
    - **GPUの場合:** `PyTorch`はCUDAのバージョンに合ったものをインストールする必要があります。
      ```bash
      # PyTorch for CUDA 12.1
      pip install torch --index-url https://download.pytorch.org/whl/cu121
      ```
    - **CPUの場合:**
      ```bash
      pip install torch
      ```
    - **共通の依存ライブラリ:**
      ```bash
      # 本体をインストール
      pip install -e .
      # テスト用の依存関係をインストール
      pip install -e '.[test]'
      ```
4.  **インストールの確認:**
    ```bash
    # GPUでテスト
    DEVICE=cuda pytest
    # CPUでテスト
    DEVICE=cpu pytest
    ```

## 6. 基本的な実行手順

以下は、`EvoSpikeNet`を使用して簡単な2層のSNNモデルを構築し、実行するサンプルコードです。

```python
import torch
import os
from evospikenet.core import LIFNeuronLayer, SynapseMatrixCSR, SNNModel

def run_simple_snn():
    """
    EvoSpikeNetの基本的な使い方を示すサンプル関数。
    """
    # 環境変数からデバイスを選択。なければデフォルトでCUDAかCPUを選択
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. モデルのコンポーネントを定義
    layers = [
        LIFNeuronLayer(num_neurons=784, device=device),
        LIFNeuronLayer(num_neurons=10, device=device)
    ]
    synapses = [
        SynapseMatrixCSR(pre_size=784, post_size=10, connectivity_ratio=0.05, device=device)
    ]

    # 2. SNNモデルを構築
    model = SNNModel(layers=layers, synapses=synapses)
    print("SNNModel created successfully.")

    # 3. ダミーの入力データを生成
    time_steps = 20
    input_size = 784
    input_spikes = torch.randint(0, 2, (time_steps, input_size), dtype=torch.int8, device=device)
    print(f"\nCreated random input spikes with shape: {input_spikes.shape}")

    # 4. シミュレーションを実行
    print("Running simulation...")
    output_spikes = model.forward(input_spikes)
    print("Simulation finished.")

    # 5. 結果を確認
    print(f"Output spikes shape: {output_spikes.shape}")
    print(f"Total spikes in output: {torch.sum(output_spikes)}")

if __name__ == '__main__':
    run_simple_snn()
```

## 7. プロジェクト構成

| パス | 説明 |
| :--- | :--- |
| `evospikenet/` | フレームワークの主要なソースコード。 |
| `frontend/` | DashベースのWeb UIアプリケーションのソースコード。 |
| `tests/` | `pytest`を使用したユニットテスト。 |
| `scripts/` | 開発、テスト、実行を簡略化するためのCPU/GPU別シェルスクリプト群。 |
| `examples/` | フレームワークの具体的な使用方法を示すサンプルプログラム。 |
| `Dockerfile` | プロジェクトの実行環境を定義するDockerイメージの設計図。 |
| `docker-compose.yml` | CPUモード用のDockerサービス定義。 |
| `docker-compose.gpu.yml`| GPUモード用の追加設定ファイル。 |
| `pyproject.toml` | プロジェクトのメタデータとPythonの依存関係を定義する。 |
| `REMAINING_FEATURES.md`| プロジェクトの機能実装状況と今後のロードマップ。 |
| `README.md` | このファイル。プロジェクトの概要、セットアップ手順、使い方などを記述。 |
