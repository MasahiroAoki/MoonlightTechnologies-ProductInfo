# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet - 分散型進化ニューロモーフィックフレームワーク

**最終更新日:** 2025年10月19日

## 1. プロジェクト概要

EvoSpikeNetは、GPUの並列計算能力またはCPUを活用し、脳の可塑性を模倣した、スケーラブルでエネルギー効率の高い「動的スパイクニューラルネットワーク（SNN）」のシミュレーションフレームワークです。浮動小数点演算を排除し、整数演算とスパース計算に特化することで、リアルタイムの時系列データ処理やエッジAI応用を目指します。

## 2. 主な実装済み機能

本フレームワークは、`REMAINING_FEATURES.md`で定義されたフェーズに基づき、以下の主要な機能セットを実装しています。

- **コアSNNエンジン (フェーズ1)**:
    - `LIFNeuronLayer`: 整数演算に基づくLIFニューロン層。
    - `SynapseMatrixCSR`: メモリ効率の良いスパースなシナプス結合。
    - `SNNModel`: 複数の層とシナプスを統合し、シミュレーションを実行するコアモデル。
- **動的進化と可視化 (フェーズ2)**:
    - `STDP`: スパイクタイミング依存可塑性（STDP）によるオンライン学習。
    - `DynamicGraphEvolutionEngine`: シナプスの生成と削除を行う動的グラフ進化。
    - `InsightEngine`: 発火活動のラスタープロットや接続構造グラフなど、ネットワーク内部を可視化。
- **エネルギー駆動型コンピューティング (フェーズ3)**:
    - `EnergyManager`: エネルギーレベルに基づきニューロンの発火を管理し、より生物学的に現実的なシミュレーションを実現。
- **マルチモーダル処理 (テキスト・画像)**:
    - `WordEmbeddingLayer`, `PositionalEncoding`, `RateEncoder`によるテキストデータのスパイク列への変換。
    - `torchvision`を利用した画像処理を行うマルチモーダル言語モデル（`MultiModalEvoSpikeNetLM`）の実装。
- **量子インスパイアード機能 (フェーズ3)**:
    - `EntangledSynchronyLayer`: コンテキストに基づきニューロン群の同期を動的に制御する特殊な層。
    - `HardwareFitnessEvaluator`: ハードウェア性能指標を考慮した進化計算の適応度評価関数。
    - `GraphAnnealingRule`: シミュレーテッドアニーリングを用いてグラフ構造を最適化し、自己組織化を促進するルール。
- **LLMによるデータ蒸留**:
    - `evospikenet/distillation.py`: 大規模言語モデル（LLM）を用いて、テストや訓練用の高品質な合成データを生成する機能。
- **本格的なSNN言語モデル (`SpikingEvoSpikeNetLM`)**:
    - `snnTorch`ベースの、実際にスパイクで動作するTransformerベースの言語モデル。
    - `TAS-Encoding`: テキストを時間適応型のスパイク列に変換するエンコーディング層。
    - `ChronoSpikeAttention`: スパイク領域で動作するハイブリッドなアテンション機構。
- **検索拡張生成 (RAG) 機能**:
- **自己学習 (SSL) 機能**:
    - `evospikenet/ssl.py`: 対照学習（NT-Xent損失）を用いた自己学習層を実装し、ラベルなしデータからの表現学習を可能に。

## 3. Web UI
シミュレーションの実行、パラメータ調整、結果の可視化をブラウザからインタラクティブに行うためのWebフロントエンドが利用可能です。Dockerスクリプトは、UIと共にMilvusデータベースも起動します。

新しいUIは、各機能が個別のページに分割されたマルチページアプリケーションとして再構築されました。`http://localhost:8050`にアクセスすると、ナビゲーションバーから全ての機能ページに移動できます。

以下のコマンドで起動できます。
```bash
# GPUモードでWeb UIとMilvusを起動
./scripts/run_frontend_gpu.sh

# CPUモードでWeb UIとMilvusを起動
./scripts/run_frontend_cpu.sh
```

## 4. Dockerを使用した環境設定と実行 (推奨)

本プロジェクトでは、Dockerを使用してCPUまたはGPUの実行環境を容易に構築できます。**Milvusベクターデータベースもコンテナとして起動**され、完全な開発・実行環境を提供します。

### 前提条件
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2以降, `docker compose` コマンド)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (GPUモード利用時)

### 環境のビルдと実行
まず、プロジェクトのルートディレクトリで以下のコマンドを実行してDockerイメージをビルドします。
```bash
docker compose build
```
次に、`scripts`ディレクトリ内のスクリプトを使用して目的のサービスを起動します。例えば、GPUモードでWeb UIを実行するには、以下のコマンドを使用します。これにより、`frontend`サービスとその依存関係である`milvus-standalone`の両方が起動します。
```bash
./scripts/run_frontend_gpu.sh
```
その後、`http://localhost:8050` にアクセスしてダッシュボードを利用できます。

### その他のコマンド
- **開発環境 (GPU):** `./scripts/run_dev_gpu.sh`
- **テスト実行 (CPU):** `./scripts/run_tests_cpu.sh`

## 5. ローカルでの開発環境設定 (非推奨)

Dockerを使用しない場合、手動でMilvusデータベースを起動し、Pythonの依存関係をインストールする必要があります。環境差異による問題を避けるため、**Dockerの使用（セクション4）を強く推奨します。**

1.  **Milvusの起動:** Dockerまたは他の方法でMilvusを別途起動します。
2.  **依存関係のインストール:**
    ```bash
    # PyTorchのインストール（環境に合わせて調整）
    pip install torch
    # プロジェクトのインストール
    pip install -e .[test]
    ```

## 6. 基本的な実行手順

`EvoSpikeNet`を使用して、簡単な2層SNNモデルを構築・実行するサンプルコードです。

```python
import torch
import os
from evospikenet.core import LIFNeuronLayer, SynapseMatrixCSR, SNNModel

def run_simple_snn():
    """
    EvoSpikeNetの基本的な使い方を示すサンプル関数。
    """
    # 環境変数からデバイスを選択、またはCUDA/CPUにデフォルト設定
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")

    # 1. モデルコンポーネントの定義
    layers = [
        LIFNeuronLayer(num_neurons=784, device=device),
        LIFNeuronLayer(num_neurons=10, device=device)
    ]
    synapses = [
        SynapseMatrixCSR(pre_size=784, post_size=10, connectivity_ratio=0.05, device=device)
    ]

    # 2. SNNモデルの構築
    model = SNNModel(layers=layers, synapses=synapses)
    print("SNNModelが正常に作成されました。")

    # 3. ダミー入力データの生成
    time_steps = 20
    input_size = 784
    input_spikes = torch.randint(0, 2, (time_steps, input_size), dtype=torch.int8, device=device)
    print(f"\nランダムな入力スパイクを生成しました。形状: {input_spikes.shape}")

    # 4. シミュレーションの実行
    print("シミュレーションを実行中...")
    output_spikes = model.forward(input_spikes)
    print("シミュレーションが完了しました。")

    # 5. 結果の確認
    print(f"出力スパイクの形状: {output_spikes.shape}")
    print(f"出力の合計スパイク数: {torch.sum(output_spikes)}")

if __name__ == '__main__':
    run_simple_snn()
```

## 7. プロジェクト構成

| パス | 説明 |
| :--- | :--- |
| `evospikenet/` | フレームワークの主要なソースコード。 |
| `frontend/` | DashベースのWeb UIアプリケーションのソースコード。`app.py`がエントリーポイントで、`pages/`ディレクトリに各機能ページが格納されています。 |
| `tests/` | `pytest`を使用したユニットテスト。 |
| `scripts/` | CPU/GPU用の開発、テスト、実行を簡略化するシェルスクリプト群。 |
| `examples/` | フレームワークの特定用途を示すサンプルプログラム。 |
| `data/` | サンプルデータセットとRAG用の知識ベース (`knowledge_base.json`)。 |
| `Dockerfile` | プロジェクトの実行環境を定義するDockerイメージの設計図。 |
| `docker-compose.yml` | Milvusサービスを含むCPUモード用のDockerサービス定義。 |
| `docker-compose.gpu.yml`| GPUモード用の追加設定ファイル。 |
| `pyproject.toml` | プロジェクトのメタデータとPythonの依存関係を定義する。 |
| `REMAINING_FEATURES.md`| プロジェクトの機能実装状況と将来のロードマップ。 |
| `README.md` | このファイル。プロジェクトの概要、設定手順、使用方法などを記述。 |

## 8. ロードマップ

EvoSpikeNetは、継続的な研究開発を通じて、より高度でエネルギー効率の高いニューロモーフィックコンピューティングの実現を目指します。
現状の開発承認済みの機能は`REMAINING_FEATURES.md`を参照してください。未承認の開発検討項目の詳細は`FEATURE_ANALYSIS.md`を参照してください。
