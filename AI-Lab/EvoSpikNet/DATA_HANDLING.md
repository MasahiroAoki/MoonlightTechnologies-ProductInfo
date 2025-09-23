# EvoSpikeNet: データハンドリングガイド

このドキュメントは、`EvoSpikeNet`フレームワークで使用するAIのデータ（スパイクデータ）の作成、出力、および検証方法について詳述します。

## 1. スパイクデータの主なデータソース

スパイクデータは、様々なソースから取得または生成することができます。以下に主な例を挙げます。

### 1.1. アルゴリズムによる生成
テストや特定の機能検証のために、アルゴリズムを用いて人工的なスパイクデータを生成する方法です。
- **ランダムデータ**: ポアソン分布などを用いて、特定の発火率を持つランダムなスパイクを生成します。ネットワークの基本的な動作検証に有用です。
- **人工的なパターン**: 特定のニューロン群が同期して発火するパターンなど、意図したスパイクパターンを設計します。ネットワークの学習能力やパターン認識能力のテストに用います。

### 1.2. 既存データからの変換
一般的なデータをSNNで扱えるスパイクデータに変換するアプローチです。
- **画像データ (例: MNIST)**: 静止画像は、各ピクセルの輝度を一定期間内のニューロンの発火率に変換する「レートエンコーディング」によってスパイクデータに変換できます。明るいピクセルほど高い頻度で発火します。
- **音声・時系列データ**: 音声波形やセンサーの時系列データは、その値の変化や大きさをスパイクの発火タイミングや頻度にエンコードすることで、スパイクデータに変換できます。

### 1.3. ニューロモーフィックセンサーからの直接入力
生物の神経系を模倣して作られたセンサーは、ネイティブなスパイクデータを出力します。
- **イベントベースカメラ (DVS)**: 従来のカメラのようにフレームを出力するのではなく、各ピクセルが輝度の変化を検知したタイミングで「イベント（スパイク）」を非同期に出力します。これにより、非常に高速で冗長性の少ない視覚情報が得られます。

---

## 2. データの作成方法

モデルへの入力データは、特定のフォーマットに従ったスパイクデータである必要があります。

### データ仕様

- **フォーマット**: `torch.Tensor`
- **形状 (Shape)**: `(time_steps, num_input_neurons)` の2次元テンソル。
- **データ型 (dtype)**: `torch.int8`
- **デバイス (Device)**: `'cuda'`
- **値 (Values)**: `0` または `1` のバイナリ値。

### データ生成スクリプト

プロジェクトには、テストや実験に使用できるサンプルのスパイクデータを生成するためのスクリプト `scripts/generate_spike_data.py` が含まれています。

#### 使い方

このスクリプトは、コマンドラインから実行します。以下の引数で生成するデータセットの特性をカスタマイズできます。

```bash
python scripts/generate_spike_data.py --num-samples 1000 --time-steps 250 --firing-rate 15 --output-file my_dataset.pt
```

**主な引数:**
- `--num-samples`: 生成するデータサンプルの数。
- `--time-steps`: 各サンプルのタイムステップ数。
- `--num-neurons`: 各タイムステップのニューロン数。
- `--firing-rate`: ニューロンの平均発火率 (Hz)。ポアソン分布に基づきスパイクが生成されます。
- `--output-file`: 出力ファイル名。`.pt` 形式で保存されます。
- `--no-cuda`: このフラグを立てると、CPU上でデータを生成・保存します。

#### 生成されるデータ

スクリプトは、指定された `output-file` にPyTorchテンソルを保存します。データは以下のキーを持つ辞書形式です。
- `data`: スパイクデータ本体。形状は `(num_samples, time_steps, num_neurons)`。
- `labels`: 各サンプルに対応するダミーのラベル。形状は `(num_samples,)`。

### データの読み込み方法

スクリプトによって生成されたデータは、`torch.load()` を使って簡単に読み込むことができます。

```python
import torch

# CUDAが利用可能かチェック
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# データを読み込み、適切なデバイスに配置
dataset = torch.load('my_dataset.pt', map_location=device)

# データを取得
spike_data = dataset['data']
labels = dataset['labels']

print("読み込んだデータセット:")
print(f"  - データ形状: {spike_data.shape}")
print(f"  - ラベル形状: {labels.shape}")
print(f"  - デバイス: {spike_data.device}")

# SNNモデルで使用するために、個々のサンプルを取り出す
# この例では、モデルは (time_steps, num_neurons) の形状を期待するため、
# バッチの次元を取り除く必要がある
first_sample = spike_data[0]
print(f"\nモデルに渡す単一サンプルの形状: {first_sample.shape}")
```

## 3. データの出力方法

モデルのシミュレーション結果として、出力層からのスパイクデータが得られます。

### 出力データの仕様
出力データは、入力データと基本的に同じ仕様です。
- **フォーマット**: `torch.Tensor`
- **形状 (Shape)**: `(time_steps, num_output_neurons)`
- **データ型 (dtype)**: `torch.int8`
- **デバイス (Device)**: `'cuda'`
- **値 (Values)**: `0` または `1`

### サンプルコード
```python
# SNNModelのインスタンスが 'model' という変数に格納されていると仮定
# 'input_spikes' はセクション2で作成したデータ
output_spikes = model.forward(input_spikes)
print("出力データの形状:", output_spikes.shape)
```

## 4. データの検証手順
モデルにデータを渡す前に、そのデータが正しい形式であることを検証することが重要です。

### 検証チェックリスト
1.  **型チェック**: データは `torch.Tensor` であるか？
2.  **次元チェック**: テンソルの次元は2であるか？
3.  **形状チェック**: テンソルの2番目の次元がモデルの入力ニューロン数と一致しているか？
4.  **データ型チェック**: データ型は `torch.int8` であるか？
5.  **デバイスチェック**: テンソルは `'cuda'` デバイス上にあるか？
6.  **値チェック**: テンソルの値はすべて0または1であるか？

### 検証用サンプル関数
```python
import torch
from evospikenet.core import SNNModel

def validate_input_spikes(input_tensor: torch.Tensor, model: SNNModel) -> bool:
    """
    入力スパイクテンソルがSNNモデルに対して有効か検証します。
    """
    try:
        # 1. 型チェック
        assert isinstance(input_tensor, torch.Tensor), "Input is not a torch.Tensor"

        # 2. 次元チェック
        assert input_tensor.dim() == 2, f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions"

        # 3. 形状チェック (入力層のニューロン数と照合)
        expected_neurons = model.layers[0].num_neurons
        actual_neurons = input_tensor.shape[1]
        assert actual_neurons == expected_neurons, \
            f"Input neuron count mismatch. Model expects {expected_neurons}, but tensor has {actual_neurons}"

        # 4. データ型チェック
        assert input_tensor.dtype == torch.int8, f"Input dtype must be torch.int8, but got {input_tensor.dtype}"

        # 5. デバイスチェック
        # model.parameters()から最初のパラメータのデバイスを取得してモデル全体のデバイスとみなす
        model_device = next(model.parameters()).device
        assert input_tensor.device == model_device, \
            f"Device mismatch. Model is on {model_device}, but tensor is on {input_tensor.device}"

        # 6. 値チェック (0か1のみ)
        assert torch.all((input_tensor == 0) | (input_tensor == 1)), "Input tensor values must be binary (0 or 1)"

        print("✅ Validation successful: Input tensor is valid for the SNN model.")
        return True

    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        return False
```

---

## 5. テキストコーパスのデータソース

言語モデル (`EvoSpikeNetLM`) の訓練には、テキストベースのコーパスが必要です。`evospikenet/dataloaders.py`モジュールは、様々なソースからテキストデータを統一的に読み込むためのフレームワークを提供します。

### 5.1. 基本設計

-   **`CorpusLoader` (基底クラス):**
    すべてのデータローダーが継承する抽象基底クラスです。`load(**kwargs)`メソッドを定義し、具体的なデータソースからのデータ読み込みロジックはサブクラスに委ねられます。

### 5.2. 実装済みローダー

-   **`WikipediaLoader`:**
    -   **目的:** Pythonの`wikipedia-api`ライブラリを使用して、指定されたWikipediaの記事を取得します。
    -   **使用法:** `load(page_title="記事のタイトル")`のように、記事のタイトルを渡して呼び出します。
-   **`AozoraBunkoLoader`:**
    -   **目的:** 青空文庫のHTMLページから本文テキストを抽出します。`requests`と`BeautifulSoup4`ライブラリを利用してWebスクレイピングを行います。
    -   **処理:** 取得したHTMLから、ルビ（読み仮名）やその他の不要なHTMLタグを除去し、クリーンなテキストデータのみを返します。
    -   **使用法:** `load(url="作品のURL")`のように、青空文庫の作品ページのURLを渡して呼び出します。
-   **`LocalFileLoader`:**
    -   **目的:** ローカルファイルシステム上のテキストファイル (`.txt`) を読み込みます。
    -   **使用法:** `load(file_path="ファイルへのパス")`のように、ファイルパスを渡して呼び出します。
