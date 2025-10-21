# EvoSpikeNet: データハンドリングガイド

**最終更新日:** 2025年10月11日

このドキュメントは、`EvoSpikeNet`フレームワークで使用する様々なAIデータ（スパイクデータ、テキストコーパス、RAG知識ベース、マルチモーダルデータセット）の作成、フォーマット、および検証方法について詳述します。

---

## 1. スパイクデータの生成とフォーマット

SNNモデルへの直接入力となるスパイクデータの仕様と生成方法です。

### 1.1. データ仕様
- **フォーマット**: `torch.Tensor`
- **形状**: `(time_steps, num_input_neurons)` の2次元テンソル。バッチ処理の場合は `(batch_size, time_steps, num_input_neurons)` の3次元テンソル。
- **データ型**: `torch.int8`
- **値**: `0` (スパイクなし) または `1` (スパイクあり)。

### 1.2. データ生成スクリプト
テスト用の人工的なスパイクデータは `scripts/generate_spike_data.py` で生成可能です。ポアソン分布に基づき、指定された発火率のスパイクデータを生成します。

```bash
python scripts/generate_spike_data.py --num-samples 100 --time-steps 200
```

---

## 2. テキストコーパスのデータソース

言語モデル (`EvoSpikeNetLM`, `SpikingEvoSpikeNetLM`) の訓練には、テキストベースのコーパスが必要です。`evospikenet/dataloaders.py` モジュールが、様々なソースからのデータ読み込みをサポートします。

- **`WikipediaLoader`**: Wikipediaの記事をテキストデータとして読み込みます。
- **`AozoraBunkoLoader`**: 青空文庫のHTMLページから本文テキストを抽出します。
- **`LocalFileLoader`**: ローカルのテキストファイルを読み込みます。

これらのローダーは、`examples/train_evospikenet_lm.py` スクリプトから利用できます。

---

## 3. RAG知識ベースのデータハンドリング

検索拡張生成（RAG）機能は、外部知識を格納したベクターデータベース（Milvus）を利用します。知識ベース内のデータは、`evospikenet/rag_milvus.py`モジュールを通じて直接Milvus上で管理されます。

### 3.1. データ構造
Milvusに格納される各ドキュメントは、以下のフィールドで構成されます。
- `id`: 一意の識別子（自動採番）。
- `embedding`: テキストから生成された384次元のベクトル。
- `text`: ドキュメントの本文。
- `source`: ドキュメントの出所を示す文字列（例: "wikipedia", "user_input"）。

### 3.2. データの管理 (UI経由)
ダッシュボードの「RAG System」ページ内にある「Data CURD」タブから、知識ベース内のデータを包括的に管理できます。
- **閲覧 (Read):** Milvusに保存されている全てのドキュメントがテーブル形式で表示されます。
- **追加 (Create):** 「Add Row」ボタンで新しい行を追加し、テキストとソースを入力することで、新しいドキュメントを知識ベースに登録できます。
- **更新 (Update):** テーブル内のセルを直接編集することで、既存のドキュメントの内容を更新できます。
- **削除 (Delete):** テーブルで特定の行を選択し、「Delete Selected Row」ボタンを押すことで、ドキュメントを削除できます。

このインターフェースにより、`knowledge_base.json`のような中間ファイルを介さず、データベースと直接やり取りしてデータを管理することが可能です。

---

## 4. マルチモーダルデータセットのフォーマット

マルチモーダルモデル (`MultiModalEvoSpikeNetLM`) は、画像とそれに対応するキャプション（テキスト）のペアで学習します。フレームワークは以下のデータ構造を想定しています。

### 4.1. ディレクトリ構造
```
data/
└── multi_modal_dataset/
    ├── images/
    │   ├── image_0.png
    │   ├── image_1.jpg
    │   └── ...
    └── annotations.json
```

### 4.2. `annotations.json` のフォーマット
このJSONファイルは、画像ファイルへのパスと、それに対応するキャプションを記述したオブジェクトのリストです。

- **フォーマット**:
  ```json
  [
    {
      "image_path": "images/image_0.png",
      "caption": "夕暮れ時の海岸線に沿って立つ灯台。"
    },
    {
      "image_path": "images/image_1.jpg",
      "caption": "雪を頂いた山々を背景にした静かな湖。"
    }
  ]
  ```
`examples/train_multi_modal_lm.py` スクリプトは、この`annotations.json`を読み込み、画像とキャプションをペアとしてモデルに提供します。