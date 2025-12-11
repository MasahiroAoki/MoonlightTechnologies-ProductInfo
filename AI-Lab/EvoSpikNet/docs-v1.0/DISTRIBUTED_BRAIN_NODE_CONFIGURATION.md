# 分散脳シミュレーション - ノード構成とモデルマッピング

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Last Updated:** December 10, 2025

## 概要

EvoSpikeNetの分散脳シミュレーションは、複数のノードが協調して動作する階層的なアーキテクチャを採用しています。各ノードは特定の脳領域をシミュレートし、Zenoh通信プロトコルを介してスパイク信号を交換します。

## シミュレーションタイプ

分散脳シミュレーションは以下の5つのタイプをサポートしています：

1. **Language Focus** (10プロセス) - 言語処理に特化
2. **Image Focus** (11プロセス) - 視覚処理に特化
3. **Audio Focus** (12プロセス) - 聴覚・発声処理に特化
4. **Motor Focus** (11プロセス) - 運動制御に特化
5. **Full Brain** (21プロセス) - 全機能統合

---

## ノード階層構成

### 1. Language Focus (10プロセス)

言語処理を中心とした構成。埋め込みベクトル生成、タスク分解、RAG検索などの高度な言語機能を含みます。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Visual (視覚入力)
  ├─ Rank 2: Motor (運動出力)
  ├─ Rank 3: Compute (計算処理)
  ├─ Rank 4: Lang-Main (メイン言語処理)
  ├─ Rank 5: Auditory (聴覚入力)
  ├─ Rank 6: Speech (発声出力)
  ├─ Rank 7: Lang-Embed (埋め込みベクトル生成)
  ├─ Rank 8: Lang-TAS (タスク分解)
  └─ Rank 9: Lang-RAG (RAG検索)
```

**使用モデル:**
- PFC: `SimpleLIFNode`
- Lang-Main, Lang-Embed, Lang-TAS, Lang-RAG: `SpikingEvoTextLM`
- Visual: `SpikingEvoVisionEncoder`
- Auditory, Speech: `SpikingEvoAudioEncoder`
- Motor: `SimpleLIFNode` (フォールバック)
- Compute: `SpikingEvoTextLM`

---

### 2. Image Focus (11プロセス)

視覚処理階層を重点的にカバー。エッジ検出、形状認識、物体認識の3段階処理。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Visual (メイン視覚処理)
  ├─ Rank 2: Motor (運動出力)
  ├─ Rank 3: Compute (計算処理)
  ├─ Rank 4: Lang-Main (言語処理)
  ├─ Rank 5: Auditory (聴覚入力)
  ├─ Rank 6: Speech (発声出力)
  ├─ Rank 7: Vis-Edge (エッジ検出)
  ├─ Rank 8: Vis-Shape (形状認識)
  ├─ Rank 9: Vis-Object (物体認識)
  └─ Rank 10: Extra-1 (予備)
```

**使用モデル:**
- Visual, Vis-Edge, Vis-Shape, Vis-Object: `SpikingEvoVisionEncoder`
- その他は Language Focus と同様

---

### 3. Audio Focus (12プロセス)

聴覚・発声処理階層を重点的にカバー。MFCC特徴抽出、音素認識、意味理解、音素生成、波形合成の5段階処理。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Visual (視覚入力)
  ├─ Rank 2: Motor (運動出力)
  ├─ Rank 3: Compute (計算処理)
  ├─ Rank 4: Lang-Main (言語処理)
  ├─ Rank 5: Auditory (メイン聴覚処理)
  ├─ Rank 6: Speech (メイン発声処理)
  ├─ Rank 7: Aud-MFCC (MFCC特徴抽出)
  ├─ Rank 8: Aud-Phoneme (音素認識)
  ├─ Rank 9: Aud-Semantic (意味理解)
  ├─ Rank 10: Speech-Phoneme (音素生成)
  └─ Rank 11: Speech-Wave (波形合成)
```

**使用モデル:**
- Auditory, Aud-MFCC, Aud-Phoneme, Aud-Semantic: `SpikingEvoAudioEncoder`
- Speech, Speech-Phoneme, Speech-Wave: `SpikingEvoAudioEncoder`
- その他は Language Focus と同様

---

### 4. Motor Focus (11プロセス)

運動制御階層を重点的にカバー。軌道計画、小脳による協調運動、PWM制御の3段階処理。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Visual (視覚入力)
  ├─ Rank 2: Motor (メイン運動制御)
  ├─ Rank 3: Compute (計算処理)
  ├─ Rank 4: Lang-Main (言語処理)
  ├─ Rank 5: Auditory (聴覚入力)
  ├─ Rank 6: Speech (発声出力)
  ├─ Rank 7: Motor-Traj (軌道計画)
  ├─ Rank 8: Motor-Cereb (小脳・協調運動)
  ├─ Rank 9: Motor-PWM (PWM制御)
  └─ Rank 10: Extra-1 (予備)
```

**使用モデル:**
- Motor, Motor-Traj, Motor-Cereb, Motor-PWM: `SimpleLIFNode` (フォールバック)
  - 注: Motor専用モデル未統合、将来的に`MotorModule`を使用予定
- その他は Language Focus と同様

---

### 5. Full Brain (21プロセス)

全ての階層処理を統合した完全な脳シミュレーション。

```
Rank 0:  PFC (実行制御)
  ├─ Rank 1:  Visual (視覚メイン)
  ├─ Rank 2:  Motor (運動メイン)
  ├─ Rank 3:  Compute (計算処理)
  ├─ Rank 4:  Lang-Main (言語メイン)
  ├─ Rank 5:  Auditory (聴覚メイン)
  ├─ Rank 6:  Speech (発声メイン)
  ├─ Rank 7:  Vis-Edge (エッジ検出)
  ├─ Rank 8:  Vis-Shape (形状認識)
  ├─ Rank 9:  Vis-Object (物体認識)
  ├─ Rank 10: Motor-Traj (軌道計画)
  ├─ Rank 11: Motor-Cereb (小脳)
  ├─ Rank 12: Motor-PWM (PWM制御)
  ├─ Rank 13: Aud-MFCC (MFCC特徴)
  ├─ Rank 14: Aud-Phoneme (音素認識)
  ├─ Rank 15: Aud-Semantic (意味理解)
  ├─ Rank 16: Speech-Phoneme (音素生成)
  ├─ Rank 17: Speech-Wave (波形合成)
  ├─ Rank 18: Lang-Embed (埋め込み)
  ├─ Rank 19: Lang-TAS (タスク分解)
  └─ Rank 20: Extra-1 (予備)
```

---

## モデルマッピング

### ノードタイプ正規化

サブノードは`_get_base_module_type()`メソッドによってベースタイプに正規化され、対応するSpikingモデルが割り当てられます。

```python
# 言語系サブノード → lang-main
"lang-embed", "lang-rag", "lang-tas", "compute" 
→ SpikingEvoTextLM

# 視覚系サブノード → visual
"vis-edge", "vis-shape", "vis-object"
→ SpikingEvoVisionEncoder

# 聴覚系サブノード → audio
"aud-mfcc", "aud-phoneme", "aud-semantic"
→ SpikingEvoAudioEncoder

# 発声系サブノード → audio
"speech-phoneme", "speech-wave"
→ SpikingEvoAudioEncoder

# 運動系サブノード → motor
"motor-traj", "motor-cereb", "motor-pwm"
→ SimpleLIFNode (フォールバック、将来的にMotorModule使用予定)
```

### モデル詳細

#### SpikingEvoTextLM (言語モデル)
- **vocab_size**: 30522 (BERT互換)
- **d_model**: 128
- **n_heads**: 4
- **num_transformer_blocks**: 2
- **time_steps**: 10

**使用ノード:**
- Lang-Main, Lang-Embed, Lang-TAS, Lang-RAG, Compute

#### SpikingEvoVisionEncoder (視覚モデル)
- **input_channels**: 1 (グレースケール)
- **output_dim**: 128
- **image_size**: (28, 28) (MNIST互換)
- **time_steps**: 10
- **アーキテクチャ**: 2層CNN + LIF

**使用ノード:**
- Visual, Vis-Edge, Vis-Shape, Vis-Object

#### SpikingEvoAudioEncoder (音声モデル)
- **input_features**: 13 (MFCC係数)
- **output_neurons**: 128
- **time_steps**: 10

**使用ノード:**
- Auditory, Aud-MFCC, Aud-Phoneme, Aud-Semantic
- Speech, Speech-Phoneme, Speech-Wave

#### SimpleLIFNode (汎用スパイキングニューロン)
- **size**: 128
- **基本的なLIF (Leaky Integrate-and-Fire) ニューロンレイヤー**

**使用ノード:**
- PFC
- Motor系 (Motor, Motor-Traj, Motor-Cereb, Motor-PWM)

---

## ノード起動方法

### 基本コマンド

```bash
python examples/run_zenoh_distributed_brain.py \
  --node-id <node-id> \
  --module-type <module-type> \
  [--model-artifact-id <artifact-id>]
```

### 起動例

#### 1. PFCノード起動
```bash
python examples/run_zenoh_distributed_brain.py \
  --node-id pfc-0 \
  --module-type pfc
```

#### 2. 言語メインノード起動（学習済みモデル使用）
```bash
python examples/run_zenoh_distributed_brain.py \
  --node-id lang-main-4 \
  --module-type lang-main \
  --model-artifact-id 236d1796-547b-4b74-9f67-02e506c1a706
```

#### 3. 視覚サブノード起動
```bash
python examples/run_zenoh_distributed_brain.py \
  --node-id vis-object-9 \
  --module-type vis-object
```

#### 4. 聴覚サブノード起動
```bash
python examples/run_zenoh_distributed_brain.py \
  --node-id aud-mfcc-13 \
  --module-type aud-mfcc
```

---

## モデルロード動作

### 1. 明示的なアーティファクトID指定時

```
1. --model-artifact-id で指定されたアーティファクトを検索
2. セッションIDを解決
3. APIからconfig.jsonとweights.pthをダウンロード
4. モデルを初期化して重みをロード
5. 言語モデルの場合、tokenizerもロード
```

### 2. アーティファクトID未指定時

```
1. APIから最新のモデルセッションを取得
2. 最新セッションのアーティファクトをダウンロード
3. モデルを初期化して重みをロード
```

### 3. API接続失敗時

```
1. デフォルトパラメータでモデルを初期化
2. ランダム重みで動作（学習なし）
3. 言語モデルの場合、デフォルトBERTトークナイザーを使用
```

---

## FPGAセーフティコントローラー統合

全てのノードはFPGAセーフティコントローラーと統合され、2秒ごとにハートビートを送信します。

### ハートビート送信タイミング

1. セッション解決中
2. モデルロード前
3. config.json ダウンロード後
4. モデル初期化後
5. weights.pth ダウンロード後
6. 重みロード後

これにより、長時間のモデルロード中にもウォッチドッグタイムアウトが発生しません。

---

## ノード検証

### テストスクリプト

全ノードタイプのモデル初期化を検証：

```bash
python test_node_types.py
```

**検証項目:**
- Lang系: lang-main, lang-embed, lang-rag, lang-tas, compute
- Visual系: visual, vision, vis-main
- Audio系: audio, auditory, aud-main, speech
- Multimodal: multimodal

### 検証結果

```
✅ lang-main: SpikingEvoTextLM
✅ lang-embed: SpikingEvoTextLM
✅ lang-rag: SpikingEvoTextLM
✅ lang-tas: SpikingEvoTextLM
✅ compute: SpikingEvoTextLM
✅ visual: SpikingEvoVisionEncoder
✅ vision: SpikingEvoVisionEncoder
✅ vis-main: SpikingEvoVisionEncoder
✅ audio: SpikingEvoAudioEncoder
✅ auditory: SpikingEvoAudioEncoder
✅ aud-main: SpikingEvoAudioEncoder
✅ speech: SpikingEvoAudioEncoder
✅ multimodal: SpikingEvoMultiModalLM

Passed: 13/13
```

---

## トラブルシューティング

### 1. FPGAウォッチドッグタイムアウト

**症状:** 50秒以上のモデルロード中にタイムアウト

**解決策:**
- `AutoModelSelector.get_model()`にsafety_controllerを渡す
- モデルロード中の6箇所でハートビートを送信

### 2. モデル初期化エラー (TypeError)

**症状:** `name 'd_model' is not defined`

**解決策:**
- `_load_from_api()`でTypeErrorをキャッチ
- デフォルトパラメータでフィルタリング
- フィルタ済みパラメータで再初期化

### 3. API接続エラー

**症状:** `Failed to resolve 'api'`

**解決策:**
- Docker環境: `api`ホスト名を使用
- ローカル環境: `localhost`を使用
- 環境変数`API_HOST`で制御可能

### 4. トークナイザーロード失敗

**症状:** 言語モデルでトークナイザーが見つからない

**解決策:**
- `_load_tokenizer_from_session()`失敗時に自動フォールバック
- デフォルトBERTトークナイザー(`bert-base-uncased`)を使用

---

## 今後の開発

### 優先度: 高

1. **MotorModule統合**
   - `MotorModule`を`AutoModelSelector`に統合
   - Motor系サブノードで専用モデルを使用

2. **モデル保存・読込の強化**
   - トークナイザーの自動保存
   - config.jsonの完全性チェック

### 優先度: 中

3. **階層的モデル選択**
   - サブノードごとに異なるモデルを使用可能に
   - 視覚階層: Edge→Shape→Objectで異なるモデル

4. **動的ノード発見**
   - ノード発見サービスの強化
   - 動的なトポロジー変更対応

### 優先度: 低

5. **分散学習サポート**
   - 複数ノード間での協調学習
   - 勾配同期機構

---

## config.jsonの構造

各モデルタイプで保存される`config.json`の詳細な構造です。

### SpikingEvoTextLM (言語モデル)

```json
{
    "vocab_size": 30522,
    "d_model": 128,
    "n_heads": 4,
    "num_transformer_blocks": 2,
    "time_steps": 16,
    "neuron_type": "lif"
}
```

**パラメータ説明:**
- `vocab_size`: 語彙サイズ（BERT互換は30522）
- `d_model`: Transformerの隠れ層次元
- `n_heads`: マルチヘッドアテンションのヘッド数
- `num_transformer_blocks`: Transformerブロックの層数
- `time_steps`: スパイキングニューロンの時間ステップ数
- `neuron_type`: ニューロンタイプ（"lif" または "alif"）

**保存スクリプト:**
```bash
python examples/train_spiking_evospikenet_lm.py \
  --run-name lang_20251209_132845 \
  --upload-to-db
```

**アーティファクト:**
- `config.json` - モデル設定
- `spiking_lm.pth` - モデル重み
- `spiking_lm_tokenizer.zip` - BERTトークナイザー

---

### SpikingEvoVisionEncoder (視覚モデル)

```json
{
    "input_channels": 1,
    "output_dim": 128,
    "time_steps": 20,
    "num_classes": 10,
    "dataset": "mnist",
    "image_size": [28, 28]
}
```

**パラメータ説明:**
- `input_channels`: 入力画像のチャンネル数（1=グレースケール、3=RGB）
- `output_dim`: 出力特徴ベクトルの次元
- `time_steps`: スパイキングニューロンの時間ステップ数
- `num_classes`: 分類クラス数（分類タスクの場合）
- `dataset`: 学習に使用したデータセット名
- `image_size`: 入力画像サイズ [高さ, 幅]

**保存スクリプト:**
```bash
python examples/train_vision_encoder.py \
  --run-name vision_20251208_164304 \
  --dataset mnist \
  --upload-to-db
```

**アーティファクト:**
- `config.json` - モデル設定
- `vision_encoder.pth` - エンコーダー重み
- `vision_classifier.pth` - 分類器重み（オプション）

---

### SpikingEvoAudioEncoder (音声モデル)

```json
{
    "input_features": 13,
    "output_neurons": 128,
    "time_steps": 10
}
```

**パラメータ説明:**
- `input_features`: 入力特徴量の次元（通常はMFCC係数数）
- `output_neurons`: 出力ニューロン数
- `time_steps`: スパイキングニューロンの時間ステップ数

**保存スクリプト:**
```bash
python examples/train_audio_modal_lm.py \
  --run-name audio_20251208_164638 \
  --upload-to-db
```

**アーティファクト:**
- `config.json` - モデル設定
- `audio_encoder.pth` - モデル重み

---

### config.jsonの使用フロー

#### 1. モデル保存時

```python
# 学習スクリプト内
config_to_save = {
    'vocab_size': tokenizer.vocab_size,
    'd_model': args.d_model,
    'n_heads': args.n_heads,
    'num_transformer_blocks': args.num_blocks,
    'time_steps': args.time_steps,
    'neuron_type': args.neuron_type,
}

with open('config.json', 'w') as f:
    json.dump(config_to_save, f, indent=4)
```

#### 2. モデルロード時（AutoModelSelector）

```python
# config.jsonをダウンロード
with open(config_path, 'r') as f:
    config = json.load(f)

# deviceを追加
config['device'] = 'cpu'

# モデル初期化
try:
    model = SpikingEvoTextLM(**config)
except TypeError as e:
    # パラメータ不一致時はフィルタリング
    default_params = _get_default_params(task_type)
    filtered_config = {k: v for k, v in config.items() 
                      if k in default_params or k == 'device'}
    model = SpikingEvoTextLM(**filtered_config)
```

#### 3. エラーハンドリング

`config.json`に余分なパラメータが含まれている場合：

```python
# 例: config.jsonに"aeg_module"などモデルが受け付けないパラメータがある
{
    "vocab_size": 30522,
    "d_model": 128,
    "aeg_module": {...}  # SpikingEvoTextLMは受け付けない
}

# TypeError発生 → デフォルトパラメータでフィルタリング
filtered_config = {
    "vocab_size": 30522,
    "d_model": 128,
    "time_steps": 10,  # デフォルト値で補完
    "device": "cpu"
}
```

---

## 参考資料

- [分散脳システム概要](DISTRIBUTED_BRAIN_SYSTEM.md)
- [ノード発見システム](ADVANCED_NODE_DISCOVERY.md)
- [データハンドリング](DATA_HANDLING.md)
- [SDKクイックスタート](SDK_QUICKSTART.md)

---

**最終更新:** 2025年12月10日  
**バージョン:** v1.1
