<!-- Reviewed against source: 2025-12-21. English translation pending. -->
# 分散脳シミュレーション - ノード構成とモデルマッピング

> 実装ノート（アーティファクト）: トレーニングスクリプトが出力する `artifact_manifest.json` と推奨CLIフラグについては `docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Last Updated:** December 12, 2025

## このドキュメントの目的と使い方
- 目的: 分散脳のノード構成、モデルマッピング、ハブ構成を素早く参照できるようにする。
- 対象読者: ノード配備・モデル割当を担当する実装/運用メンバー。
- まず読む順: 概要 → 新しいアーキテクチャ → シミュレーションタイプ/各ノード設定。
- 関連リンク: 実行スクリプトは `examples/run_zenoh_distributed_brain.py`、PFC/Zenoh/Executive詳細は [implementation/PFC_ZENOH_EXECUTIVE.md](implementation/PFC_ZENOH_EXECUTIVE.md)。

EvoSpikeNetの分散脳シミュレーションは、複数のノードが協調して動作する階層的なアーキテクチャを採用しています。各ノードは特定の脳領域をシミュレートし、Zenoh通信プロトコルを介してスパイク信号を交換します。

## 概要

EvoSpikeNetの分散脳シミュレーションは、複数のノードが協調して動作する階層的なアーキテクチャを採用しています。各ノードは特定の脳領域をシミュレートし、Zenoh通信プロトコルを介してスパイク信号を交換します。

## 新しいアーキテクチャ: Sensor Hub と Motor Hub の分離

**2025年12月12日更新**: 運動野とセンサー情報を管理する分類を分離し、より効率的なアーキテクチャを導入。

### Sensor Hub (センサーハブ)
- すべてのセンサー入力（視覚、聴覚、触覚）を統合管理
- センサーデータの前処理と統合を担当
- PFCに統合されたセンサーデータを提供

### Motor Hub (モーターハブ)
- すべての運動出力（軌道制御、小脳協調、PWM制御）を統合管理
- PFCからのコマンドを実際の運動制御に変換
- 複数の運動サブシステムの協調を管理

### 利点
- **並列処理**: センサー入力の同時処理が可能
- **専門化**: 各ハブが専門の機能を担当
- **拡張性**: 新しいセンサー/運動タイプの追加が容易
- **効率性**: センサーと運動の分離により、処理の最適化が可能

### データフロー
## 概要

EvoSpikeNetの分散脳シミュレーションは、複数のノードが協調して動作する階層的なアーキテクチャを採用しています。各ノードは特定の脳領域をシミュレートし、Zenoh通信プロトコルを介してスパイク信号を交換します。
Sensor Hub → PFC → Motor Hub
     ↓         ↓         ↓
  Visual    Compute   Motor-Traj
 Auditory   Lang-Main  Motor-Cereb
            Speech     Motor-PWM
```

## シミュレーションタイプ

分散脳シミュレーションは以下の5つのタイプをサポートしています：

1. **Language Focus** (6プロセス) - 言語処理に特化
2. **Image Focus** (7プロセス) - 視覚処理に特化（Sensor Hub使用）
3. **Audio Focus** (10プロセス) - 聴覚・発声処理に特化（Sensor Hub使用）
4. **Motor Focus** (7プロセス) - 運動制御に特化（Motor Hub使用）
5. **Full Brain** (25プロセス) - 全機能統合（Sensor Hub + Motor Hub）

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

### 2. Image Focus (7プロセス)

視覚処理階層を重点的にカバー。Sensor Hubの下にVisualモジュールを配置し、エッジ検出、形状認識、物体認識の3段階処理。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Compute (計算処理)
  └─ Rank 2: Sensor-Hub (センサー統合管理)
      └─ Rank 3: Visual (メイン視覚処理)
          ├─ Rank 4: Vis-Edge (エッジ検出)
          ├─ Rank 5: Vis-Shape (形状認識)
          └─ Rank 6: Vis-Object (物体認識)
```

**使用モデル:**
- Visual, Vis-Edge, Vis-Shape, Vis-Object: `SpikingEvoVisionEncoder`
- Sensor-Hub: `SimpleLIFNode` (統合管理)
- PFC, Compute: `SpikingEvoTextLM`

---

### 3. Audio Focus (10プロセス)

聴覚・発声処理階層を重点的にカバー。Sensor Hubの下にAuditoryモジュールを配置し、MFCC特徴抽出、音素認識、意味理解の3段階処理。Speechモジュールは独立して配置。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Compute (計算処理)
  ├─ Rank 2: Sensor-Hub (センサー統合管理)
  │   └─ Rank 3: Auditory (メイン聴覚処理)
  │       ├─ Rank 4: Aud-MFCC (MFCC特徴抽出)
  │       ├─ Rank 5: Aud-Phoneme (音素認識)
  │       └─ Rank 6: Aud-Semantic (意味理解)
  └─ Rank 7: Speech (メイン発声処理)
      ├─ Rank 8: Speech-Phoneme (音素生成)
      └─ Rank 9: Speech-Wave (波形合成)
```

**使用モデル:**
- Auditory, Aud-MFCC, Aud-Phoneme, Aud-Semantic: `SpikingEvoAudioEncoder`
- Speech, Speech-Phoneme, Speech-Wave: `SpikingEvoAudioEncoder`
- Sensor-Hub: `SimpleLIFNode` (統合管理)
- PFC, Compute: `SpikingEvoTextLM`

---

### 4. Motor Focus (7プロセス)

運動制御階層を重点的にカバー。Motor Hubの下にMotorモジュールを配置し、軌道制御、小脳協調、PWM制御の3段階処理。

```
Rank 0: PFC (実行制御)
  ├─ Rank 1: Compute (計算処理)
  └─ Rank 2: Motor-Hub (運動統合管理)
      └─ Rank 3: Motor (メイン運動制御)
          ├─ Rank 4: Motor-Traj (軌道制御)
          ├─ Rank 5: Motor-Cereb (小脳協調)
          └─ Rank 6: Motor-PWM (PWM制御)
```

**使用モデル:**
- Motor, Motor-Traj, Motor-Cereb, Motor-PWM: `SimpleLIFNode` (運動制御用)
- Motor-Hub: `SimpleLIFNode` (統合管理)
- PFC, Compute: `SpikingEvoTextLM`

---

### 5. Full Brain (25プロセス)

全ての階層処理を統合した完全な脳シミュレーション。Sensor HubとMotor Hubを使用した新しいアーキテクチャを採用。

```
Rank 0:  PFC (実行制御)
  ├─ Rank 1:  Sensor-Hub (センサー統合管理)
  │   ├─ Rank 2:  Visual (視覚メイン)
  │   └─ Rank 3:  Auditory (聴覚メイン)
  ├─ Rank 4:  Motor-Hub (運動統合管理)
  │   └─ Rank 5:  Motor (運動メイン)
  ├─ Rank 6:  Compute (計算処理)
  ├─ Rank 7:  Lang-Main (言語メイン)
  ├─ Rank 8:  Speech (発声メイン)
  ├─ Rank 9:  Vis-Edge (エッジ検出)
  ├─ Rank 10: Vis-Shape (形状認識)
  ├─ Rank 11: Vis-Object (物体認識)
  ├─ Rank 12: Motor-Traj (軌道制御)
  ├─ Rank 13: Motor-Cereb (小脳協調)
  ├─ Rank 14: Motor-PWM (PWM制御)
  ├─ Rank 15: Aud-MFCC (MFCC特徴)
  ├─ Rank 16: Aud-Phoneme (音素認識)
  ├─ Rank 17: Aud-Semantic (意味理解)
  ├─ Rank 18: Speech-Phoneme (音素生成)
  ├─ Rank 19: Speech-Wave (波形合成)
  ├─ Rank 20: Lang-Embed (埋め込み)
  ├─ Rank 21: Lang-TAS (タスク分解)
  └─ Rank 22: Extra-1 (予備)
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
- **Model Type**: "text"
- **Category**: "LangText"

**使用ノード:**
- Lang-Main, Lang-Embed, Lang-TAS, Lang-RAG, Compute

#### SpikingEvoVisionEncoder (視覚モデル)
- **input_channels**: 1 (グレースケール)
- **output_dim**: 128
- **image_size**: (28, 28) (MNIST互換)
- **time_steps**: 10
- **アーキテクチャ**: 2層CNN + LIF
- **Model Type**: "vision"
- **Category**: "Vision"

**使用ノード:**
- Visual, Vis-Edge, Vis-Shape, Vis-Object

#### SpikingEvoAudioEncoder (音声モデル)
- **input_features**: 13 (MFCC係数)
- **output_neurons**: 128
- **time_steps**: 10
- **Model Type**: "audio"
- **Category**: "Audio"

**使用ノード:**
- Auditory, Aud-MFCC, Aud-Phoneme, Aud-Semantic
- Speech, Speech-Phoneme, Speech-Wave

#### SpikingEvoMultiModalLM (マルチモーダルモデル)
- **text_vocab_size**: 30522
- **vision_input_channels**: 1
- **audio_input_features**: 13
- **d_model**: 128
- **n_heads**: 4
- **time_steps**: 10
- **Model Type**: "multimodal"
- **Category**: "MultiModal"

**使用ノード:**
- 統合タスク用 (将来的拡張)

#### SimpleLIFNode (汎用スパイキングニューロン)
- **size**: 128
- **基本的なLIF (Leaky Integrate-and-Fire) ニューロンレイヤー**
- **Model Type**: N/A (汎用)
- **Category**: N/A

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
python tests/test_node_types.py
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
