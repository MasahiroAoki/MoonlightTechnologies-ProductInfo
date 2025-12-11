# シミュレーションデータ記録・解析機能

**作成日:** 2025年12月6日  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki

## 概要

分散脳シミュレーション実行時のデータ記録・解析システムが追加されました。

## 新規追加ファイル

### コアモジュール
- `evospikenet/sim_recorder.py` - データ記録システム
- `evospikenet/sim_analyzer.py` - データ解析ツール

### ドキュメント
- `docs/SIMULATION_RECORDING_GUIDE.md` - 詳細ガイド
- `examples/example_simulation_recording.py` - 使用例

## クイックスタート

### 1. 記録を有効化してシミュレーション実行

```bash
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording
```

### 2. 記録データを解析

```bash
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234
```

### 3. サンプルスクリプトを実行

```bash
python examples/example_simulation_recording.py
```

## 記録されるデータ

- ✅ **スパイクデータ**: 各層からのスパイク列
- ✅ **膜電位データ**: ニューロンの膜電位（オプション）
- ✅ **重みデータ**: ネットワークの重み行列（オプション）
- ✅ **制御データ**: ノードの状態遷移

## 主な機能

### 記録機能
- オプションで有効/無効を切り替え
- サブサンプリングでストレージ削減
- GZIP圧縮対応
- バッファリングで効率的な書き込み

### 解析機能
- 発火率の自動計算
- スパイクラスタープロット生成
- 発火率の時系列プロット
- サマリーレポート自動生成

## 使用例

```python
from evospikenet.sim_recorder import SimulationRecorder, RecorderConfig

# 記録設定
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=True,
    session_name="my_experiment"
)

# 記録開始
with SimulationRecorder(config) as recorder:
    # シミュレーション実行
    for step in range(1000):
        # スパイクを記録
        recorder.record_spike_data(
            node_id="pfc-0",
            layer_name="lif",
            spikes=output_spikes
        )
```

## 詳細情報

詳しくは `docs/SIMULATION_RECORDING_GUIDE.md` を参照してください。
