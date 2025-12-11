# シミュレーションデータ記録・解析ガイド

**作成日**: 2025-12-06  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**対象**: EvoSpikeNet Zenoh分散脳シミュレーション

## 概要

分散脳シミュレーション実行時に、以下のデータを記録・解析できるシステムを実装しました：

1. **スパイクデータ**: 各ニューロン層からのスパイク列
2. **膜電位データ**: ニューロンの膜電位（オプション）
3. **重みデータ**: ネットワークの重み行列スナップショット（オプション）
4. **制御データ**: ノードの状態遷移、タスク実行状況

## クイックスタート

### 記録有効化してシミュレーション実行

```bash
# 基本的な記録（スパイク+制御データ）
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording

# 全データを記録（膜電位+重みも含む）
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --record-membrane \
    --record-weights \
    --session-name my_experiment_001
```

### 記録データの解析

```bash
# 自動解析（レポート+グラフ生成）
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234

# プロット生成をスキップ
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234 --no-plots
```

## 詳細ガイド

### 1. 記録オプション

#### コマンドライン引数

| 引数                 | 説明                 | デフォルト                 |
| -------------------- | -------------------- | -------------------------- |
| `--enable-recording` | 記録を有効化         | False（無効）              |
| `--record-spikes`    | スパイクデータを記録 | True                       |
| `--record-membrane`  | 膜電位を記録         | False                      |
| `--record-weights`   | 重み行列を記録       | False                      |
| `--record-control`   | 制御状態を記録       | True                       |
| `--recording-dir`    | 記録保存ディレクトリ | `./sim_recordings`         |
| `--session-name`     | セッション名         | 自動生成（タイムスタンプ） |

#### Python APIからの使用

```python
from evospikenet.sim_recorder import SimulationRecorder, RecorderConfig

# 記録設定を作成
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=True,
    record_weights=False,
    output_dir="./my_recordings",
    session_name="experiment_xor_task",
    spike_subsample_rate=1.0,  # 全スパイクを記録
    membrane_subsample_rate=0.1,  # 膜電位は10%サンプリング
    max_recording_duration=300.0  # 最大5分間記録
)

# レコーダーを初期化
recorder = SimulationRecorder(config)

# グローバルレコーダーとして設定（全ノードで使用可能）
from evospikenet.sim_recorder import set_global_recorder
set_global_recorder(recorder)

# ... シミュレーション実行 ...

# 記録を終了
recorder.close()
```

### 2. 記録されるデータ構造

#### ディレクトリ構造

```
sim_recordings/
└── sim_20251206_001234/         # セッションディレクトリ
    ├── simulation_data.h5        # HDF5データファイル（スパイク、膜電位、重み）
    ├── control_states.jsonl      # 制御状態（JSONL形式）
    ├── recording_statistics.json # 記録統計
    └── plots/                    # 自動生成されるプロット（解析後）
        ├── pfc-0_lif_raster.png
        ├── pfc-0_lif_timeline.png
        └── ...
```

#### HDF5データ構造

```
simulation_data.h5
├── /spikes                   # スパイクデータ
│   ├── /pfc-0
│   │   ├── /input
│   │   │   └── /t_1733420400000000000  # タイムスタンプ付きデータセット
│   │   └── /output
│   ├── /visual-0
│   └── ...
├── /membrane                 # 膜電位データ
│   ├── /pfc-0
│   │   └── /lif_layer
│   └── ...
├── /weights                  # 重みスナップショット
│   └── /lang-main
│       └── /transformer_layer_0
└── /metadata                 # メタデータ（設定情報など）
```

### 3. データ解析

#### 基本的な解析

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

# 記録をロード
analyzer = SimulationAnalyzer("./sim_recordings/sim_20251206_001234")

# 記録されたノードを表示
nodes = analyzer.get_recorded_nodes()
print(f"Recorded nodes: {nodes}")

# 各ノードの層を表示
for node_id in nodes:
    layers = analyzer.get_recorded_layers(node_id)
    print(f"{node_id}: {layers}")

# スパイクデータを取得
timestamps, spike_arrays = analyzer.get_spike_data("pfc-0", "output")
print(f"Recorded {len(spike_arrays)} timesteps")

# 発火率を計算
stats = analyzer.compute_firing_rate("pfc-0", "output")
print(f"Mean firing rate: {stats['mean_rate_hz']:.2f} Hz")
print(f"Total spikes: {stats['total_spikes']:,}")

# 解析を終了
analyzer.close()
```

#### 可視化

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

with SimulationAnalyzer("./sim_recordings/sim_20251206_001234") as analyzer:
    # スパイクラスタープロット
    analyzer.plot_spike_raster(
        node_id="pfc-0",
        layer_name="output",
        max_neurons=100,  # 表示する最大ニューロン数
        save_path="./pfc_raster.png"
    )
    
    # 発火率の時系列プロット
    analyzer.plot_firing_rate_timeline(
        node_id="pfc-0",
        layer_name="output",
        bin_size_ms=50.0,  # 50msビンで集計
        save_path="./pfc_timeline.png"
    )
    
    # サマリーレポート生成
    report = analyzer.generate_summary_report("./analysis_report.txt")
    print(report)
```

#### ノード動作の解析

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

analyzer = SimulationAnalyzer("./sim_recordings/sim_20251206_001234")

# 制御状態をロード
control_states = analyzer.load_control_states()
print(f"Total control records: {len(control_states)}")

# ノード動作の統計
behavior = analyzer.analyze_node_behavior("pfc-0")
print(f"Task active ratio: {behavior['task_active_ratio']:.2%}")
print(f"Unique statuses: {behavior['unique_statuses']}")
print(f"Step range: {behavior['step_range']}")

analyzer.close()
```

### 4. 高度な使用例

#### カスタムメタデータの記録

```python
# ZenohBrainNode内でのカスタム記録
def _process_pfc_timestep(self):
    # 既存の処理...
    
    # PFC特有のメタデータを記録
    if self.recorder and self.pfc_engine:
        # PFCエントロピーを記録
        entropy = self.pfc_engine.calculate_entropy()
        
        self.recorder.record_control_state(
            node_id=self.node_id,
            module_type=self.module_type,
            status="Processing",
            active_task=self.active_task,
            step_count=self.step_count,
            metadata={
                "pfc_entropy": float(entropy),
                "working_memory_size": len(self.working_memory),
                "quantum_modulation": self.pfc_engine.alpha_t
            }
        )
```

#### サブサンプリングでストレージ削減

```python
# 長時間シミュレーション用の設定
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=False,  # 膜電位は記録しない
    spike_subsample_rate=0.1,  # スパイクの10%のみ記録
    max_recording_duration=3600.0,  # 最大1時間
    buffer_size=2000,  # バッファサイズを増やして書き込み回数削減
    compression="gzip",  # GZIP圧縮
    compression_level=6  # 圧縮レベル（1-9、高いほど小さいが遅い）
)
```

#### バッチ記録の手動フラッシュ

```python
recorder = SimulationRecorder(config)

for step in range(10000):
    # シミュレーションステップ実行
    process_timestep()
    
    # 100ステップごとに手動でフラッシュ
    if step % 100 == 0:
        recorder.flush_all()
        stats = recorder.get_statistics()
        print(f"Step {step}: {stats['total_spikes_recorded']:,} spikes recorded")

recorder.close()
```

### 5. パフォーマンス考慮事項

#### メモリ使用量

| 記録設定             | 推定メモリ使用量（1ノード、1000ステップ） |
| -------------------- | ----------------------------------------- |
| スパイクのみ         | ~10-50 MB                                 |
| スパイク+膜電位      | ~50-200 MB                                |
| 全データ（重み含む） | ~500 MB - 2 GB                            |

#### ストレージ要件

```python
# 推定ストレージサイズ（圧縮なし）
neurons = 1000
timesteps = 10000
nodes = 4

# スパイク: binary (1 byte/neuron/timestep)
spike_size = neurons * timesteps * nodes * 1  # ~40 MB

# 膜電位: float32 (4 bytes/neuron/timestep)
membrane_size = neurons * timesteps * nodes * 4  # ~160 MB

# 重み: float32, 例えば1000x1000行列
weight_size = neurons * neurons * 4  # ~4 MB per snapshot

# 合計（圧縮で50-70%削減可能）
total_uncompressed = spike_size + membrane_size + weight_size
total_compressed = total_uncompressed * 0.3  # GZIP圧縮時
```

#### 最適化のヒント

1. **サブサンプリング**: 長時間シミュレーションではサブサンプリングを使用
   ```python
   spike_subsample_rate=0.1  # 10%のみ記録
   ```

2. **バッファサイズ**: メモリに余裕がある場合はバッファを大きく
   ```python
   buffer_size=5000  # ディスクI/O回数を削減
   ```

3. **圧縮**: ストレージ優先の場合は圧縮レベルを上げる
   ```python
   compression="gzip"
   compression_level=9  # 最高圧縮（ただし遅い）
   ```

4. **選択的記録**: 必要なデータのみ記録
   ```python
   record_membrane=False  # 膜電位は通常不要
   record_weights=False   # 重みは定期的なスナップショットのみ
   ```

### 6. トラブルシューティング

#### 問題: HDF5ファイルが破損

**原因**: シミュレーション中断時にバッファが未フラッシュ

**解決策**:
```python
# コンテキストマネージャーを使用（自動クローズ）
with SimulationRecorder(config) as recorder:
    # シミュレーション実行
    pass  # 自動的にcloseされる

# または明示的なtry-finally
recorder = SimulationRecorder(config)
try:
    # シミュレーション
    pass
finally:
    recorder.close()
```

#### 問題: ディスク容量不足

**解決策**:
```python
# 最大記録時間を設定
config = RecorderConfig(
    max_recording_duration=600.0,  # 10分で自動停止
    ...
)

# または定期的にストレージをチェック
import shutil
disk_usage = shutil.disk_usage(config.output_dir)
if disk_usage.free < 1e9:  # 1GB未満
    recorder.close()
    logger.warning("Disk space low, stopped recording")
```

#### 問題: 記録がパフォーマンスに影響

**解決策**:
```python
# より積極的なサブサンプリング
config = RecorderConfig(
    spike_subsample_rate=0.05,  # 5%のみ
    auto_flush=False,  # 自動フラッシュを無効化
    ...
)

# 手動で定期的にフラッシュ
if step % 1000 == 0:
    recorder.flush_all()
```

## ユースケース例

### ユースケース1: デバッグ用の詳細記録

```bash
# 短時間の詳細記録（全データ）
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording \
    --record-spikes \
    --record-membrane \
    --record-weights \
    --record-control \
    --session-name debug_session
```

### ユースケース2: 長期実験の効率的記録

```bash
# 長時間シミュレーション（サブサンプリング）
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --record-spikes \
    --session-name long_run_experiment
```

対応するPython設定:
```python
config = RecorderConfig(
    enable_recording=True,
    spike_subsample_rate=0.1,
    max_recording_duration=7200.0,  # 2時間
    compression_level=6
)
```

### ユースケース3: 複数ノードの協調動作解析

```bash
# ターミナル1: PFCノード（記録有効）
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording \
    --session-name multi_node_test

# ターミナル2: Visualノード（同じセッション）
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --session-name multi_node_test

# ターミナル3: Lang-Mainノード（同じセッション）
python examples/run_zenoh_distributed_brain.py \
    --node-id lang-main-0 \
    --module-type lang-main \
    --enable-recording \
    --session-name multi_node_test
```

解析:
```python
analyzer = SimulationAnalyzer("./sim_recordings/multi_node_test")

# 全ノードの動作を比較
for node_id in analyzer.get_recorded_nodes():
    behavior = analyzer.analyze_node_behavior(node_id)
    print(f"\n{node_id}:")
    print(f"  Task active: {behavior['task_active_ratio']:.2%}")
    
    for layer in analyzer.get_recorded_layers(node_id):
        stats = analyzer.compute_firing_rate(node_id, layer)
        print(f"  {layer}: {stats['mean_rate_hz']:.2f} Hz")
```

## まとめ

- ✅ **オプション機能**: `--enable-recording`で簡単に有効化/無効化
- ✅ **柔軟な記録**: スパイク、膜電位、重み、制御状態を個別に制御
- ✅ **効率的**: サブサンプリング、圧縮、バッファリングでパフォーマンス最適化
- ✅ **解析ツール**: 自動レポート生成、可視化、統計計算
- ✅ **スケーラブル**: 長時間・大規模シミュレーションに対応

## 参考資料

- `evospikenet/sim_recorder.py`: レコーダー実装
- `evospikenet/sim_analyzer.py`: 解析ツール実装
- `examples/run_zenoh_distributed_brain.py`: 統合例
