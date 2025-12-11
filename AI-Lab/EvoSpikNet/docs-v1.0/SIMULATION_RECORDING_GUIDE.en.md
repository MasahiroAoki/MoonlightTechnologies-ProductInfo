# Simulation Data Recording and Analysis Guide

**Created:** December 6, 2025  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Target:** EvoSpikeNet Zenoh Distributed Brain Simulation

## Overview

A system has been implemented to record and analyze the following data during distributed brain simulation execution:

1. **Spike Data**: Spike trains from each neuron layer
2. **Membrane Potential Data**: Neuronal membrane potentials (optional)
3. **Weight Data**: Network weight matrix snapshots (optional)
4. **Control Data**: Node state transitions and task execution status

## Quick Start

### Run Simulation with Recording Enabled

```bash
# Basic recording (spikes + control data)
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording

# Record all data (including membrane potentials and weights)
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --record-membrane \
    --record-weights \
    --session-name my_experiment_001
```

### Analyze Recorded Data

```bash
# Automatic analysis (report + graph generation)
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234

# Skip plot generation
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234 --no-plots
```

## Detailed Guide

### 1. Recording Options

#### Command-Line Arguments

| Argument             | Description                          | Default                        |
| -------------------- | ------------------------------------ | ------------------------------ |
| `--enable-recording` | Enable recording                     | False (disabled)               |
| `--record-spikes`    | Record spike data                    | True                           |
| `--record-membrane`  | Record membrane potentials           | False                          |
| `--record-weights`   | Record weight matrices               | False                          |
| `--record-control`   | Record control states                | True                           |
| `--recording-dir`    | Recording storage directory          | `./sim_recordings`             |
| `--session-name`     | Session name                         | Auto-generated (timestamp)     |

#### Usage from Python API

```python
from evospikenet.sim_recorder import SimulationRecorder, RecorderConfig

# Create recording configuration
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=True,
    record_weights=False,
    output_dir="./my_recordings",
    session_name="experiment_xor_task",
    spike_subsample_rate=1.0,  # Record all spikes
    membrane_subsample_rate=0.1,  # Sample 10% of membrane potentials
    max_recording_duration=300.0  # Record for maximum 5 minutes
)

# Initialize recorder
recorder = SimulationRecorder(config)

# Set as global recorder (available to all nodes)
from evospikenet.sim_recorder import set_global_recorder
set_global_recorder(recorder)

# ... Execute simulation ...

# Close recording
recorder.close()
```

### 2. Recorded Data Structure

#### Directory Structure

```
sim_recordings/
└── sim_20251206_001234/         # Session directory
    ├── simulation_data.h5        # HDF5 data file (spikes, membrane, weights)
    ├── control_states.jsonl      # Control states (JSONL format)
    ├── recording_statistics.json # Recording statistics
    └── plots/                    # Auto-generated plots (after analysis)
        ├── pfc-0_lif_raster.png
        ├── pfc-0_lif_timeline.png
        └── ...
```

#### HDF5 Data Structure

```
simulation_data.h5
├── /spikes                   # Spike data
│   ├── /pfc-0
│   │   ├── /input
│   │   │   └── /t_1733420400000000000  # Timestamped dataset
│   │   └── /output
│   ├── /visual-0
│   └── ...
├── /membrane                 # Membrane potential data
│   ├── /pfc-0
│   │   └── /lif_layer
│   └── ...
├── /weights                  # Weight snapshots
│   └── /lang-main
│       └── /transformer_layer_0
└── /metadata                 # Metadata (configuration info, etc.)
```

### 3. Data Analysis

#### Basic Analysis

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

# Load recording
analyzer = SimulationAnalyzer("./sim_recordings/sim_20251206_001234")

# Display recorded nodes
nodes = analyzer.get_recorded_nodes()
print(f"Recorded nodes: {nodes}")

# Display layers for each node
for node_id in nodes:
    layers = analyzer.get_recorded_layers(node_id)
    print(f"{node_id}: {layers}")

# Get spike data
timestamps, spike_arrays = analyzer.get_spike_data("pfc-0", "output")
print(f"Recorded {len(spike_arrays)} timesteps")

# Calculate firing rate
stats = analyzer.compute_firing_rate("pfc-0", "output")
print(f"Mean firing rate: {stats['mean_rate_hz']:.2f} Hz")
print(f"Total spikes: {stats['total_spikes']:,}")

# Close analysis
analyzer.close()
```

#### Visualization

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

with SimulationAnalyzer("./sim_recordings/sim_20251206_001234") as analyzer:
    # Spike raster plot
    analyzer.plot_spike_raster(
        node_id="pfc-0",
        layer_name="output",
        max_neurons=100,  # Maximum neurons to display
        save_path="./pfc_raster.png"
    )
    
    # Firing rate time series plot
    analyzer.plot_firing_rate_timeline(
        node_id="pfc-0",
        layer_name="output",
        bin_size_ms=50.0,  # Aggregate in 50ms bins
        save_path="./pfc_timeline.png"
    )
    
    # Generate summary report
    report = analyzer.generate_summary_report("./analysis_report.txt")
    print(report)
```

#### Node Behavior Analysis

```python
from evospikenet.sim_analyzer import SimulationAnalyzer

analyzer = SimulationAnalyzer("./sim_recordings/sim_20251206_001234")

# Load control states
control_states = analyzer.load_control_states()
print(f"Total control records: {len(control_states)}")

# Node behavior statistics
behavior = analyzer.analyze_node_behavior("pfc-0")
print(f"Task active ratio: {behavior['task_active_ratio']:.2%}")
print(f"Unique statuses: {behavior['unique_statuses']}")
print(f"Step range: {behavior['step_range']}")

analyzer.close()
```

### 4. Advanced Usage Examples

#### Recording Custom Metadata

```python
# Custom recording within ZenohBrainNode
def _process_pfc_timestep(self):
    # Existing processing...
    
    # Record PFC-specific metadata
    if self.recorder and self.pfc_engine:
        # Record PFC entropy
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

#### Reduce Storage with Subsampling

```python
# Configuration for long-duration simulations
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=False,  # Don't record membrane potentials
    spike_subsample_rate=0.1,  # Record only 10% of spikes
    max_recording_duration=3600.0,  # Maximum 1 hour
    buffer_size=2000,  # Increase buffer to reduce write operations
    compression="gzip",  # GZIP compression
    compression_level=6  # Compression level (1-9, higher = smaller but slower)
)
```

#### Manual Flush of Batch Recording

```python
recorder = SimulationRecorder(config)

for step in range(10000):
    # Execute simulation step
    process_timestep()
    
    # Manual flush every 100 steps
    if step % 100 == 0:
        recorder.flush_all()
        stats = recorder.get_statistics()
        print(f"Step {step}: {stats['total_spikes_recorded']:,} spikes recorded")

recorder.close()
```

### 5. Performance Considerations

#### Memory Usage

| Recording Configuration | Estimated Memory Usage (1 node, 1000 steps) |
| ----------------------- | ------------------------------------------- |
| Spikes only             | ~10-50 MB                                   |
| Spikes + membrane       | ~50-200 MB                                  |
| All data (with weights) | ~500 MB - 2 GB                              |

#### Storage Requirements

```python
# Estimated storage size (uncompressed)
neurons = 1000
timesteps = 10000
nodes = 4

# Spikes: binary (1 byte/neuron/timestep)
spike_size = neurons * timesteps * nodes * 1  # ~40 MB

# Membrane: float32 (4 bytes/neuron/timestep)
membrane_size = neurons * timesteps * nodes * 4  # ~160 MB

# Weights: float32, e.g., 1000x1000 matrix
weight_size = neurons * neurons * 4  # ~4 MB per snapshot

# Total (compression can reduce by 50-70%)
total_uncompressed = spike_size + membrane_size + weight_size
total_compressed = total_uncompressed * 0.3  # With GZIP compression
```

#### Optimization Tips

1. **Subsampling**: Use subsampling for long-duration simulations
   ```python
   spike_subsample_rate=0.1  # Record only 10%
   ```

2. **Buffer Size**: Increase buffer if memory allows
   ```python
   buffer_size=5000  # Reduce disk I/O operations
   ```

3. **Compression**: Increase compression level if storage is priority
   ```python
   compression="gzip"
   compression_level=9  # Maximum compression (but slower)
   ```

4. **Selective Recording**: Record only necessary data
   ```python
   record_membrane=False  # Membrane potentials usually not needed
   record_weights=False   # Weights only for periodic snapshots
   ```

### 6. Troubleshooting

#### Issue: HDF5 File Corrupted

**Cause**: Buffer not flushed when simulation interrupted

**Solution**:
```python
# Use context manager (automatic close)
with SimulationRecorder(config) as recorder:
    # Execute simulation
    pass  # Automatically closed

# Or explicit try-finally
recorder = SimulationRecorder(config)
try:
    # Simulation
    pass
finally:
    recorder.close()
```

#### Issue: Disk Space Insufficient

**Solution**:
```python
# Set maximum recording duration
config = RecorderConfig(
    max_recording_duration=600.0,  # Auto-stop after 10 minutes
    ...
)

# Or periodically check storage
import shutil
disk_usage = shutil.disk_usage(config.output_dir)
if disk_usage.free < 1e9:  # Less than 1GB
    recorder.close()
    logger.warning("Disk space low, stopped recording")
```

#### Issue: Recording Impacts Performance

**Solution**:
```python
# More aggressive subsampling
config = RecorderConfig(
    spike_subsample_rate=0.05,  # Only 5%
    auto_flush=False,  # Disable auto flush
    ...
)

# Manual periodic flush
if step % 1000 == 0:
    recorder.flush_all()
```

## Use Case Examples

### Use Case 1: Detailed Recording for Debugging

```bash
# Short-duration detailed recording (all data)
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

### Use Case 2: Efficient Recording for Long-Term Experiments

```bash
# Long-duration simulation (with subsampling)
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --record-spikes \
    --session-name long_run_experiment
```

Corresponding Python configuration:
```python
config = RecorderConfig(
    enable_recording=True,
    spike_subsample_rate=0.1,
    max_recording_duration=7200.0,  # 2 hours
    compression_level=6
)
```

### Use Case 3: Multi-Node Coordinated Behavior Analysis

```bash
# Terminal 1: PFC node (recording enabled)
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording \
    --session-name multi_node_test

# Terminal 2: Visual node (same session)
python examples/run_zenoh_distributed_brain.py \
    --node-id visual-0 \
    --module-type visual \
    --enable-recording \
    --session-name multi_node_test

# Terminal 3: Lang-Main node (same session)
python examples/run_zenoh_distributed_brain.py \
    --node-id lang-main-0 \
    --module-type lang-main \
    --enable-recording \
    --session-name multi_node_test
```

Analysis:
```python
analyzer = SimulationAnalyzer("./sim_recordings/multi_node_test")

# Compare behavior across all nodes
for node_id in analyzer.get_recorded_nodes():
    behavior = analyzer.analyze_node_behavior(node_id)
    print(f"\n{node_id}:")
    print(f"  Task active: {behavior['task_active_ratio']:.2%}")
    
    for layer in analyzer.get_recorded_layers(node_id):
        stats = analyzer.compute_firing_rate(node_id, layer)
        print(f"  {layer}: {stats['mean_rate_hz']:.2f} Hz")
```

## Summary

- ✅ **Optional Feature**: Easy to enable/disable with `--enable-recording`
- ✅ **Flexible Recording**: Individual control of spikes, membrane potentials, weights, and control states
- ✅ **Efficient**: Performance optimization through subsampling, compression, and buffering
- ✅ **Analysis Tools**: Automatic report generation, visualization, and statistical calculation
- ✅ **Scalable**: Supports long-duration and large-scale simulations

## References

- `evospikenet/sim_recorder.py`: Recorder implementation
- `evospikenet/sim_analyzer.py`: Analysis tool implementation
- `examples/run_zenoh_distributed_brain.py`: Integration examples
