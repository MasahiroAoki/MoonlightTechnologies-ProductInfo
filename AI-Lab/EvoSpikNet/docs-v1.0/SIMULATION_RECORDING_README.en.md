# Simulation Data Recording and Analysis Feature

**Created:** December 6, 2025  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki

## Overview

A data recording and analysis system for distributed brain simulations has been added.

## Newly Added Files

### Core Modules
- `evospikenet/sim_recorder.py` - Data recording system
- `evospikenet/sim_analyzer.py` - Data analysis tool

### Documentation
- `docs/SIMULATION_RECORDING_GUIDE.md` - Detailed guide
- `examples/example_simulation_recording.py` - Usage examples

## Quick Start

### 1. Run Simulation with Recording Enabled

```bash
python examples/run_zenoh_distributed_brain.py \
    --node-id pfc-0 \
    --module-type pfc \
    --enable-recording
```

### 2. Analyze Recorded Data

```bash
python evospikenet/sim_analyzer.py ./sim_recordings/sim_20251206_001234
```

### 3. Run Sample Script

```bash
python examples/example_simulation_recording.py
```

## Recorded Data

- ✅ **Spike Data**: Spike trains from each layer
- ✅ **Membrane Potential Data**: Neuronal membrane potentials (optional)
- ✅ **Weight Data**: Network weight matrices (optional)
- ✅ **Control Data**: Node state transitions

## Key Features

### Recording Features
- Optional enable/disable toggle
- Subsampling for storage reduction
- GZIP compression support
- Buffered efficient writing

### Analysis Features
- Automatic firing rate calculation
- Spike raster plot generation
- Firing rate time series plots
- Automatic summary report generation

## Usage Example

```python
from evospikenet.sim_recorder import SimulationRecorder, RecorderConfig

# Configure recording
config = RecorderConfig(
    enable_recording=True,
    record_spikes=True,
    record_membrane=True,
    session_name="my_experiment"
)

# Start recording
with SimulationRecorder(config) as recorder:
    # Execute simulation
    for step in range(1000):
        # Record spikes
        recorder.record_spike_data(
            node_id="pfc-0",
            layer_name="lif",
            spikes=output_spikes
        )
```

## Detailed Information

For more details, see `docs/SIMULATION_RECORDING_GUIDE.md`.
