# Data Acquisition Feature

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Last Updated:** December 10, 2025

## Overview
The Brain Simulation tab now includes checkboxes to control data acquisition during distributed brain simulation execution. This feature allows users to:
- Enable/disable data acquisition completely
- Selectively enable specific data types (spikes, energy, membrane potential)
- See only relevant visualization panels based on their selections

## User Interface

### Data Acquisition Settings Card
Located at the top of the Brain Simulation tab, this card contains:

1. **Enable Data Acquisition** (Master Switch)
   - A toggle switch to enable or disable all data acquisition
   - Default: Enabled
   - When disabled, all visualization panels are hidden

2. **Data Type Selection** (Conditional Checkboxes)
   - Only visible when data acquisition is enabled
   - Three checkboxes for different data types:
     - **Spike Data**: Controls visibility of PFC Spike Train graph
     - **Energy & Entropy Data**: Controls visibility of PFC Energy and Entropy graph
     - **Membrane Potential Data**: Controls visibility of PFC Membrane Potential graph
   - Default: All enabled

## Behavior

### When Data Acquisition is Disabled
- The `--disable-recording` flag is passed to the backend simulation processes
- All visualization panels (graphs and figures) are hidden from view
- The simulation runs without collecting temporal change data
- Reduces computational overhead and memory usage

### When Data Acquisition is Enabled
- Data recording is enabled in the backend (default behavior)
- Individual visualization panels are shown/hidden based on selected data types
- Only relevant graphs are displayed to the user

## Implementation Details

### Frontend Changes (`frontend/pages/distributed_brain.py`)

1. **New UI Components**:
   - Data Acquisition Settings card with checkboxes
   - Container divs for individual visualization panels

2. **New Store**:
   - `data-acquisition-settings`: Persists user's data acquisition preferences

3. **New Callbacks**:
   - `update_data_acquisition_settings`: Updates the store when checkboxes change
   - `toggle_data_type_checkboxes`: Shows/hides data type checkboxes based on master switch
   - `toggle_visualization_panels`: Controls visibility of visualization panels
   - Modified `manage_simulation`: Passes `--disable-recording` flag to backend when needed

### Backend Integration
The feature integrates with the existing `--disable-recording` flag in `examples/run_zenoh_distributed_brain.py`:
- When data acquisition is disabled, this flag is added to each node's startup command
- The backend simulation recorder respects this flag and skips data collection

## Usage Example

### Scenario 1: Full Data Collection (Default)
1. User navigates to Brain Simulation tab
2. "Enable Data Acquisition" is already checked
3. All data types are selected
4. User starts simulation
5. All visualization panels are displayed and update with real-time data

### Scenario 2: Lightweight Simulation
1. User navigates to Brain Simulation tab
2. User unchecks "Enable Data Acquisition"
3. Data type checkboxes disappear
4. User starts simulation
5. Simulation runs without data collection
6. No visualization panels are shown (hidden)

### Scenario 3: Selective Data Collection
1. User navigates to Brain Simulation tab
2. "Enable Data Acquisition" is checked
3. User unchecks "Spike Data" and "Membrane Potential Data"
4. User starts simulation
5. Only the "PFC Energy and Entropy" graph is visible and updating
6. Other graphs are hidden

## Benefits

1. **Performance**: Users can disable data collection for faster simulations
2. **Clarity**: UI only shows relevant visualizations based on what's being collected
3. **Flexibility**: Users can selectively collect only the data they need
4. **User Experience**: Cleaner interface when data isn't being collected

## Future Enhancements

Potential improvements:
- Add more granular control over specific data streams
- Allow runtime toggling of data acquisition (currently requires simulation restart)
- Add data export options for collected data
- Display storage usage estimates based on selected data types
