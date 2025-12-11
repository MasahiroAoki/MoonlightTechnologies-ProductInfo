# AEG-Comm (Adaptive Energy-based Gating for Communication) Implementation Plan

**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki  
**Created:** December 6, 2025  
**Last Updated:** December 10, 2025  
**Production Target:** Implementation completion by January 15, 2026

> **Note:** This is a summary document. For complete implementation specifications, safety architecture details, and performance analysis, please refer to the Japanese version: [AEG_COMM_IMPLEMENTATION_PLAN.md](AEG_COMM_IMPLEMENTATION_PLAN.md)

## Executive Summary

### Overview

**AEG-Comm** (Adaptive Energy-based Gating for Communication) extends the existing AEG (Activity-driven Energy Gating) to inter-node communication, implementing **energy-based intelligent spike transmission control**.

### Expected Impact (November 2025 Measurements)

| Metric                      | Current | With AEG-Comm | Improvement   |
| --------------------------- | ------- | ------------- | ------------- |
| **Communication Volume**    | 1.2GB/s | 90MB/s        | **93% reduction** |
| **Actual Latency**          | 180ms   | 28ms          | **84% reduction** |
| **Power Consumption**       | 48W     | 16W           | **67% reduction** |
| **Battery Life**            | 1 day   | 3 days        | **3x increase** |
| **Grasp Success Rate**      | 98.2%   | 99.6%         | **1.4% improvement** |
| **Emergency Stop Response** | 42ms    | 8ms           | **81% faster** |

### Final Decision

**✅ Adopt AEG-Comm**

However, wrap it completely with a **3-Layer Safety Architecture** to guarantee:
- Maximum communication efficiency (85-93% reduction)
- 100% safety assurance (reliable transmission of critical spikes)
- Improved biological plausibility (brain-like selective communication)

## Current Analysis

### Existing AEG Implementation

**File**: `evospikenet/control.py`

```python
class AEG(nn.Module):
    """Activity-driven Energy Gating (AEG) Implementation"""
    
    def update(self, spikes: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        """
        Gate (filter) spikes based on energy levels
        
        Returns:
            torch.Tensor: Gated spikes (local processing only)
        """
        # Energy consumption calculation
        # Active neuron mask generation
        # Local spike gating
        return spikes * active_mask
```

**Characteristics:**
- ✅ Energy-based gating
- ✅ Consumption adjusted by importance
- ❌ No inter-node communication functionality

### Existing Zenoh Communication

**File**: `evospikenet/zenoh_comm.py`

```python
class ZenohBrainCommunicator:
    def publish_spikes(self, target: str, spikes: torch.Tensor, metadata: Dict = None):
        """Send spike data to specified target"""
        # Current implementation sends all spikes
```

**Issue:** Current implementation sends all generated spikes without filtering

## AEG-Comm Design Philosophy

### Core Concept

Extend AEG's energy-based gating to control **what spikes are transmitted** over the network, not just which spikes are processed locally.

### Key Principles

1. **Energy Budget**: Each node has a communication energy budget
2. **Importance-Based Filtering**: High-importance spikes transmitted first
3. **Adaptive Thresholds**: Dynamically adjust based on available energy
4. **Safety Guarantees**: Critical spikes always transmitted

## 3-Layer Safety Architecture

### Layer 1: Critical Spike Whitelist

Certain spikes are **always transmitted** regardless of energy:
- Emergency signals
- Safety-critical commands
- Synchronization messages

### Layer 2: Priority-Based Queuing

Non-critical spikes queued by importance:
- High priority: Transmitted immediately if energy available
- Medium priority: Transmitted when budget permits
- Low priority: Transmitted opportunistically

### Layer 3: Energy Recovery

Energy budget replenishes over time:
- Fixed recovery rate
- Burst allowance for critical periods
- Degradation prevention

## Technical Specification

### Enhanced AEG Class

```python
class AEGWithUpstream(AEG):
    """AEG with automatic upstream spike transmission"""
    
    def __init__(self, num_neurons: int, 
                 communicator: ZenohBrainCommunicator,
                 target_node: str, **kwargs):
        super().__init__(num_neurons, **kwargs)
        self.comm = communicator
        self.target = target_node
        self.comm_energy = kwargs.get('comm_energy_budget', 1000.0)
    
    def update(self, spikes: torch.Tensor, 
               importance: torch.Tensor,
               critical_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Gate spikes and intelligently transmit to upstream
        
        Args:
            spikes: Spike tensor
            importance: Spike importance scores
            critical_mask: Mask indicating critical spikes (always sent)
        """
        # Local gating (existing AEG logic)
        gated_spikes = super().update(spikes, importance)
        
        # Communication gating (new AEG-Comm logic)
        if self.comm:
            comm_spikes = self._select_spikes_for_transmission(
                gated_spikes, importance, critical_mask
            )
            self.comm.publish_spikes(self.target, comm_spikes, metadata={
                "energy": self.comm_energy,
                "gated": True
            })
        
        return gated_spikes
    
    def _select_spikes_for_transmission(self, spikes, importance, critical_mask):
        """Select spikes for network transmission based on energy budget"""
        # Safety Layer 1: Always send critical spikes
        if critical_mask is not None:
            critical_spikes = spikes * critical_mask
        else:
            critical_spikes = torch.zeros_like(spikes)
        
        # Safety Layer 2: Priority-based selection for non-critical
        available_energy = self.comm_energy - critical_spikes.sum()
        if available_energy > 0:
            # Select high-importance spikes within budget
            selected = self._priority_selection(spikes, importance, available_energy)
            return critical_spikes + selected
        else:
            return critical_spikes
```

## Implementation Plan

### Phase 1: Core Implementation (Weeks 1-2)
- Implement `AEGWithUpstream` class
- Add communication energy tracking
- Implement priority-based selection

### Phase 2: Safety Architecture (Weeks 3-4)
- Implement critical spike whitelist
- Add priority queuing
- Implement energy recovery

### Phase 3: Integration (Weeks 5-6)
- Integrate with ZenohBrainNode
- Update distributed brain examples
- Comprehensive testing

### Phase 4: Optimization (Weeks 7-8)
- Performance tuning
- Memory optimization
- Documentation

## Test Strategy

### Unit Tests
- AEG-Comm spike selection logic
- Energy budget management
- Critical spike handling

### Integration Tests
- End-to-end communication flow
- Multi-node coordination
- Safety guarantees

### Performance Tests
- Communication volume reduction
- Latency improvements
- Energy consumption

## Performance Goals

- **Communication Reduction**: 85-93%
- **Latency Reduction**: 80%+
- **Energy Savings**: 65%+
- **Safety**: 100% critical spike delivery
- **Accuracy**: No degradation in task performance

## Risk Management

### Technical Risks
- **Complexity**: Mitigate with comprehensive testing
- **Performance**: Benchmark at each development stage
- **Safety**: Multiple validation layers

### Operational Risks
- **Debugging**: Enhanced logging and monitoring
- **Deployment**: Gradual rollout with fallback options
- **Maintenance**: Clear documentation and examples

## Success Metrics

1. ✅ Communication volume reduced by 85%+
2. ✅ Latency reduced by 80%+
3. ✅ 100% critical spike delivery rate
4. ✅ No task performance degradation
5. ✅ Energy consumption reduced by 65%+

## References

- AEG Implementation: `evospikenet/control.py`
- Zenoh Communication: `evospikenet/zenoh_comm.py`
- Distributed Brain: `examples/run_zenoh_distributed_brain.py`

For complete implementation details, safety architecture specifications, mathematical formulations, and benchmark results, please refer to the Japanese documentation: [AEG_COMM_IMPLEMENTATION_PLAN.md](AEG_COMM_IMPLEMENTATION_PLAN.md)
