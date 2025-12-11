# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# Advanced Decision Engine - Implementation Guide

## Overview

We have implemented an advanced decision-making engine for EvoSpikeNet's distributed brain simulation. This implementation extends the existing PFCDecisionEngine (quantum-modulated feedback loop) with the following features:

1. **Hierarchical Planning**
2. **Meta-Cognitive Monitoring**
3. **Multi-Step Reasoning**
4. **Dynamic Resource Allocation**
5. **Error Detection & Recovery**

## Architecture

### Key Components

```
AdvancedPFCEngine
├── PFCDecisionEngine (Base)
│   ├── QuantumModulationSimulator
│   ├── WorkingMemory (LIF Layer)
│   └── ChronoSpikeAttention
│
└── ExecutiveControlEngine
    ├── HierarchicalPlanner
    ├── MetaCognitiveMonitor
    ├── GoalManager
    └── ResourceAllocator
```

### New Modules

#### 1. ExecutiveControlEngine (`evospikenet/executive_control.py`)

**Role:** Highest-level executive control for full brain simulation

**Key Features:**
- Goal management and prioritization
- Plan creation and execution tracking
- Performance monitoring and adaptation
- Context-aware decision making

**Class Structure:**
```python
class ExecutiveControlEngine(nn.Module):
    def __init__(self, input_dim, num_modules, max_concurrent_goals=5)
    def add_goal(goal, goal_embedding)
    def select_next_action(current_state)
    def allocate_resources(action)
    def execute_step(action, result_state)
    def replan(failed_plan)
    def get_status_summary()
```

#### 2. MetaCognitiveMonitor

**Role:** Monitor and evaluate system's own decision-making process

**Functions:**
- Uncertainty Estimation
- Confidence Assessment
- Error Detection
- Self-Assessment

**Neural Network Architecture:**
- `uncertainty_net`: Estimates decision uncertainty (0=certain, 1=uncertain)
- `confidence_net`: Evaluates confidence (0=low, 1=high)
- `error_detector`: Calculates error probability

**Decision Quality Classification:**
- `high_quality`: Low uncertainty & high confidence
- `moderate`: Medium uncertainty/confidence
- `low_quality`: High uncertainty or low confidence
- `critical`: High error probability or very low confidence

#### 3. HierarchicalPlanner

**Role:** Hierarchical task decomposition and dependency management

**Functions:**
- Recursive goal decomposition (max depth 3)
- Dependency prediction between sub-goals
- Automatic priority assignment
- Executable plan generation

**Algorithm:**
1. Goal encoding
2. Sub-goal generation (num_modules)
3. Activation filtering (keep above threshold)
4. Dependency prediction (pairwise evaluation)
5. Priority assignment (CRITICAL/HIGH/NORMAL/LOW)

#### 4. AdvancedPFCEngine (added to `evospikenet/pfc.py`)

**Role:** Integration of base PFC engine and Executive Control

**New Methods:**
```python
def forward_with_planning(input_data, context) -> Dict:
    """
    Extended forward pass with planning & meta-cognition
    
    Returns:
        - route_probs: Routing probability distribution
        - entropy: Cognitive entropy
        - spikes, potential: LIF states
        - decision_state: Decision state vector
        - meta_assessment: Meta-cognitive evaluation
        - next_action: Next execution action (when planning enabled)
        - resource_allocation: Resource distribution
        - executive_status: Execution status summary
    """

def add_goal(goal_description, priority, metadata) -> str:
    """Goal addition interface"""

def execute_step(action, result_state) -> bool:
    """Step execution and state update"""

def get_performance_stats() -> Dict:
    """Performance statistics retrieval"""
```

## Usage

### 1. Basic Usage (Compatible with Standard PFC)

```python
from evospikenet.pfc import AdvancedPFCEngine

# Initialization
pfc = AdvancedPFCEngine(
    size=128,
    num_modules=4,
    n_heads=4,
    time_steps=16,
    enable_executive_control=True  # Enable Executive Control
)

# Standard forward (compatible with existing code)
input_tensor = torch.randint(0, 256, (1, 32), dtype=torch.long)
route_probs, entropy, spikes, potential = pfc.forward(input_tensor)
```

### 2. Advanced Usage (Planning & Meta-Cognition)

```python
# Extended forward with planning
result = pfc.forward_with_planning(
    input_tensor,
    context={"enable_planning": True}
)

# Check meta-cognitive assessment
meta = result["meta_assessment"]
print(f"Decision Quality: {meta['quality']}")
print(f"Confidence: {meta['confidence']:.3f}")
print(f"Uncertainty: {meta['uncertainty']:.3f}")
print(f"Error Probability: {meta['error_probability']:.3f}")

# If recommended action exists
if "next_action" in result:
    action = result["next_action"]
    allocation = result["resource_allocation"]
    print(f"Next Action: {action['step_id']}")
    print(f"Resource Allocation: {allocation}")
```

### 3. Goal Management

```python
# Add goal
goal_id = pfc.add_goal(
    goal_description="Execute image recognition task",
    priority="HIGH",
    metadata={"image_path": "/path/to/image.jpg"}
)

# Execute step
if "next_action" in result:
    success = pfc.execute_step(
        action=result["next_action"],
        result_state=torch.randn(128)  # State after execution
    )
    print(f"Step executed: {success}")

# Check execution status
status = pfc.get_executive_status()
print(f"Active Goals: {status['goals']['in_progress']}")
print(f"Completed Goals: {status['goals']['completed']}")
print(f"Active Plans: {status['plans']['active']}")
```

### 4. Integration with Distributed Brain Nodes

Integration in `examples/run_zenoh_distributed_brain.py`:

```python
# PFC node initialization
if module_type == "pfc":
    self.advanced_pfc = AdvancedPFCEngine(
        size=config.get("d_model", 128),
        num_modules=len(self.module_mapping),
        n_heads=4,
        time_steps=16,
        enable_executive_control=True
    )

# Prompt processing
result = self.advanced_pfc.forward_with_planning(
    prompt_tensor,
    context={"enable_planning": False}
)

# Meta-cognitive logging
if "meta_assessment" in result:
    meta = result["meta_assessment"]
    logger.info(
        f"[META-COGNITION] Quality={meta['quality']} | "
        f"Confidence={meta['confidence']:.3f} | "
        f"Uncertainty={meta['uncertainty']:.3f}"
    )
```

## Configuration Options

### Distributed Brain Node Configuration (`docker-compose.yml`)

```yaml
pfc-0:
  environment:
    - USE_ADVANCED_PFC=true  # Enable Advanced PFC (default: true)
```

### Zenoh Topics

Newly added topics:

- `pfc/add_goal`: Goal addition request
- `pfc/goal_added`: Goal addition completion notification
- `pfc/get_status`: Execution status request
- `pfc/status_response`: Execution status response

**Goal Addition Example:**
```python
comm.publish("pfc/add_goal", {
    "description": "Process visual information to recognize objects",
    "priority": "HIGH",
    "metadata": {"timeout": 30.0}
})
```

**Status Retrieval Example:**
```python
comm.publish("pfc/get_status", {})
# Receive on pfc/status_response
```

## Performance Statistics

Tracked metrics:

- `total_decisions`: Total number of decisions
- `successful_decisions`: Number of successful decisions
- `failed_decisions`: Number of failed decisions
- `average_entropy`: Average entropy
- `success_rate`: Success rate

```python
stats = pfc.get_performance_stats()
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Entropy: {stats['average_entropy']:.3f}")
```

## Testing

Test suite: `tests/test_advanced_pfc.py`

**Run tests:**
```bash
cd /Users/maoki/Documents/GitHub/EvoSpikeNet
python3 tests/test_advanced_pfc.py
```

**Test Coverage:**
- MetaCognitiveMonitor: Uncertainty/confidence/error detection
- HierarchicalPlanner: Goal decomposition/dependencies/priorities
- ExecutiveControlEngine: Goal management/action selection/resource allocation
- AdvancedPFCEngine: Integration tests/performance tracking
- Integration Scenarios: Multi-step/error recovery

## Technical Details

### Data Structures

```python
@dataclass
class Goal:
    goal_id: str
    description: str
    priority: TaskPriority
    created_at: float
    deadline: Optional[float]
    parent_goal_id: Optional[str]
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

@dataclass
class Plan:
    plan_id: str
    goal_id: str
    steps: List[Dict[str, Any]]
    current_step: int
    dependencies: Dict[str, List[str]]
    estimated_duration: float
    actual_duration: float
    status: TaskStatus
    metadata: Dict[str, Any]

@dataclass
class ExecutionContext:
    active_goals: List[Goal]
    active_plans: List[Plan]
    resource_allocation: Dict[str, float]
    performance_metrics: Dict[str, float]
    error_history: List[Dict[str, Any]]
```

### Decision Flow

```
Input
  ↓
PFCDecisionEngine.forward()
  ↓
[Quantum Modulation] → Generate α(t) → Control routing temperature
  ↓
Get decision_state
  ↓
MetaCognitiveMonitor(decision_state)
  ↓
Calculate [uncertainty, confidence, error_prob]
  ↓
HierarchicalPlanner.select_next_action(decision_state)
  ↓
ResourceAllocator.allocate_resources(action)
  ↓
Output: {route_probs, meta_assessment, next_action, allocation}
```

## Future Extensions

### Planned Features

1. **Reinforcement Learning Integration**: Self-improvement using meta-cognitive feedback
2. **Long-term Memory**: Episodic memory for goal history
3. **Multi-modal Integration**: Integrated planning across visual/auditory/language
4. **Distributed Planning**: Collaborative decision-making across multiple PFC nodes
5. **Attention Mechanism Extension**: Transformer-style hierarchical attention

### Research Directions

- Meta-learning for automatic planning strategy optimization
- Causal inference integration
- Counterfactual reasoning
- Explainable AI

## License

Copyright 2025 Moonlight Technologies Inc.

## References

- `evospikenet/executive_control.py`: Executive Control implementation
- `evospikenet/pfc.py`: PFC Decision Engine & Advanced PFC
- `examples/run_zenoh_distributed_brain.py`: Distributed brain integration
- `tests/test_advanced_pfc.py`: Test suite
- `docs/DISTRIBUTED_BRAIN_SYSTEM.en.md`: Distributed brain system specification
