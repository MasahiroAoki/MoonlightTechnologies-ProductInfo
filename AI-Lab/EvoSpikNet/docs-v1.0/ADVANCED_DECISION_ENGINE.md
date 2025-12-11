# Copyright 2025 Moonlight Technologies Inc. All Rights Reserved.
# Auth Masahiro Aoki

# 高度な意思決定エンジン - 実装ガイド

## 概要

EvoSpikeNetの分散脳シミュレーションに、高度な意思決定エンジンを実装しました。この実装は、既存のPFCDecisionEngine（量子変調フィードバックループ）を拡張し、以下の機能を追加します:

1. **階層的プランニング (Hierarchical Planning)**
2. **メタ認知モニタリング (Meta-Cognitive Monitoring)**
3. **マルチステップ推論 (Multi-Step Reasoning)**
4. **動的リソース割り当て (Dynamic Resource Allocation)**
5. **エラー検出と回復 (Error Detection & Recovery)**

## アーキテクチャ

### 主要コンポーネント

```
AdvancedPFCEngine
├── PFCDecisionEngine (Base)
│   ├── QuantumModulationSimulator
│   ├── WorkingMemory (LIF層)
│   └── ChronoSpikeAttention
│
└── ExecutiveControlEngine
    ├── HierarchicalPlanner
    ├── MetaCognitiveMonitor
    ├── GoalManager
    └── ResourceAllocator
```

### 新規モジュール

#### 1. ExecutiveControlEngine (`evospikenet/executive_control.py`)

**役割:** 全脳シミュレーションの最高レベルの実行制御

**主要機能:**
- ゴール管理と優先順位付け
- プラン作成と実行追跡
- パフォーマンスモニタリングと適応
- コンテキスト認識型意思決定

**クラス構成:**
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

**役割:** システム自身の意思決定プロセスの監視と評価

**機能:**
- 不確実性推定 (Uncertainty Estimation)
- 信頼度評価 (Confidence Assessment)
- エラー検出 (Error Detection)
- 自己評価 (Self-Assessment)

**ニューラルネットワーク構成:**
- `uncertainty_net`: 決定の不確実性を推定 (0=確実, 1=不確実)
- `confidence_net`: 信頼度を評価 (0=低信頼, 1=高信頼)
- `error_detector`: エラー確率を計算

**決定品質の分類:**
- `high_quality`: 低不確実性 & 高信頼度
- `moderate`: 中程度の不確実性/信頼度
- `low_quality`: 高不確実性 or 低信頼度
- `critical`: 高エラー確率 or 極低信頼度

#### 3. HierarchicalPlanner

**役割:** 高レベルゴールの階層的タスク分解と依存関係管理

**機能:**
- ゴールの再帰的分解 (最大深度3)
- サブゴール間の依存関係予測
- 優先順位の自動割り当て
- 実行可能プランの生成

**アルゴリズム:**
1. ゴールエンコーディング
2. サブゴール生成 (num_modules個)
3. 活性化フィルタリング (閾値以上のみ保持)
4. 依存関係予測 (ペアワイズ評価)
5. 優先順位割り当て (CRITICAL/HIGH/NORMAL/LOW)

#### 4. AdvancedPFCEngine (`evospikenet/pfc.py` に追加)

**役割:** 基本PFCエンジンとExecutive Controlの統合

**新規メソッド:**
```python
def forward_with_planning(input_data, context) -> Dict:
    """
    拡張forward pass with planning & meta-cognition
    
    Returns:
        - route_probs: ルーティング確率分布
        - entropy: 認知エントロピー
        - spikes, potential: LIF状態
        - decision_state: 決定状態ベクトル
        - meta_assessment: メタ認知評価 (quality, uncertainty, confidence)
        - next_action: 次の実行アクション (planning有効時)
        - resource_allocation: リソース配分
        - executive_status: 実行状態サマリー
    """

def add_goal(goal_description, priority, metadata) -> str:
    """ゴール追加インターフェース"""

def execute_step(action, result_state) -> bool:
    """ステップ実行と状態更新"""

def get_performance_stats() -> Dict:
    """パフォーマンス統計取得"""
```

## 使用方法

### 1. 基本的な使用 (標準PFCとの互換性)

```python
from evospikenet.pfc import AdvancedPFCEngine

# 初期化
pfc = AdvancedPFCEngine(
    size=128,
    num_modules=4,
    n_heads=4,
    time_steps=16,
    enable_executive_control=True  # Executive Control有効化
)

# 標準forward (既存コードと互換)
input_tensor = torch.randint(0, 256, (1, 32), dtype=torch.long)
route_probs, entropy, spikes, potential = pfc.forward(input_tensor)
```

### 2. 高度な使用 (プランニング & メタ認知)

```python
# 拡張forward with planning
result = pfc.forward_with_planning(
    input_tensor,
    context={"enable_planning": True}
)

# メタ認知評価の確認
meta = result["meta_assessment"]
print(f"Decision Quality: {meta['quality']}")
print(f"Confidence: {meta['confidence']:.3f}")
print(f"Uncertainty: {meta['uncertainty']:.3f}")
print(f"Error Probability: {meta['error_probability']:.3f}")

# 推奨アクションがある場合
if "next_action" in result:
    action = result["next_action"]
    allocation = result["resource_allocation"]
    print(f"Next Action: {action['step_id']}")
    print(f"Resource Allocation: {allocation}")
```

### 3. ゴール管理

```python
# ゴール追加
goal_id = pfc.add_goal(
    goal_description="画像認識タスクの実行",
    priority="HIGH",
    metadata={"image_path": "/path/to/image.jpg"}
)

# ステップ実行
if "next_action" in result:
    success = pfc.execute_step(
        action=result["next_action"],
        result_state=torch.randn(128)  # 実行後の状態
    )
    print(f"Step executed: {success}")

# 実行状態確認
status = pfc.get_executive_status()
print(f"Active Goals: {status['goals']['in_progress']}")
print(f"Completed Goals: {status['goals']['completed']}")
print(f"Active Plans: {status['plans']['active']}")
```

### 4. 分散脳ノードでの使用

`examples/run_zenoh_distributed_brain.py`での統合:

```python
# PFCノード初期化時
if module_type == "pfc":
    self.advanced_pfc = AdvancedPFCEngine(
        size=config.get("d_model", 128),
        num_modules=len(self.module_mapping),
        n_heads=4,
        time_steps=16,
        enable_executive_control=True
    )

# プロンプト処理時
result = self.advanced_pfc.forward_with_planning(
    prompt_tensor,
    context={"enable_planning": False}
)

# メタ認知ログ
if "meta_assessment" in result:
    meta = result["meta_assessment"]
    logger.info(
        f"[META-COGNITION] Quality={meta['quality']} | "
        f"Confidence={meta['confidence']:.3f} | "
        f"Uncertainty={meta['uncertainty']:.3f}"
    )
```

## 設定オプション

### 分散脳ノード設定 (`docker-compose.yml`)

```yaml
pfc-0:
  environment:
    - USE_ADVANCED_PFC=true  # Advanced PFC有効化 (デフォルト: true)
```

### Zenohトピック

新規追加されたトピック:

- `pfc/add_goal`: ゴール追加リクエスト
- `pfc/goal_added`: ゴール追加完了通知
- `pfc/get_status`: 実行状態リクエスト
- `pfc/status_response`: 実行状態レスポンス

**ゴール追加例:**
```python
comm.publish("pfc/add_goal", {
    "description": "視覚情報を処理して物体を認識",
    "priority": "HIGH",
    "metadata": {"timeout": 30.0}
})
```

**状態取得例:**
```python
comm.publish("pfc/get_status", {})
# pfc/status_response で受信
```

## パフォーマンス統計

追跡される指標:

- `total_decisions`: 総意思決定回数
- `successful_decisions`: 成功した決定数
- `failed_decisions`: 失敗した決定数
- `average_entropy`: 平均エントロピー
- `success_rate`: 成功率

```python
stats = pfc.get_performance_stats()
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Entropy: {stats['average_entropy']:.3f}")
```

## テスト

テストスイート: `tests/test_advanced_pfc.py`

**実行方法:**
```bash
cd /Users/maoki/Documents/GitHub/EvoSpikeNet
python3 tests/test_advanced_pfc.py
```

**テストカバレッジ:**
- MetaCognitiveMonitor: 不確実性/信頼度/エラー検出
- HierarchicalPlanner: ゴール分解/依存関係/優先順位
- ExecutiveControlEngine: ゴール管理/アクション選択/リソース配分
- AdvancedPFCEngine: 統合テスト/パフォーマンス追跡
- Integration Scenarios: マルチステップ/エラー回復

## 技術詳細

### データ構造

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

### 決定フロー

```
Input
  ↓
PFCDecisionEngine.forward()
  ↓
[Quantum Modulation] → α(t) 生成 → ルーティング温度制御
  ↓
decision_state 取得
  ↓
MetaCognitiveMonitor(decision_state)
  ↓
[uncertainty, confidence, error_prob] 計算
  ↓
HierarchicalPlanner.select_next_action(decision_state)
  ↓
ResourceAllocator.allocate_resources(action)
  ↓
Output: {route_probs, meta_assessment, next_action, allocation}
```

## 今後の拡張

### 計画中の機能

1. **強化学習統合**: メタ認知フィードバックを使った自己改善
2. **長期記憶**: ゴール履歴のエピソード記憶
3. **マルチモーダル統合**: 視覚/聴覚/言語の統合的プランニング
4. **分散プランニング**: 複数PFCノード間での協調的意思決定
5. **注意機構の拡張**: Transformer風の階層的注意

## 実装状況

### ✅ 実装済み機能 (v1.0)

| 機能 | ステータス | 詳細 |
|------|-----------|------|
| **ExecutiveControlEngine** | ✅ 完全実装 | ゴール管理、プラン作成、実行追跡 |
| **HierarchicalPlanner** | ✅ 完全実装 | ゴール分解、依存関係予測、優先順位付け |
| **MetaCognitiveMonitor** | ✅ 完全実装 | 不確実性推定、信頼度評価、エラー検出 |
| **add_goal()** | ✅ 実装済 | ゴール追加、プラン生成 |
| **select_next_action()** | ✅ 実装済 | 実行可能なステップの選択 |
| **allocate_resources()** | ✅ 実装済 | ニューラルモジュールへのリソース割り当て |
| **execute_step()** | ✅ 実装済 | ステップの実行とステータス更新 |
| **replan()** | ✅ 実装済 | 失敗時の再プラン機能 |
| **get_executive_status()** | ✅ 実装済 | 実行状態の取得 |
| **get_performance_stats()** | ✅ 実装済 | パフォーマンス統計（決定数、成功率、エントロピー） |
| **forward_with_planning()** | ✅ 実装済 | プランニング統合フォワードパス |
| **distributed_brain統合** | ✅ 実装済 | Zenoh経由の分散脳ノード統合 |

### ⏳ 計画中・部分実装

| 機能 | ステータス | 説明 |
|------|-----------|------|
| **Zenohトピック統合** | 📋 計画中 | `pfc/add_goal`, `pfc/status_response` などの非同期トピック通信 |
| **エラー回復戦略の拡張** | 🔄 部分実装 | 基本的なリプラン機能のみ実装、3段階回復メカニズムは Future |
| **並列実行切り替え** | 📋 計画中 | 複数プランの並列実行サポート |

### 📝 API ドキュメンテーション更新

#### get_executive_status()
```python
def get_executive_status(self) -> Dict[str, Any]:
    """
    実行エンジンの現在の状態を取得します。
    get_status_summary() へのエイリアスメソッドです。
    
    戻り値:
        {
            'goals': {
                'total': int,           # 総ゴール数
                'completed': int,       # 完了したゴール数
                'failed': int,          # 失敗したゴール数
                'in_progress': int      # 進行中のゴール数
            },
            'plans': {
                'total': int,           # 総プラン数
                'active': int           # アクティブなプラン数
            },
            'errors': int,              # エラー履歴数
            'resource_allocation': dict # リソース割り当て
        }
    """
    return self.get_status_summary()
```

#### get_performance_stats()
```python
def get_performance_stats(self) -> Dict[str, Any]:
    """
    意思決定エンジンのパフォーマンスメトリクスを取得します。
    
    戻り値:
        {
            'total_decisions': int,          # 実施された意思決定の総数
            'successful_decisions': int,     # 成功した意思決定数
            'failed_decisions': int,         # 失敗した意思決定数
            'average_entropy': float,        # 平均エントロピー（0-1）
            'success_rate': float            # 成功率（0-1）
        }
    """
```

**実装詳細:**
- decision_history フィールドは ExecutionContext に追加されました
- 各意思決定は {'success': bool, 'entropy': float, ...} として記録されます
- 決定履歴がない場合は、すべてのメトリクスが 0 で返却されます

### 研究方向

- メタ学習によるプランニング戦略の自動最適化
- 因果推論の統合
- 反事実的推論 (Counterfactual Reasoning)
- 説明可能性 (Explainable AI)

## ライセンス

Copyright 2025 Moonlight Technologies Inc.

## 参照

- `evospikenet/executive_control.py`: Executive Control実装
- `evospikenet/pfc.py`: PFC Decision Engine & Advanced PFC
- `examples/run_zenoh_distributed_brain.py`: 分散脳統合
- `tests/test_advanced_pfc.py`: テストスイート
- `docs/DISTRIBUTED_BRAIN_SYSTEM.md`: 分散脳システム仕様
