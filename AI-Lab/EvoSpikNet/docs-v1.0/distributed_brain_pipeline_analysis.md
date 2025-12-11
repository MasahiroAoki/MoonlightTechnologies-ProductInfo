# 分散脳シミュレーションにおけるRank0入力のノードパイプライン決定方法の解析

**作成日:** 2025年12月6日  
**Copyright:** 2025 Moonlight Technologies Inc. All Rights Reserved.  
**Author:** Masahiro Aoki

## 1. 概要

本ドキュメントでは、分散脳シミュレーション（Distributed Brain Simulation）において、**Rank 0**（主に **PFC: Prefrontal Cortex / 前頭前野** ノードが該当）が外部からの入力を受け取った際、どのように後続の処理ノード（パイプライン）を決定しているかをソースコードベースで解析した結果をまとめる。

解析対象のソースコードは以下の通りである。
- **実行スクリプト**: `examples/run_zenoh_distributed_brain.py`
- **PFCロジック定義**: `evospikenet/pfc.py`
- **API定義**: `evospikenet/api.py`

現状の実装には、デモ用の**簡易実装**と、ライブラリとして定義されている**高度な意思決定エンジン**の2つの層が存在する。

---

## 2. 現行の分散シミュレーションにおける実装 (`run_zenoh_distributed_brain.py`)

現在稼働している分散脳シミュレーションのデモ（`run_zenoh_distributed_brain.py`）では、Rank 0（PFCノード）のパイプライン決定は**静的（ハードコード）**に行われている。

### 2.1 入力の流れ (Input Flow)

1.  **ユーザー入力**: ユーザーはAPI (`/api/distributed_brain/prompt`) に対してプロンプトを送信する。
2.  **API処理**: `evospikenet/api.py` がリクエストを受け取り、Zenohトピック `evospikenet/api/prompt` にメッセージをPublishする。
3.  **PFC受信**: `run_zenoh_distributed_brain.py` で起動している PFCノード（Rank 0）がこのトピックをSubscribeしており、`_on_api_prompt` メソッドで受信する。

### 2.2 パイプライン決定ロジック (Pipeline Determination)

PFCノード内の `_on_api_prompt` メソッドにおいて、以下の通り処理が行われる。

```python
# examples/run_zenoh_distributed_brain.py (抜粋)

def _on_api_prompt(self, data: Dict):
    """Handle prompt received directly from API via Zenoh."""
    text_prompt = data.get("prompt")
    prompt_id = data.get("prompt_id")
    
    if text_prompt and prompt_id:
        # (中略: 重複チェック)
        
        self.logger.info(f"Received prompt via Zenoh (id: {prompt_id}): '{text_prompt}'")
        
        # Dispatch to Lang-Main
        # ここで静的に "pfc/text_prompt" トピックへ送信している
        self.comm.publish("pfc/text_prompt", {"prompt": text_prompt, "prompt_id": prompt_id})
        self.active_task = True
```

**解析結果**:
- 現状のデモコードでは、入力されたテキストプロンプトは**無条件で `Lang-Main` ノード（言語処理メインノード）へルーティング**される。
- 動的な判断（例：画像処理ノードへ送るか、言語ノードへ送るか等）のロジックはこの層には実装されていない。

---

## 3. 高度な意思決定エンジンの実装 (`evospikenet/pfc.py`)

ライブラリのコア部分である `evospikenet/pfc.py` には、特許技術に基づく**量子変調フィードバックループ (Quantum-Modulated Feedback Loop)** を用いた動的なパイプライン決定ロジック `PFCDecisionEngine` が実装されている。将来的にはこのロジックが分散ノードに統合されることが想定される。

### 3.1 PFCDecisionEngine の構造

`PFCDecisionEngine` クラスは以下の要素で構成されている。

1.  **ワーキングメモリ (Working Memory)**: `LIFNeuronLayer` を使用したリカレントな短期記憶。
2.  **注意機構 (Attention Router)**: `ChronoSpikeAttention` を使用し、入力スパイクから重要な特徴を抽出する。
3.  **量子変調シミュレータ (QuantumModulationSimulator)**: 認知エントロピーから変調係数 $\alpha(t)$ を生成する。

### 3.2 パイプライン決定アルゴリズム

パイプライン（ルーティング先）の決定は `forward` メソッド内で行われる。

#### 手順1: 認知エントロピーの計算
各モジュールへのルーティングスコア（`route_scores`）から、現在の意思決定における不確実性（エントロピー）を計算する。

```python
# エントロピー計算 (Softmax分布のエントロピー)
entropy = -torch.sum(torch.softmax(route_scores, dim=-1) * torch.log_softmax(route_scores, dim=-1), dim=-1).mean()
```

#### 手順2: 量子変調係数 $\alpha(t)$ の生成
エントロピーを量子ビットの回転角 $\theta$ にマッピングし、状態 $|0\rangle$ が観測される確率を $\alpha(t)$ とする。

$$ \theta = \pi \times \frac{\text{Entropy}}{\text{MaxEntropy}} $$
$$ \alpha(t) = P(|0\rangle) = \cos^2(\frac{\theta}{2}) $$

#### 手順3: ルーティング温度の制御
$\alpha(t)$ を用いて、ルーティングの「温度（Temperature）」を動的に制御する。

- **高エントロピー（不確実）** $\rightarrow$ 低 $\alpha(t)$ $\rightarrow$ **高温** $\rightarrow$ **探索的 (Exploratory) ルーティング**
- **低エントロピー（確実）** $\rightarrow$ 高 $\alpha(t)$ $\rightarrow$ **低温** $\rightarrow$ **活用的 (Exploitative) ルーティング**

```python
routing_temp = 1.0 / (alpha_t + 1e-9)
route_probs = torch.softmax(route_scores / routing_temp, dim=-1)
```

#### 手順4: 自己ダイナミクスの変調
$\alpha(t)$ はルーティングだけでなく、PFC自身のニューロン発火閾値も変調させる（可塑性の制御）。

```python
# alphaが低い（探索的）ほど閾値が下がり、発火しやすくなる（過可塑性）
modulation_factor = 0.5 + alpha_t
self.working_memory.threshold = (self.base_lif_threshold * modulation_factor).to(torch.int16)
```

### 3.3 結論

`evospikenet/pfc.py` におけるパイプライン決定は、単なる条件分岐ではなく、**認知状態（エントロピー）に応じた量子力学的変調**を取り入れた動的な確率的ルーティングである。

---

## 4. まとめ

| 項目           | 現行デモ (`run_zenoh_distributed_brain.py`)       | コアライブラリ (`evospikenet/pfc.py`)              |
| :------------- | :------------------------------------------------ | :------------------------------------------------- |
| **決定方法**   | ✅ **動的確率的ルーティング (2025-12-05実装完了)** | **動的確率的ルーティング (Dynamic Probabilistic)** |
| **ロジック**   | 量子変調フィードバックループによる温度制御        | 量子変調フィードバックループによる温度制御         |
| **パラメータ** | 認知エントロピー、量子変調係数 $\alpha(t)$        | 認知エントロピー、量子変調係数 $\alpha(t)$         |
| **目的**       | 自律的な意思決定と探索・活用のバランス調整        | 自律的な意思決定と探索・活用のバランス調整         |

**✅ 実装完了 (2025年12月5日)**

Rank 0（PFC）のパイプライン決定ロジック `PFCDecisionEngine` は `run_zenoh_distributed_brain.py` の `ZenohBrainNode` に統合され、状況に応じた柔軟なタスク振り分け（例：視覚野への注意喚起、運動野への指令、言語野へのクエリなど）が可能になった。

### 実装の詳細

**統合内容:**
1. `ZenohBrainNode.__init__()` において、`module_type == "pfc"` の場合に `PFCDecisionEngine` をインスタンス化
2. モジュールマッピング: `["visual", "audio", "lang-main", "motor"]` の4つの下流モジュールを定義
3. `_on_api_prompt()` メソッドで動的ルーティングを実装:
   - プロンプトをテンソルに変換
   - `PFCDecisionEngine.forward()` を呼び出して `route_probs`, `entropy` を取得
   - 確率分布に基づいてターゲットモジュールをサンプリング
   - 選択されたモジュールに対応するZenohトピックへメッセージをPublish
4. ログ出力: `[Q-PFC ROUTING]` タグでエントロピー、確率分布、選択モジュールを記録

**技術的特徴:**
- **量子変調係数 $\alpha(t)$**: エントロピーから動的に計算され、ルーティング温度を制御
- **探索・活用のバランス**: 高エントロピー時は探索的（高温）、低エントロピー時は活用的（低温）
- **フォールバック機構**: PFCエンジン初期化失敗時は従来の静的ルーティングへ自動フォールバック
- **後方互換性**: 既存の `Lang-Main` ノードとの互換性を維持
