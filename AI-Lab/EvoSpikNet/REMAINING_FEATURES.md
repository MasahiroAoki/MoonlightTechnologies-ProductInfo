# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki


# プロジェクト機能実装ステータス

このドキュメントは、EvoSpikeNetプロジェクトの機能と実装ステータスを追跡します。

---
**注記:**
- **フェーズ8（分散グリッド）について:** この機能は、Pythonの`multiprocessing`ライブラリを使用して実装がしたが、困難なデッドロック（プロセスの永久停止）問題に直面しました。これは現在の実行環境に起因し、不安定なコードがプロジェクトに残ることを避けるため、プロジェクトの安定性を優先し、この機能の実装は削除しています。
- **GraphAnnealingRuleについて:** フェーズ3の量子インスパイアード機能の一部である`GraphAnnealingRule`は、数値安定性の問題を解決し、`float64`への統一と堅牢なテストスイートと共に安定的に再実装されました。
- **Spiking Transformerについて:** 当初のフェーズ5は、純粋なスパイクベースの実装の複雑さを考慮し、標準的な（浮動小数点ベースの）Transformerブロックを実装しました。詳細な設計案に基づき、**フェーズSNN-LM**として、`snnTorch`を活用した本格的なハイブリッド・スパイキング言語モデル (`SpikingEvoSpikeNetLM`) を実装しました。

---

| フェーズ | 機能 | 実装状況 | テスト/検証スクリプト | テスト状況 |
| :--- | :--- | :---: | :--- | :---: |
| **フェーズ1** | **コアSNNエンジン** | | | |
| | `LIFNeuronLayer`, `SynapseMatrixCSR`, `SNNModel` | ✔️ | `tests/test_core.py` | ✅ 成功 |
| | 検証プログラム | ✔️ | `example.py` | ✅ 成功 |
| **フェーズ2** | **動的グラフ進化エンジンとInsight Engine** | | | |
| | 可塑性ルール (`STDP`, `Homeostasis`) | ✔️ | `tests/test_plasticity.py` | ✅ 成功 |
| | `MetaPlasticity` | ✔️ | `tests/test_evolution.py` | ✅ 成功 |
| | `GraphUpdateManager` | ✔️ | `tests/test_evolution.py` | ✅ 成功 |
| | 監視・可視化 (`DataMonitorHook`, `InsightEngine`) | ✔️ | `tests/test_insight.py` | ✅ 成功 |
| | 検証プログラム | ✔️ | `examples/run_plasticity_demo.py` | ✅ 成功 |
| **フェーズ3** | **エネルギー駆動型 / 量子インスパイアード** | | | |
| | エネルギー駆動型コンピューティング (`EnergyManager`) | ✔️ | `tests/test_energy.py` | ✅ 成功 |
| | 動作検証デモ | ✔️ | `examples/run_energy_demo.py` | ✅ 成功 |
| | 量子インスパイアード機能 | | | |
| | - `EntangledSynchronyLayer` | ✔️ | `tests/test_quantum_layers.py` | ✅ 成功 |
| | - `HardwareFitnessEvaluator` | ✔️ | `tests/test_fitness.py` | ✅ 成功 |
| | - `GraphAnnealingRule` | ✔️ | `tests/test_annealing.py` | ✅ 成功 |
| **フェーズ4** | **テキスト学習 - エンコーディング手法の確立** | | | |
| | 単語埋め込み (Word Embedding) レイヤー | ✔️ | `tests/test_text.py` | ✅ 成功 |
| | エンコーディングモジュール (`RateEncoder`, `LatencyEncoder`) | ✔️ | `tests/test_text.py` | ✅ 成功 |
| | 位置エンコーディング | ✔️ | `tests/test_text.py` | ✅ 成功 |
| **フェーズ5** | **テキスト学習 - スパイキングTransformer** | | | |
| | スパイキング自己アテンション (Spiking Self-Attention) | ✔️ | `tests/test_transformer.py` | ✅ 成功 |
| | ネットワークコンポーネント (残差接続, レイヤー正規化) | ✔️ | `tests/test_transformer.py` | ✅ 成功 |
| **フェーズ6** | **テキスト学習 - 勾配ベース学習** | | | |
| | 代理勾配 (Surrogate Gradient) の実装 | ✔️ | `evospikenet/surrogate.py`, `tests/test_surrogate.py` | ✅ 成功 |
| | 損失関数とデコーディング | ✔️ | `examples/run_gradient_training_demo.py` | ✅ 成功 |
| **フェーズ7** | **テキスト学習 - 統合と実験** | | | |
| | モデルの統合 | ✔️ | `evospikenet/models.py`, `tests/test_models.py` | ✅ 成功 |
| | 実験の実施 | ✔️ | `examples/run_text_classification_experiment.py` | ✅ 成功 |
| **フェーズ8** | **分散ニューロモーフィックグリッド** | | | |
| | 分散モデルの構築 (`DistributedEvoSpikeNet`) | ❌ | (未実装) | ❌ 未実行 |
| | 非同期通信 (`SpikeCommunicator`) | ❌ | (未実装) | ❌ 未実行 |
| **フェーズEX**| **LLMデータ蒸留** | | | |
| | データ蒸留モジュール (`DataDistiller`) | ✔️ | `evospikenet/distillation.py` | ✅ 成功 |
| | データ生成デモ | ✔️ | `examples/generate_distilled_dataset.py` | ✅ 成功 |
| **フェーズ9-12**| **フロントエンドUI** | | | |
| | マルチページUIへのリファクタリング | ✔️ | `frontend/app.py`, `frontend/pages/` | ✅ 成功 |
| | - ホームページ (マニュアル表示) | ✔️ | `frontend/pages/home.py` | ✅ 成功 |
| | - SNNデータ作成ページ | ✔️ | `frontend/pages/data_creation.py` | ✅ 成功 |
| | - モデル可視化ページ | ✔️ | `frontend/pages/visualization.py` | ✅ 成功 |
| | - EvoSpikeNetLMページ | ✔️ | `frontend/pages/evospikenet_lm.py` | ✅ 成功 |
| | - SpikingEvoSpikeNetLMページ | ✔️ | `frontend/pages/spiking_lm.py` | ✅ 成功 |
| | - マルチモーダルLMページ | ✔️ | `frontend/pages/multi_modal_lm.py` | ✅ 成功 |
| | - テキスト分類ページ | ✔️ | `frontend/pages/text_classifier.py` | ✅ 成功 |
| | - システムユーティリティページ | ✔️ | `frontend/pages/system_utils.py` | ✅ 成功 |
| | - RAGシステムページ | ✔️ | `frontend/pages/rag.py` | ✅ 成功 |
| |   - Milvusデータ管理 (CURD) | ✔️ | `frontend/pages/rag.py` | ✅ 成功 |
| **フェーズSNN-LM-DATA**| **SNN-LM - データパイプライン** | | | |
| | 外部データローダー (`Wikipedia`, `Aozora`) | ✔️ | `evospikenet/dataloaders.py` | ✅ 成功 |
| | `TAS-Encoding` (テキスト→スパイク変換) | ✔️ | `evospikenet/encoding.py`, `tests/test_encoding.py` | ✅ 成功 |
| **フェーズSNN-LM-MODEL**| **SNN-LM - モデルアーキテクチャ** | | | |
| | `ChronoSpikeAttention` (ハイブリッド) | ✔️ | `evospikenet/attention.py`, `tests/test_attention.py` | ✅ 成功 |
| | `SpikingTransformerBlock` | ✔️ | `evospikenet/attention.py`, `tests/test_attention.py` | ✅ 成功 |
| | `SpikingEvoSpikeNetLM` (モデル統合) | ✔️ | `evospikenet/models.py` | ✅ 成功 |
| | `AEG`, `MetaSTDP` (制御機構) | ✔️ | `evospikenet/control.py`, `tests/test_control.py` | ✅ 成功 |
| **フェーズSNN-LM-TRAIN**| **SNN-LM - 訓練と評価** | | | |
| | 訓練・評価スクリプト | ✔️ | `examples/train_spiking_evospikenet_lm.py` | ✅ 成功 |
| | ハイパーパラメータ調整スクリプト | ✔️ | `scripts/run_hp_tuning.sh` | ✅ 成功 |
| **フェーズSNN-LM-VIZ**| **SNN-LM - 可視化** | | | |
| | 内部スパイク活動の可視化 | ✔️ | `frontend/pages/spiking_lm.py` | ✅ 成功 |
| | アテンション重みの可視化 | ✔️ | `frontend/pages/spiking_lm.py` | ✅ 成功 |

**凡例:**
*   ✔️: 実装済み
*   ❌: 未実装
*   ✅: テスト成功
*   ⚠️: 未検証 (環境要因によりテスト実行不可)
*   (N/A): 対象外

---
## 次期開発計画: マルチモーダル SNN

| フェーズ | 機能 | 実装状況 | テスト/検証スクリプト | テスト状況 |
| :--- | :--- | :---: | :--- | :---: |
| **フェーズMM-1**| **ビジョン・エンコーダー** | | | |
| | 畳み込みSNNエンコーダーの実装 | ✔️ | `evospikenet/vision.py`, `tests/test_vision.py` | ✅ 成功 |
| **フェーズMM-2**| **マルチモーダル・モデル統合** | | | |
| | テキストとビジョンの特徴量融合 | ✔️ | `evospikenet/models.py` | ✅ 成功 |
| | `MultiModalEvoSpikeNetLM` の構築 | ✔️ | `tests/test_models.py` | ✅ 成功 |
| | **学習・推論パイプラインの実装** | ✔️ | `examples/train_multi_modal_lm.py` | ✅ 成功 |
| | **注記:** | `train_multi_modal_lm.py`は、コマンドラインから学習と推論を実行できる完全なパイプラインとして実装済み。カスタムデータローダー、モデルの保存・読込機能を含む。 |

---
## 次期開発計画: マルチモーダル機能の完成

| フェーズ | 機能 | 実装状況 | テスト/検証スクリプト | テスト状況 |
| :--- | :--- | :---: | :--- | :---: |
| **フェーズMM-3**| **UI統合とインタラクティブ推論** | | | |
| | Web UIへの画像アップロード機能の追加 | ❌ | (未実装) | ❌ 未実行 |
| | アップロードされた画像に対するキャプション生成機能 | ❌ | (未実装) | ❌ 未実行 |
| **フェーズMM-4**| **評価とベンチマーキング** | | | |
| | BLEU, ROUGE等の評価指標を用いた性能評価スクリプト | ❌ | (未実装) | ❌ 未実行 |
| | 大規模データセット（例: COCO）での学習と評価 | ❌ | (未実装) | ❌ 未実行 |
