<!-- Reviewed against source: 2025-12-21. English translation pending. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# 完全脳（フルブレイン）実装のための分散脳ノード：概念・構成・24ノードパイプライン

このドキュメントは、分散脳（Distributed Brain）を「完全脳」として実装するための設計書です。概念、全体概要、プロセス構成、24ノードによる具体的プロセスパイプライン、各ノードに割り当てるモデル（LLM/エンコーダ等）・データ・学習方法までをまとめ、完全脳が提供する機能を明確にします。

---

## 1. 概念（Concept）
- 完全脳は、生体脳の機能分化（感覚→エンコーディング→認知→意思決定→運動）を分散ノードとして実装し、各ノードが専門性を持ちながら協調して高次機能を実現するシステムです。
- 各ノードは「モデル（LLM/エンコーダ/検出器）」と「データ／メモリ／通信チャネル」を持ち、低遅延でメッセージ（観測・埋め込み・推論結果・指令）をやり取りします。

> 実装ノート（アーティファクト）: ノードごとのモデルアーティファクト生成に関しては、`docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。`artifact_manifest.json` の必須項目と CLI フラグ仕様（`--artifact-name` / `--node-type` など）を記載しています。

## 2. 全体概要（Overview）
- 層構造:
	- 観測層：物理センサからデータを取得
	- エンコード層：観測データを特徴量／埋め込みに変換
	- 認知・推論層：埋め込みを用い意味理解・分類・生成を行う
	- 記憶層：コンテキスト・履歴を保持し検索を提供
	- 学習層：モデルの継続学習・微調整を担う
	- 意思決定層：高レベル目標から行動を決定、アクチュエータに出力
	- 管理層：認証、監視、ログ、設定配布

## 3. プロセス構成（Process Composition）
- メッセージ形式: 標準化されたメタデータ付きペイロード（timestamp, node_id, model_version, embedding_dims, confidence）を採用
- 通信: Pub/Sub（Zenoh等）とREST/gRPCのハイブリッド。重要操作は認証付き（`X-API-Key` / mTLS）。
- 一貫性: 状態は最小限にし、イベント駆動で処理。長期状態は記憶ノードに保存。

## 4. 24ノードプロセスパイプライン（提案）
以下は「完全脳」を24ノードで表現する具体例です。各ノードは役割・割当モデル・主なデータソース・学習方法を示します。

### ノード配分（要約）
- 観測（Sensing）: 4ノード
- エンコーダ（Encoders）: 4ノード
- 推論（Inference / LM）: 6ノード
- 意思決定（Decision）: 2ノード
- 記憶（Memory / Retriever）: 3ノード
- 学習（Trainer）: 1ノード
- 集約／調停（Aggregator / Federator）: 2ノード
- 管理／ユーティリティ（Management）: 2ノード

### 各ノードの詳細

- ノード1-3: 観測ノード（Sensing x3）
	- 役割: カメラ入力、マイク入力、環境センサ（温度/IMU等）を収集し初期フィルタ・同期を行う
	- モデル: 軽量前処理（ノイズ除去、正規化）。場合によってはオンデバイス簡易エンコーダ
	- データ: カメラストリーム（映像）、マイク（WAV）、IoTセンサ時系列
	- 学習: データ拡張・自己教師ありフィルタ（ノイズ耐性向上）

- ノード4: Vision Encoder
	- 役割: 画像→埋め込み変換
	- モデル: ViT/Vision Transformer 系 or ResNet→Projection、またはイベント向け Spiking-ViT
	- データ: ImageNet, COCO, ドメイン固有データ（収集時のメタデータ付き）
	- 学習: 事前学習（大規模データ）→ドメイン微調整（Fine-tune）、場合により継続学習

- ノード5: Audio Encoder
	- モデル: Wav2Vec2 / HuBERT →埋め込み
	- データ: LibriSpeech, AudioSet, ドメイン音声コーパス
	- 学習: 事前学習 + タスク微調整（音声分類・転写）

- ノード6: Text Encoder
	- モデル: SentenceTransformer（SBERT系）または軽量トランスフォーマー埋め込み
	- データ: Wikipedia, CC-News, 専門ドメインコーパス
	- 学習: 事前学習→タスク微調整（意味検索用）

- ノード7: Spiking Encoder
	- モデル: SNN（Spiking Neural Network）ベースのエンコーダ（イベントカメラ向け）
	- データ: DVS (Dynamic Vision Sensor) データセット等
	- 学習: STDP/Surrogate-gradient training / 変換学習

- ノード8-12: 推論ノード（Inference x5）
	- ノード8: LM-Inference（短文/対話）
		- モデル: 小中規模のトランスフォーマーLM（数百M〜数十億パラメータ）
		- データ: 会話コーパス、システムプロンプト、履歴
		- 学習: 事前学習済みモデルのオンライン微調整（リトレーニングはTrainerノードが管理）

	- ノード9: Classifier/Detector
		- モデル: YOLOvX / Faster-RCNN / ResNet-based classifier
		- データ: COCO, OpenImages, 専用アノテーション
		- 学習: 転移学習＋継続的ラベリング（Human-in-the-loop）

	- ノード10: Spiking-LM（生体指向生成）
		- モデル: スパイキングニューラルネットワークを用いた生成/記憶インタフェース
		- データ: センサ時系列+イベント履歴
		- 学習: 生体模倣のオンライン適応（少量データでの微調整）

	- ノード11: Ensemble / Multimodal Inference
		- 役割: エンコーダ群／推論ノードの出力を統合して高信頼度出力を生成
		- 手法: 重み付きアンサンブル、メタ学習による信頼度推定

	- ノード12: Retriever-Augmented Generation (RAG)
		- 役割: 記憶ノードからのコンテキストを付与してLM推論を支援
		- モデル: 軽量検索器＋LM

	- ノード13-14: 長期記憶ノード（Long-Term Memory x2）
		- 役割: エピソディックメモリ（出来事ベース）とセマンティックメモリ（知識ベース）を管理
		- モデル: FAISSベースのベクトル検索、Zenoh通信統合
		- データ: スパイク埋め込み、時系列イベント、メタデータ
		- 学習: オンライン適応、重要度ベースの保持/忘却
		- 機能: 類似検索、連想想起、メモリ統合

- ノード17-18: 意思決定ノード（Decision x2）
	- ノード17: 高レベルプランナー
		- 役割: 目標を受け、複数候補（サブゴール）を生成
		- モデル: 強化学習ベースのポリシー or Symbolic Planner + 学習済みポリシー
		- 学習: シミュレーションでの強化学習（PPO/IMPALA 等）＋現地微調整

	- ノード18: 実行コントローラ
		- 役割: プランをモーター指令に変換（MotorConsensusと連携）
		- モデル: 既存モーターモデル＋分散合意によるアクチュエータ出力の協調

- ノード19-20: 記憶ノード（Memory x2）
	- ノード19: ベクトルDB（Milvus/FAISS）
		- 役割: 埋め込みの格納と高速近傍検索
		- データ: 埋め込み、メタデータ、シグナルの時間情報
		- 運用: レプリカ、シャーディング、TTLポリシー

	- ノード16: エピソード記憶（時系列ストレージ）
		- 役割: 生のイベントログ・トランザクション保存、長期履歴
		- ストレージ: MinIO / Time-series DB

- ノード21: 学習ノード（Trainer x1）
	- 役割: モデルのバッチ学習／分散トレーニング／フェデレーテッド集約の指揮
	- 手法: PyTorch DDP / Horovod / Federated Averaging（サーバ側）
	- 機能: チェックポイント、メトリクス集約、A/Bテスト用のモデル配信

- ノード22-23: 集約／調停ノード（Aggregator x2）
	- ノード22: フェデレータ（Federator）
		- 役割: フェデレーテッド学習の集約（安全な集計、差分プライバシーの適用）
	- ノード23: 結果アグリゲータ
		- 役割: 複数ノードの出力を統合（重み付け、信頼度管理）、ポリシー決定支援

- ノード24-25: 管理／ユーティリティ（Management x2）
	- ノード24: 認証／認可／構成配布
		- 役割: APIキー管理、RBAC、TLS証明書管理
	- ノード25: 監視・ロギング
		- 役割: Prometheus/Grafanaでメトリクス可視化、ELKでログ集約

---

## 5. 各ノードのモデル割当と学習方法（詳細）
- 事前原則:
	- 主要モデルは「事前学習（Pretrain）→ドメイン微調整（Fine-tune）→運用中継続学習（Continual / Federated）」のパスを採る。
	- プライバシー保護はデフォルトで適用（差分プライバシー、Secure Aggregation、暗号化転送）。

- データパイプライン:
	- 観測→エンコーダ→ベクトルDB／推論 のパスはストリーミングを基本とし、同時にサンプリングしたデータはTrainerへバッチで供給。
	- アノテーションはHuman-in-the-loopで段階的に追加し、品質保証されたデータのみを学習に投入。

- 学習方式（ノード別）:
	- エンコーダ（Vision/Audio/Text/Spiking）: 大規模事前学習（分散GPU）→ドメイン微調整（少量データで高速）→エッジでの知識蒸留
	- 推論ノード（LM等）: 事前学習済みLMのパラメータの一部を固有タスクで微調整。オンライン微調整はTrainerの承認下で行う。
	- Trainer（ノード17）: モデル重みの集約、検証、モデル署名、配信。フェデレーテッド学習時はAggregatorと共同で安全に集約。

## 6. 運用上の注意（安全性・冗長性・監視）
- 冗長性: 重要ノード（記憶、aggregator、auth）は複数レプリカで運用し、自動フェイルオーバーを設定
- セキュリティ: mTLS、RBAC、APIキー回転、監査ログの保存
- モニタリング: レイテンシ、スループット、メモリ/GPU使用率、訓練メトリクスを収集

## 7. 完全脳の機能（End-to-End）
- 観測: センサがデータを収集し、エンコーダに渡す
- 知覚: エンコーダが特徴を抽出、ベクトルDBへ格納し、同時に推論ノードへ提供
- 推論: RAGやLMで文脈付き応答や分類を実行
- 記憶・検索: ベクトルDBが類似コンテキストを返し、長期のエピソード記憶が履歴を提供
- 学習: Trainerが新しいデータでモデルを更新し、Aggregatorで安全に集約して配信
- 意思決定・行動: Decisionノードが行動計画を作成し、MotorConsensusを通じてアクチュエータを制御

---

## 8. 次のアクション（推奨）
1. 上記ノードごとに実装可能なモデルアーティファクト名（例: `resnet50-v2`, `wav2vec2-large`, `gpt-small-v1`）の候補リスト化
2. 24ノードの上でのスケジューリング（K8s）テンプレート作成（CPU/GPU/メモリ要求を含む）
3. フルスケール負荷試験計画とCI統合（自動ベンチマーク）

この文書を基に、さらに詳しいノード実装仕様書（API、メッセージスキーマ、モデルバージョニングポリシー等）を作成できます。希望するアウトプット（例えば `docs/` 内の別ファイル、CSV、PR作成）を教えてください。

---

## 使い方
- 各セクションは「分類（機能）」を示します。
- `カテゴリ例` は実装やモデル登録時に使える候補ラベルです。

---

## 観測ノード (Sensing)
- カテゴリ例: `vision`, `audio`, `sensor`
- ノード機能: 環境データの取り込み・前処理（画像/音声/センサ値）
- 入力: 生データストリーム（カメラ、マイク、IoTセンサ）
- 出力: 正規化された特徴量、エンコーダ入力（テンソル/バイナリ）
- 必要資源: CPU/GPU、低遅延I/O、変換ライブラリ（OpenCV, librosa等）
- 権限: センサデータ取得の認可、プライバシー制御
- 備考: フィルタ／サンプリングやローカルプライバシー処理を推奨

## エンコードノード (Encoding / Feature Extractor)
- カテゴリ例: `vision-encoder`, `audio-encoder`, `text-encoder`, `spiking-encoder`
- ノード機能: 生データを埋め込み／低次元表現へ変換
- 入力: 観測ノード出力、生データ
- 出力: 埋め込みベクトル、特徴テンソル
- 必要資源: GPU、モデルアーティファクト、バッチ処理パイプライン
- 権限: モデル利用許可（ライセンス/APIキー）
- 備考: 埋め込み次元・フォーマットの互換性を管理

## 理解／推論ノード (Perception / Inference)
- カテゴリ例: `classifier`, `detector`, `lm-inference`, `spiking-lm`
- ノード機能: 分類、検出、生成などの推論処理
- 入力: 埋め込み、プロンプト、コンテキスト
- 出力: ラベル/スコア、生成テキスト、確信度
- 必要資源: 大型モデル（GPU/TPU）、低レイテンシ推論環境
- 権限: 機密データの扱い制御、認証（`X-API-Key`等）
- 備考: レイテンシ制約に応じ分散推論を設計

## 意思決定／行動ノード (Decision / Actuation)
- カテゴリ例: `planner`, `policy`, `controller`, `action-executor`
- ノード機能: 推論結果に基づく行動決定、外部システム制御
- 入力: 推論結果、目標、制約
- 出力: 制御コマンド、行動プラン、API呼び出し
- 必要資源: リアルタイム通信、セーフガード、認可フロー
- 権限: アクチュエータ制御の多段認可
- 備考: フェイルセーフ、ログ記録を必須とする

## 記憶ノード (Memory / Storage / Retriever)
- カテゴリ例: `episodic-memory`, `vector-db`, `retriever`, `knowledge-base`
- ノード機能: 長短期記憶の保持・検索、履歴管理
- 入力: 生成結果、センサ履歴、メタデータの書き込み要求
- 出力: 検索結果、コンテキスト断片（テキスト/埋め込み）
- 必要資源: 永続ストレージ（MinIO、DB）、ベクトルDB（Milvus/FAISS）
- 権限: データアクセス制御、暗号化、監査ログ
- 備考: TTL／サニタイズ、アクセス制御ポリシーを定義

## 学習ノード (Learning / Trainer / Updater)
- カテゴリ例: `trainer`, `federated-learner`, `fine-tuner`
- ノード機能: オンライン／バッチ学習、モデル更新、フェデレーテッド学習
- 入力: トレーニングデータ、検証データ、ハイパーパラメータ
- 出力: モデルアーティファクト、メトリクス、学習ログ
- 必要資源: GPUクラスタ、データ転送帯域、チェックポイント領域
- 権限: モデル署名／承認ワークフロー、アップロード権限
- 備考: 安全なモデル配布とロールバック機構を実装

## 集約／調停ノード (Aggregator / Orchestrator)
- カテゴリ例: `federator`, `aggregator`, `coordinator`
- ノード機能: 複数ノードの出力集約、重み付け、合意形成
- 入力: 各ノードの出力、メタステータス、ヘルス情報
- 出力: 集約決定、ルーティング指示、統計メトリクス
- 必要資源: 低レイテンシ通信、状態管理、トランザクション制御
- 権限: ノード間通信の認証、トラストポリシー
- 備考: フェイルオーバー戦略とバージョン互換性を含む

## ユーティリティ／管理ノード (Utility / Management)
- カテゴリ例: `monitoring`, `logging`, `health-check`, `auth`
- ノード機能: ロギング、監視、ヘルスチェック、APIキー管理
- 入力: メトリクス、ログイベント、設定変更要求
- 出力: アラート、ダッシュボード用データ、認証トークン
- 必要資源: 時系列DB、監視ツール、認証基盤
- 権限: 管理者ロール制御、監査ログ保持ポリシー
- 備考: セキュリティと可観測性を最優先

---

## 次の提案
- このMarkdownをベースに、`node-types` / `model-categories` の既存値を反映して個別候補を生成できます。
- CSV形式やリポジトリの `docs/` への自動追加が必要なら指示ください。

---

作成日: 2025-12-27

---

## 実装済み機能（Implemented Features）

### 長期間記憶システム（Long-Term Memory System）
以下の長期間記憶関連機能が実装済みです：

#### 1. メモリノードクラス
- **LongTermMemoryNode**: 基底クラス、FAISSベクトル検索とZenoh通信統合
- **EpisodicMemoryNode**: 時系列イベントベースのエピソディック記憶
- **SemanticMemoryNode**: 事実知識ベースのセマンティック記憶  
- **MemoryIntegratorNode**: エピソード記憶とセマンティック記憶の統合・連想

#### 2. コア機能
- **ベクトル類似性検索**: FAISSを使用した高速近傍検索
- **Zenoh分散通信**: ノード間メモリ操作のPub/Sub通信
- **PTP時刻同期**: ナノ秒精度のタイムスタンプ生成（テスト環境ではシステム時間にフォールバック）
- **メモリ統合**: 重要度ベースの保持・忘却ポリシー
- **クロスモーダル連想**: 異なる記憶タイプ間の関連付け

#### 3. 実装ファイル
- `evospikenet/memory_nodes.py`: メモリノードの実装
- `examples/run_zenoh_distributed_brain.py`: 分散脳へのメモリ統合
- `tests/test_memory_nodes.py`: 包括的なテストスイート（9テストケース）
- `requirements.txt`: FAISS依存関係追加
- `Dockerfile`: FAISSインストール設定

#### 4. テスト結果
```
================== test session starts ==================
collected 9 items

tests/test_memory_nodes.py::TestLongTermMemoryNode::test_initialization PASSED
tests/test_memory_nodes.py::TestLongTermMemoryNode::test_store_memory PASSED  
tests/test_memory_nodes.py::TestLongTermMemoryNode::test_query_memory PASSED
tests/test_memory_nodes.py::TestLongTermMemoryNode::test_retrieve_memory PASSED
tests/test_memory_nodes.py::TestEpisodicMemoryNode::test_store_episodic_sequence PASSED
tests/test_memory_nodes.py::TestSemanticMemoryNode::test_store_knowledge PASSED
tests/test_memory_nodes.py::TestMemoryIntegratorNode::test_associate_memories PASSED
tests/test_memory_nodes.py::TestMemoryEntry::test_memory_entry_creation PASSED
tests/test_memory_nodes.py::TestMemoryEntry::test_memory_entry_serialization PASSED

================== 9 passed, 4 warnings in 0.33s ==================
```

#### 5. 分散脳統合
- 24ノードアーキテクチャへのメモリノード追加
- Zenoh通信プロトコルを使用したリアルタイムメモリ操作
- 長期学習と適応のための永続的知識保持

---

## 次の実装予定（Remaining Features）
1. **ベクトルDB統合**: Milvusなどの分散ベクトルデータベースへの移行
2. **メモリ最適化**: 大規模メモリセットの効率的処理とシャーディング
3. **学習統合**: メモリからの経験再生による継続学習
4. **セキュリティ強化**: メモリデータの暗号化とアクセス制御
5. **パフォーマンス監視**: メモリ操作のレイテンシとスループット監視
6. **分散合意**: 複数メモリノード間の一貫性確保プロトコル
