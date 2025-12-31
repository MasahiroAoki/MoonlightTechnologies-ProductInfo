<!-- Reviewed against source: 2025-12-31. English translation pending. -->
# リリースノート

EvoSpikeNetフレームワークのバージョン履歴と変更内容です。

## 📋 バージョン履歴

---

## [v0.1.0] - 2025-12-17

### 🎉 Initial Release - Production Ready

**ステータス**: 🟢 Production Ready

### ✨ 主要機能

#### 分散脳システム
- **Zenoh通信**: 高速非同期メッセージング
- **ノード発見**: 自動的なノードディスカバリーとヘルスチェック
- **PFCフィードバック**: 前頭前野による高度な意思決定

#### RAGシステム
- **ハイブリッド検索**: Milvus（ベクトル検索）+ Elasticsearch（フルテキスト検索）
- **日本語最適化**: 形態素解析による高精度検索
- **デバッグ機能**: エラー追跡とパフォーマンス分析の可視化

#### SDK & API
- **型安全性**: 完全な型ヒント、Enum、Dataclass対応
- **エラーハンドリング**: カスタム例外、自動リトライ、接続プーリング
- **Jupyter統合**: マジックコマンド、リッチHTML出力
- **検証ツール**: APIValidator、パフォーマンスメトリクス、ベンチマーク

#### Web UI
- **3D可視化**: Three.jsによるニューロンビジュアライゼーション
- **リアルタイムモニタリング**: ニューロン活動の動的表示
- **管理機能**: データセット・モデル管理、設定UI

#### CI/CD
- **GitHub Actions**: 自動テスト、ビルド、デプロイ
- **Docker統合**: 自動コンテナビルド
- **品質保証**: テストカバレッジ92%以上

### 🔧 技術仕様

#### アーキテクチャ
- **SNNモデル**: LIF、AdEx、Izhikevichニューロン
- **可塑性**: STDP学習規則
- **並列処理**: マルチプロセス・非同期処理対応

#### データベース
- **Milvus**: ベクトルデータベース（コサイン類似度検索）
- **Elasticsearch**: フルテキスト検索エンジン
- **RRF融合**: Reciprocal Rank Fusionによる結果統合

#### API
- **REST API**: FastAPIベースの高速API
- **認証**: JWT認証対応
- **OpenAPI**: Swagger自動生成

### 🐛 主なバグ修正

- RAGシステムのUnboundLocalError修正（re import）
- rag()メソッドの返値不整合修正
- Web UIポート衝突エラー解決
- Dockerボリューム権限問題解決

### 📚 ドキュメント

- **統合ドキュメントサイト**: MkDocs + Material theme
- **多言語対応**: 日本語・英語バイリンガル（mkdocs-static-i18n）
- **80+ ページ**: 包括的なガイド・リファレンス
- **検索機能**: 全文検索対応

### 📊 統計

- **総コード行数**: 50,000+ 行
- **テストカバレッジ**: 92%
- **ドキュメントページ**: 80+ ページ
- **実装完了率**: 100%

### 🚀 インストール

```bash
# PyPIからインストール（予定）
pip install evospikenet

# または開発版
git clone https://github.com/moonlight-tech/EvoSpikeNet.git
cd EvoSpikeNet
pip install -e .
```

### 💻 クイックスタート

```python
from evospikenet import EvoSpikeNetSDK

# SDKクライアント初期化
sdk = EvoSpikeNetSDK(api_url="http://localhost:8000")

# RAG検索
results = sdk.rag_search(
    query="スパイキングニューラルネットワークとは？",
    top_k=5
)

for doc, score in results:
    print(f"スコア: {score:.3f}, 文書: {doc}")
```

---

## 📊 バージョン比較

| バージョン | リリース日 | 主要機能                        | ステータス   |
| ---------- | ---------- | ------------------------------- | ------------ |
| **v0.1.0** | 2025-12-17 | 分散脳、RAG、SDK、Web UI、CI/CD | 🟢 Production |

---

## 📧 フィードバック

バグ報告や機能要望は[GitHubイシュー](https://github.com/moonlight-tech/EvoSpikeNet/issues)でお願いします。

---

## [v0.1.1] - 2025-12-21

### 🧪 テストインフラ強化

**マルチモダリティ統合テスト実装**
- **クロスモダリティテスト**: Vision/Audio/Textの統合テスト
- **E2Eパイプラインテスト**: エンドツーエンドの完全テスト
- **干渉テスト**: モダリティ欠落時の動作検証
- **フォールバックテスト**: 品質劣化時の回復機能テスト
- **実装ファイル**: `tests/integration/test_multimodal_integration.py`

### 📚 ドキュメント更新
- **REMAINING_FEATURES.md**: 実装状況の詳細更新、重複項目削除
- **features.md**: 機能ステータス表の更新（実装率86%）
- **表フォーマット統一**: 全ドキュメントの表カラムサイズ統一

### 🔧 内部改善
- **ドキュメント一貫性**: 実装済み/未実装項目の明確化
- **テストカバレッジ**: 統合テストの追加による品質向上

---

## [v0.1.2] - 2025-12-31

### 🧠 長期記憶ノード実装 ⭐ NEW

**分散脳シミュレーション向け長期記憶システム**
- **`LongTermMemoryNode`**: FAISSベクトル検索とZenoh分散通信を統合した基底クラス
- **`EpisodicMemoryNode`**: 時系列イベント（体験）の保存と検索
- **`SemanticMemoryNode`**: 概念・知識の永続化と関連付け
- **`MemoryIntegratorNode`**: エピソード/セマンティック記憶のクロスモーダル統合
- **実装ファイル**: `evospikenet/memory_nodes.py` (355行)
- **テスト**: `tests/test_memory_nodes.py` (242行)

### 🎵 音声トレーニングUI改善

**統一化されたオーディオトレーニングコールバック**
- 複数ページ間での重複コールバック登録を解消
- 共有コールバックシステムによる保守性向上
- **実装ファイル**: `frontend/pages/audio_training_callbacks.py`

### 📚 ドキュメント更新

**新規英語版翻訳 (10ファイル)**
- DEV_BATCH_SHAPING.en.md
 - DISTRIBUTED_BRAIN_24NODE_EVALUATION.en.md
- DISTRIBUTED_BRAIN_MODEL_ARTIFACTS.en.md
- DISTRIBUTED_BRAIN_NODE_TYPES.en.md
- DISTRIBUTED_BRAIN_SEQUENCE.en.md
- LLM_TRAINING_SYSTEM.en.md
- SDK_CONFIGURATION.en.md
- api/README.en.md
- implementation/ARTIFACT_MANIFESTS.en.md

**更新済みドキュメント**
- DISTRIBUTED_BRAIN_SEQUENCE.md: 24ノードフル脳アーキテクチャと長期記憶統合
- REMAINING_FEATURES.md: 実装状況更新
- EPISODIC_MEMORY_IMPLEMENTATION.md: 新規記憶ノード情報追加

### 🔧 内部改善
- RAGシステムのフロントエンド改善
- Vision/Audioエンコーダの改善 (800+ 行更新)
- 分散脳の24ノード評価ドキュメント

---

**最終更新**: 2025年12月31日  
**現在のバージョン**: v0.1.2
