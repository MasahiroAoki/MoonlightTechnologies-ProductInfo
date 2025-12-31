````markdown
<!-- Reviewed against source: 2025-12-21. English translation pending. -->
<!-- Copyright: 2025 Moonlight Technologies Inc. All Rights Reserved. -->
<!-- Author: Masahiro Aoki -->

# 24ノード構成の検証レポート

作成日: 2025-12-21

## 概要
- 対象: 分散脳シミュレーションにおける「24ノードフルブレイン」構成の妥当性検証
- 実施: リポジトリ内のスケーラビリティテスト (`evospikenet.scalability_test.ScalabilityTester`) を用いた短時間シミュレーション（24ノード、5秒）とコードレビュー

> 実装ノート（アーティファクト）: シミュレーションやテストで生成されるモデルアーティファクトや `artifact_manifest.json` 仕様については `docs/implementation/ARTIFACT_MANIFESTS.md` を参照してください。

## 実行した軽量シミュレーション結果（要約）
```json
{"num_nodes":24,
 "duration":5.081160306930542,
 "metrics":{
   "throughput":2.4956167570360264,
   "avg_latency":400.7025506543207,
   "max_latency":1357.4155449896352,
   "cpu_usage":4.766666666666667,
   "memory_usage":36.03333333333333,
   "errors":0
 }
}
```

注意: 上記は短時間の合成シミュレーション結果であり、本番的負荷／モデル重み／I/O要件を反映していません。完全評価は長時間・実ワークロードでの再試行が必要です。

## コードレビューでの主要指摘
- `DistributedMotorConsensus._determine_consensus` 内の閾値計算:
  - 現状: `required_votes = int(self.num_nodes * self.consensus_threshold)`（小数切り捨て）
  - 問題: 切り捨てにより実際のクォーラムが想定より低くなり得る（例: 24 * 0.67 = 16.08 → int → 16）。運用上は切り上げ（ceil）で安全側にするべき（`math.ceil` を推奨、上記例では 17）。

## 妥当性評価（設計観点）
- 機能カバレッジ: 24ノードは小〜中規模の分散脳実験に適合する。観測→エンコード→推論→意思決定→記憶／学習をカバーする各役割を配置できる。
- 冗長性: 単一障害点（aggregator, vector DB, federator 等）を必ず冗長化する。重要ノードは複数インスタンス（ミラー）を推奨。
- コンセンサス: コンセンサスアルゴリズムはノード数に依存するため、クォーラム計算の厳密化が必須（ceil利用、または明示的閾値設定）。
- リソース配分: 推論/エンコーダはGPU割当、記憶ノードは専用ストレージ（Milvus/DB）、学習ノードは高スループット帯域とチェックポイント領域を確保。
- ネットワーク: 低遅延のネットワーク（10GbE以上想定）と帯域管理を行うこと。
- セキュリティ: 機密データ／制御コマンドには強い認証（APIキー／TLS／RBAC）と監査ログを適用。

## 推奨構成（24ノードの一例）
合計24ノードを以下の役割に配分する案（用途・冗長性を考慮）:

- 観測ノード（Sensing）: 4
- エンコードノード（Encoders）: 4
- 推論ノード（Inference / LM）: 6
- 意思決定／行動ノード（Planner/Controller）: 2
- 記憶ノード（Vector DB / Retriever）: 3
- 学習ノード（Trainer / Updater）: 1
- 集約／調停ノード（Aggregator / Federator）: 2
- 管理／ユーティリティ（Monitoring / Auth / Logging）: 2

この配分は用途により調整可能。例えば視覚系重視なら観測／エンコーダを増やし、言語系重視なら推論ノードを増加。

## 即時対応の推奨アクション
1. `DistributedMotorConsensus._determine_consensus` の `int(...)` を `math.ceil(...)` に変更してクォーラムを安全側へ修正する。
2. 本番に近い負荷（実モデル・I/O）で `ScalabilityTester.run_full_scalability_test` を実行し、24ノード時のスループット／遅延／エラー状況を取得する。
3. 重要サービス（vector DB、aggregator、auth、storage）は冗長構成と監視を導入する（レプリカ数を明示化）。
4. ノードごとのリソース要件（CPU/GPU/メモリ/ネットワーク）をドキュメント化して配備計画を作成する。

## 次のステップ（提案）
- 本格テスト実行（`run_full_scalability_test`）を行い、結果をこのドキュメントに追記します。
- 上記の推奨修正（クォーラム計算）をパッチとしてコミット／PR作成できます。適用しますか？

````
