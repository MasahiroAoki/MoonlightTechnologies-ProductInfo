# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# ARGUS-System 実装状況レビュー

ARGUS-System概要設計書に基づき、現在のソースコードベースの機能実装状況を評価しました。
各機能について、実装ステータス、担当サービス、評価メモ、テスト実装状況を記載します。

---

## 1. 録画機能 (Recording)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **連続録画** | <span style="color:green;">**実装済み**</span> | `recording-service` | `ffmpeg` を利用し、10秒単位のMP4ファイルとして効率的に連続録画。自己修復機能も実装済み。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **イベント録画** | <span style="color:green;">**実装済み**</span> | `recording-service`, `management-service` | AIイベントに連動して指定時間録画を開始する機能を実装。Redis Pub/Subを介してイベントを連携。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **H.265/H.264対応** | <span style="color:green;">**実装済み**</span> | `recording-service` | `-c copy`により入力コーデックをそのまま保存。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **ANR** | <span style="color:red;">**未実装**</span> | `recording-service` | NW障害時にカメラ内ストレージから録画を補完する機能は存在しない。 | <span style="color:red;">テストなし</span> | フェーズ1 |

## 2. リアルタイム表示機能 (Viewing)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **低遅延表示(WebRTC)** | <span style="color:green;">**実装済み**</span> | `streaming-service`, `web-frontend` | Pion(Go)とReactを用いたWebRTCでの配信が実装済み。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **ライブ/再生表示** | <span style="color:green;">**実装済み**</span> | `streaming-service`, `web-frontend` | ライブ映像と録画ファイルの両方を同じコンポーネントで再生可能。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **ビュー管理** | <span style="color:green;">**実装済み**</span> | `web-frontend`, `streaming-service` | 1x1, 2x2, 3x3, 4x4の画面分割レイアウト機能、およびコントロールパネルのメニュー化を実装。 | <span style="color:orange;">未検証</span> | フェーズ1 |

## 3. PTZ制御機能 (PTZ Control)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PTZ操作** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | ONVIF認証情報を保存し、PTZコマンドを中継するAPIを実装。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **排他制御** | <span style="color:green;">**実装済み**</span> | `management-service` | Redisを利用したカメラPTZ操作のロック機構を実装。 | <span style="color:red;">テストなし</span> | フェーズ1 |

## 4. AI映像解析機能 (AI Analytics)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **高度なAI解析** | <span style="color:red;">**未実装**</span> | `ai-analysis-service` | 設計書にある侵入検知、顔認識、LPR等のAIモデルを用いた解析機能はない。 | <span style="color:red;">テストなし</span> | 追加機能 |
| **AIイベント通知** | <span style="color:green;">**実装済み**</span> | `ai-analysis-service`, `management-service` | **基本的なモーション検知**（フレーム差分）のみ実装。検知結果を `management-service` に通知し、さらにRedis経由で他サービスへ連携するアーキテクチャを確立。 | <span style="color:red;">テストなし</span> | 追加機能 |

## 5. 検索・再生・管理機能 (Search, Playback & Management)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **タイムライン検索** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | タイムライン上に録画セグメント（青）とAIイベント（赤）を両方表示し、直感的な検索が可能。 | <span style="color:orange;">作成済み・未実行</span> | 機能拡充 |
| **AI属性検索** | <span style="color:red;">**未実装**</span> | `management-service`, `web-frontend` | AIの解析結果（「人物」「車両」など）で検索する機能はない。 | <span style="color:red;">テストなし</span> | 機能拡充 |
| **映像エクスポート** | <span style="color:red;">**未実装**</span> | `web-frontend` | 指定範囲の録画をファイルとしてダウンロードする機能はない。 | <span style="color:red;">テストなし</span> | 機能拡充 |
| **マップ連携** | <span style="color:red;">**未実装**</span> | `web-frontend`, `management-service` | 地図上へのカメラ配置や、アイコンからの再生機能はない。 | <span style="color:red;">テストなし</span> | 追加機能 |
| **カメラ管理** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | ONVIF/手動でのカメラ登録・一覧・削除が可能な管理UIを実装。 | <span style="color:orange;">未検証</span> | フェーズ1 |
| **ONVIF自動検出** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | カメラのIPと認証情報を基に、RTSP URL等を自動取得するAPIとUIを実装。 | <span style="color:orange;">作成済み・未実行</span> | フェーズ2 |
| **レイアウト保存機能** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | ユーザーごとに現在の画面レイアウトを保存・読込する機能を実装。 | <span style="color:orange;">未検証</span> | 機能拡充 |
| **ユーザー管理** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | 管理者はユーザーの作成・削除・役割変更が可能。 | <span style="color:orange;">作成済み・未実行</span> | フェーズ1 |
| **アクセス制御(RBAC)** | <span style="color:green;">**実装済み**</span> | `management-service` | `admin`と`operator`ロールを定義し、APIエンドポイントを権限で保護。 | <span style="color:orange;">作成済み・未実行</span> | フェーズ1 |
| **監査ログ** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | 重要な操作（ログイン、設定変更等）の記録・追跡機能を実装。管理者向けUIで閲覧・検索可能。 | <span style="color:red;">テストなし</span> | フェーズ1 |

## 6. 死活監視と運用 (Monitoring & Operations)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **カメラ/アプリ監視** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend`, etc. | 全バックエンドサービスにPrometheus形式の`/metrics`エンドポイントを実装。Grafanaダッシュボードも構築済み。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **カメラ死活監視アラート** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | カメラのオフラインを検知し、UI右上の通知メニューにリアルタイムで表示する機能を実装。 | <span style="color:red;">テストなし</span> | フェーズ1 |
| **高度なアラート通知** | <span style="color:red;">**未実装**</span> | N/A | Alertmanagerによる管理者へのメール通知機能はない。 | <span style="color:red;">テストなし</span> | フェーズ1 |

## 7. ストリーミング入力 (Streaming Ingest)

| 機能名 | 実装ステータス | 担当サービス | 評価メモと課題 | テスト実装状況 | 設計書フェーズ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PC/携帯からのRTMPストリーミング** | <span style="color:green;">**実装済み**</span> | `nginx-rtmp`, `management-service`, `recording-service` | OBS等のソフトウェアからRTMPで映像をプッシュし、動的にカメラとして登録・録画する機能を実装。 | <span style="color:red;">テストなし</span> | 追加機能 |
| **RTMPストリームの緊急アラート表示** | <span style="color:green;">**実装済み**</span> | `management-service`, `web-frontend` | RTMPストリーム開始時、緊急通報とみなし、全クライアントのUI上にビデオ付きのアラートをポップアップ表示する。 | <span style="color:red;">テストなし</span> | 追加機能 |

## 8. 非機能要件と全体的な課題 (Non-Functional Requirements & Global Issues)

- **テストの状況:**
  - RBAC、ONVIF、カメラ管理APIを対象とした**APIテストスクリプト (`tests/test_api.py`) を作成済み。**
  - **[ブロック中]** Docker Hubのレート制限により、テスト環境を起動できず、テストは未実行。
- **セキュリティ:**
  - **[改善]** RBAC、サービス間認証（APIキー）を実装。JWT秘密鍵やAPIキーを`.env`ファイルに分離し、ハードコードを排除。
- **スケーラビリティ:**
  - 設計書のKubernetesクラスタ構成ではなく、単一ホストでのDocker Compose構成となっている。プロトタイプとしては妥当だが、スケールアウトはできない。
- **ロギング:**
  - **[改善]** 全てのバックエンドサービスにおいて構造化ロギングを導入。
- **国際化 (i18n):**
  - **[実装済み]** `i18next`を導入し、UIの主要な部分を日本語と英語に多言語化。
