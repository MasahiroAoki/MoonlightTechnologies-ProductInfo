# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# ARGUS-System

## 1. システム概要

### 1.1. プロジェクト名
ARGUS-System (マイクロサービスベース VMSプロトタイプ)

### 1.2. 目的
Docker Compose環境で動作する、スケーラブルなVMSのプロトタイプを構築する。リアルタイム表示、録画、PTZ制御、死活監視、AI解析の基本機能をマイクロサービスとして実装し、将来的なKubernetesへの移行と機能拡張の基盤を確立する。

### 1.3. 対象規模
プロトタイプ段階。Dockerホスト1台で数台〜十数台のカメラを想定。

### 1.4. 主な機能要件 (実装状況)

| 機能 | ステータス | 備考 |
| :--- | :--- | :--- |
| **録画** | | |
| 連続録画 | <span style="color:green;">実装済み</span> | 登録されたカメラの映像を常時録画。 |
| イベント録画 | <span style="color:green;">実装済み</span> | AIイベントやRTMPストリーム開始時に、指定時間だけ録画。 |
| ANR | <span style="color:red;">未実装</span> | |
| **リアルタイム表示** | | |
| 低遅延表示(WebRTC) | <span style="color:green;">実装済み</span> | Webブラウザ上で500ms未満の遅延で表示。 |
| ライブ/再生表示 | <span style="color:green;">実装済み</span> | ライブ映像と録画データの両方を再生可能。 |
| **ビュー管理** | | |
| 自由な画面レイアウト | <span style="color:green;">実装済み</span> | 1/4/9/16分割に対応。レイアウトの保存/読込も可能。 |
| シーケンス表示 | <span style="color:red;">未実装</span> | |
| ビデオウォール | <span style="color:red;">未実装</span> | |
| **PTZ制御** | | |
| PTZ操作 | <span style="color:green;">実装済み</span> | ONVIF対応カメラの基本的なPTZ操作（上下左右、ズーム）。 |
| 排他制御 | <span style="color:green;">実装済み</span> | Redisを利用し、複数ユーザーによる同時操作を防止。 |
| **死活監視** | | |
| アプリケーション監視 | <span style="color:green;">実装済み</span> | Prometheusによるメトリクス収集とGrafanaによる可視化。 |
| カメラ監視 | <span style="color:green;">実装済み</span> | 定期的な接続チェックにより、カメラのオンライン/オフライン状態を管理。 |
| アラート通知 | <span style="color:green;">実装済み</span> | カメラオフライン時やRTMP受信時にUIへリアルタイム通知。 |
| **AI映像解析** | | |
| モーション検知 | <span style="color:green;">実装済み</span> | フレーム差分による基本的な動体検知。イベント録画のトリガーとして利用。 |
| 高度なAI解析 | <span style="color:red;">未実装</span> | 侵入検知、顔認識などは未実装。 |
| **検索・再生** | | |
| タイムライン検索 | <span style="color:green;">実装済み</span> | タイムライン上に通常録画（青）とイベント（赤）を同時表示。 |
| AI属性検索 | <span style="color:red;">未実装</span> | |
| 映像エクスポート | <span style="color:red;">未実装</span> | |
| **システム管理** | | |
| カメラ管理 | <span style="color:green;">実装済み</span> | ONVIF/手動でのカメラ登録・設定変更・削除。RTMPストリームも動的に登録。 |
| ユーザー管理 | <span style="color:green;">実装済み</span> | |
| RBAC | <span style="color:green;">実装済み</span> | 管理者/オペレーターの役割に応じた権限制御。 |
| 監査ログ | <span style="color:green;">実装済み</span> | ログイン、設定変更等の重要操作を記録。管理者向けUIで閲覧可能。 |
| **UI** | | |
| 多言語対応 | <span style="color:green;">実装済み</span> | 日本語/英語の切り替えに対応。 |

## 2. アーキテクチャ概要

### 2.1. コンテナアーキテクチャ (Docker Compose)
本システムは、`docker-compose.yml` によって管理されるマイクロサービスアーキテクチャを採用している。

| サービス名 | ディレクトリ | 公開ポート | 役割 |
| :--- | :--- | :--- | :--- |
| **`web-frontend`** | `./web-frontend` | `3000:80` | **UI/UX**: ユーザーが操作するReactベースのフロントエンド。 |
| **`streaming-service`** | `./streaming-service` | `8080:8080` | **映像配信**: WebRTC(Pion)を使い、ライブ・録画映像をフロントエンドに低遅延で配信。 |
| **`management-service`** | `./management-service` | `8001:8000` | **頭脳/API**: FastAPI製。全サービスの設定管理、ユーザー認証、RBAC、カメラ情報、死活監視、WebSocket通知、イベント中継などを司る。 |
| **`recording-service`**| `./recording-service`| - | **録画処理**: Go製。連続録画と、Redis経由で通知されるAIイベントに基づくイベント録画を並行して実行。 |
| **`ai-analysis-service`**| `./ai-analysis-service`| - | **映像解析**: Python/OpenCV製。映像のモーションを検知し、`management-service`にイベントを通知。 |
| **`rtmp-server`** | `./nginx-rtmp` | `1935:1935` | **RTMP受付**: Nginx製。PC/携帯からのRTMPストリームを受信し、`recording-service`へ通知。 |
| **`db`** | - | `5433:5432` | **データベース**: PostgreSQL。各種設定情報、ユーザー、ロール、録画メタデータ、監査ログ、アラートなどを保存。 |
| **`redis`** | - | `6379:6379` | **キャッシュ/PubSub**: Redis。PTZ操作の排他制御に加え、サービス間のリアルタイムイベント通知（Pub/Sub）に使用。 |
| **`prometheus`** | - | `9090:9090` | **メトリクス収集**: 各サービスの`/metrics`エンドポイントから稼働情報を収集。 |
| **`grafana`** | - | `3001:3000` | **ダッシュボード**: Prometheusが収集したデータを可視化。 |

### 2.2. ストレージ構成
Dockerホストのローカルボリュームを使用。
- `postgres_data`: PostgreSQLのデータを永続化。
- `recording_data`: 録画されたMP4ファイルを保存。
- `prometheus_data`, `grafana_data`: 各監視ツールのデータを永続化。

### 2.3. ソフトウェアスタック
- **OS:** Dockerが動作するLinux環境（例: Ubuntu Server）
- **コンテナ技術:** Docker, Docker Compose
- **リアルタイム配信:** WebRTC (Pion)
- **イベント連携:** Redis Pub/Sub
- **開発言語/フレームワーク:**
  - バックエンド: Python (FastAPI), Go
  - フロントエンド: JavaScript (React)
  - 映像処理: FFmpeg

## 3. コンポーネント詳細

### 3.1. マネジメントサービス (Management Service)
- **役割:** システム全体の管理・制御を担うAPIサーバー。
- **機能:**
  - カメラ/ユーザー/ロール等のCRUD、認証(JWT)、RBAC。
  - **カメラ死活監視:** バックグラウンドタスクでカメラのオンライン/オフラインを定期的にチェック。
  - **イベントハブ:** AIイベントをRedis Pub/Subへ発行。
  - **リアルタイム通知:** WebSocketを介して、フロントエンドにシステムアラート（カメラオフライン等）や緊急通報（RTMP受信）をプッシュ通知。

### 3.2. レコーディングサービス (Recording Service)
- **役割:** ハイブリッド録画処理。
- **機能:**
  - **連続録画:** 登録済みカメラの映像を常時録画。
  - **イベント録画:** Redis Pub/Subを介してAIイベントを購読し、イベント発生時に指定時間だけ録画。
  - **RTMP録画:** `rtmp-server`からの通知を受け、RTMPストリームをイベントとして録画。

### 3.3. ストリーミングサービス (Streaming Service)
- **役割:** WebRTCによる映像配信。
- **機能:** フロントエンドからの要求に応じ、ライブ映像(RTSP/RTMP)または録画ファイル(MP4)をWebRTCストリームに変換して配信。

### 3.4. AI映像解析サービス (AI Analysis Service)
- **役割:** 映像の簡易解析。
- **機能:** フレーム差分によるモーション検知を行い、結果をイベントとして`management-service`に通知。

## 4. 実行方法

### 4.1. シークレット設定
プロジェクトのルートに `.env` ファイルを作成し、以下の内容を記述します。

```
# .env
SECRET_KEY=09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7
INTERNAL_API_KEY=d9e8f0a7-9b8c-4f0e-b8b1-7e8e0a9b0c0d
```
**注意:** この`.env`ファイルは`.gitignore`によりバージョン管理から除外されます。

### 4.2. 起動
以下のコマンドで全てのサービスを起動します。
```bash
sudo docker compose up --build -d
```

## 5. セキュリティ
- **認証:** JWTによるAPI認証。
- **認可:** RBACに基づき、エンドポイントごとに操作権限を制限。
- **改善点:**
  - **[済]** サービス間認証のために内部APIキーを導入。
  - **[済]** `docker-compose.yml`内のシークレットを`.env`ファイルに分離。
  - **[課題]** 各サービス間通信、フロントエンド-バックエンド間通信が暗号化されていない(HTTP)。
