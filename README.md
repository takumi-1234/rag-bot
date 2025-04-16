# 大学講義支援 RAG チャットボット

## 概要

このアプリケーションは、大学の講義資料（シラバス、レジュメ、教科書など）を知識源として活用するRAG（Retrieval-Augmented Generation）ベースのチャットボットです。学生は自然言語で講義内容やレポート課題に関する質問をするだけで、システムが関連資料を検索・参照し、それに基づいた信頼性の高い回答を提供します。

**目的:**

*   LLMのハルシネーションリスクを低減し、信頼性の高い情報を提供する。
*   講義固有の文脈（配布資料、教員の指示など）を理解した回答を生成する。
*   学生のレポート作成や日々の学習を効率的に支援する。

## 技術スタック

*   **バックエンド API:** FastAPI
*   **フロントエンド UI:** Streamlit
*   **RAG フレームワーク:** LangChain
*   **ベクトルデータベース:** ChromaDB (ローカルファイル永続化)
*   **言語モデル (LLM):** Google Gemini API
*   **Embedding モデル:** Sentence Transformers (all-mpnet-base-v2 デフォルト)
*   **コンテナ化:** Docker, Docker Compose
*   **プロセス管理:** Supervisor

## 機能

*   **資料アップロード:**
    *   Streamlit UIから講義資料（PDF, DOCX, TXT）をアップロード。
    *   cURLを使用してAPI経由 (`/api/upload`) で資料をアップロード。
*   **チャットインターフェース:**
    *   Streamlit UIで自然言語による質問応答。
*   **RAGパイプライン:**
    *   アップロードされた資料をチャンク分割し、ベクトル化してChromaDBに保存。
    *   ユーザーの質問に関連する文書チャンクをChromaDBから検索。
    *   検索結果をコンテキストとしてGemini APIに渡し、回答を生成。

## セットアップと起動

### 前提条件

*   Docker
*   Docker Compose
*   Google Gemini API キー

### 手順

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd university-rag-bot
    ```

2.  **環境変数の設定:**
    *   `.env.example` ファイルをコピーして `.env` ファイルを作成します。
    *   `.env` ファイルを開き、`GEMINI_API_KEY` にあなたのGoogle Gemini APIキーを設定します。
    *   必要に応じて他の変数 (Embeddingモデル名など) を変更します。
    ```bash
    cp .env.example .env
    nano .env # または任意のエディタで編集
    ```

3.  **データディレクトリの作成:**
    *   ChromaDBのデータを永続化するためのディレクトリを作成します（`docker-compose.yml` でマウントされます）。
    ```bash
    mkdir data
    mkdir static
    mkdir static/uploads # If using host mount for uploads (check docker-compose.yml)
    ```
    *   **(注意)** `data/` ディレクトリは `.gitignore` に追加することを強く推奨します。

4.  **Dockerイメージのビルド:**
    ```bash
    docker-compose build
    ```

5.  **Dockerコンテナの起動:**
    ```bash
    docker-compose up -d
    ```
    *   `-d` オプションでバックグラウンドで起動します。ログを確認する場合は `docker-compose logs -f` を実行します。

### アクセス

*   **Streamlit UI:** ブラウザで `http://localhost:8501` を開きます。
*   **FastAPI ドキュメント (Swagger UI):** ブラウザで `http://localhost:8000/docs` を開きます。
*   **FastAPI ヘルスチェック:** `http://localhost:8000/health`

## API エンドポイント

*   **`POST /api/upload`**: ファイルをアップロードしてベクトルDBに追加します。
    *   フォームデータ: `file` (アップロードするファイル)
    *   例 (cURL):
        ```bash
        curl -X POST -F "file=@path/to/your/lecture_notes.pdf" http://localhost:8000/api/upload
        ```
*   **`POST /api/chat`**: ユーザーの質問を受け取り、RAGを実行して回答を返します。
    *   クエリパラメータ: `query` (ユーザーの質問文字列)
    *   例 (cURL):
        ```bash
        curl -X POST "http://localhost:8000/api/chat?query=講義の主要なテーマは何ですか？"
        ```
*   **`GET /api/vectorstore/count`**: ベクトルストア内のアイテム数を返します。
*   **`GET /health`**: アプリケーションのヘルス状態を返します。

## 停止

```bash
docker-compose down
```

## エラーの発生時
```
# ログを確認
docker compose logs -f
```