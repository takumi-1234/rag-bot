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
*   **RAG フレームワーク/ライブラリ:** LangChain, Sentence Transformers
*   **ベクトルデータベース:** ChromaDB (ローカルファイル永続化)
*   **言語モデル (LLM):** Google Gemini API (例: gemini-1.5-flash)
*   **Embedding モデル:** `intfloat/multilingual-e5-large` (または `.env` で指定されたモデル)
*   **コンテナ化:** Docker, Docker Compose
*   **プロセス管理:** Supervisor
*   **ファイル処理:** PyPDFLoader, Docx2txtLoader, TextLoader (LangChain Community)

## 機能

*   **資料アップロード:**
    *   Streamlit UIから講義資料（PDF, DOCX, TXT）を複数同時にアップロード。
    *   アップロード状況と結果（成功/失敗）をリアルタイムに表示。
*   **チャットインターフェース:**
    *   Streamlit UIで自然言語による質問応答。
    *   回答の根拠となった資料ソースを表示。
*   **RAGパイプライン:**
    *   アップロードされた資料をチャンク分割し、ベクトル化してChromaDBに保存。
    *   ユーザーの質問に関連する文書チャンクをChromaDBから検索（コサイン類似度）。
    *   検索結果をコンテキストとしてGemini APIに渡し、回答を生成。
*   **ベクトルストア管理:**
    *   Streamlit UIから現在のドキュメント（チャンク）数を表示。
    *   Streamlit UIからベクトルストア内の全ドキュメントを削除（確認付き）。
*   **API:**
    *   FastAPIによるRESTful APIを提供（資料アップロード、チャット、DB管理）。
    *   Swagger UI (`/docs`) によるAPIドキュメント。
    *   ヘルスチェックエンドポイント (`/health`)。

## セットアップと起動

### 前提条件

*   Docker
*   Docker Compose v2.x
*   Google Gemini API キー (Google AI Studio で取得)

### 手順

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd university-rag-bot
    ```

2.  **環境変数の設定:**
    *   `.env.example` ファイルをコピーして `.env` ファイルを作成します。
    ```bash
    cp .env.example .env
    ```
    *   `.env` ファイルを開き、`GEMINI_API_KEY` にあなたのGoogle Gemini APIキーを設定します。
    *   必要に応じて `EMBEDDING_MODEL_NAME` や `GEMINI_MODEL_NAME` を変更します。（デフォルトは `intfloat/multilingual-e5-large` と `gemini-1.5-flash`）
    ```bash
    nano .env # または任意のエディタで編集
    ```

3.  **データディレクトリの作成:**
    *   ChromaDBのデータを永続化するためのディレクトリを作成します（`docker-compose.yml` でマウントされます）。初回起動時に自動生成されることもありますが、明示的に作成しておくと権限問題を防ぎやすい場合があります。
    ```bash
    mkdir -p data/chroma
    ```
    *   **(注意)** `data/` ディレクトリは `.gitignore` に追加することを強く推奨します。

4.  **Dockerイメージのビルド:**
    *   プロジェクトのルートディレクトリで実行します。
    ```bash
    docker compose build
    ```
    *   ビルド中に Embedding モデルなどがダウンロードされるため、時間がかかることがあります。

5.  **Dockerコンテナの起動:**
    ```bash
    docker compose up -d
    ```
    *   `-d` オプションでバックグラウンドで起動します。
    *   初回起動時はモデルのダウンロードや初期化に時間がかかる場合があります。
6.  **ログを確認する場合**: 
    ```bash
    docker compose logs -f
    ```

### アクセス

*   **Streamlit UI:** ブラウザで `http://localhost:8501` を開きます。
*   **FastAPI ドキュメント (Swagger UI):** ブラウザで `http://localhost:8000/docs` を開きます。
*   **FastAPI ヘルスチェック:** ブラウザまたは curl で `http://localhost:8000/health` にアクセスします。

## API エンドポイント

*   **`GET /health`**: アプリケーションのヘルス状態を確認。
*   **`POST /api/upload`**: ファイルをアップロードしてベクトルDBに追加。
    *   フォームデータ: `file` (アップロードするファイル)
    *   成功時: `201 Created`
    *   例 (cURL):
        ```bash
        curl -X POST -F "file=@path/to/your/notes.pdf" http://localhost:8000/api/upload
        ```
*   **`POST /api/chat`**: ユーザーの質問を受け取り、RAGを実行して回答を返す。
    *   リクエストボディ (JSON): `{"query": "質問内容", "k": 3}` (kは検索するチャンク数)
    *   成功時: `200 OK`
    *   例 (cURL):
        ```bash
        curl -X POST -H "Content-Type: application/json" \
        -d '{"query": "講義の主要なテーマは何ですか？", "k": 5}' \
        http://localhost:8000/api/chat
        ```
*   **`GET /api/vectorstore/count`**: ベクトルストア内のアイテム数を取得。
*   **`DELETE /api/vectorstore/delete_all`**: ベクトルストアのコレクション（全データ）を削除。

## テスト

### 自動テストスクリプト

APIエンドポイントの基本的な動作を確認するための自動テストスクリプトが含まれています。

**必要なファイル:**

*   `get_test.py`: `/api/chat` エンドポイント (質問応答) をテストします。
*   `post_test.py`: `/api/upload` エンドポイント (ファイルアップロード) をテストします。
*   `question.txt`: `get_test.py` が使用する質問リスト (1行に1つの質問、英語)。
*   `teacher/` ディレクトリ: `post_test.py` がアップロードする学習資料 (PDF, DOCX, TXT) を格納します。

**準備:**

1.  `teacher/` ディレクトリにテスト用の講義資料ファイル (PDF, DOCX, TXT形式) をいくつか配置します。
2.  `question.txt` にテストしたい質問を英語で記述します (1行に1つの質問)。
3.  テストスクリプトは `requests` ライブラリと `tqdm` ライブラリを使用します。必要に応じてインストールしてください:
    ```bash
    pip install requests tqdm
    ```
4.  Dockerコンテナが起動していることを確認します (`docker compose ps`)。

**実行方法:**

1.  **ファイルアップロードテスト (POST):**
    *   `teacher/` ディレクトリ内のファイルを `/api/upload` に送信します。
    ```bash
    python post_test.py
    ```
    *   実行後、ターミナルに成功/失敗のログが表示されます。

2.  **質問応答テスト (GET):**
    *   `question.txt` の各質問を `/api/chat` に送信し、結果を `answer.txt` に保存します。
    ```bash
    python get_test.py
    ```
    *   実行後、`answer.txt` に質問とそれに対するボットの回答、参照ソースが書き込まれます。

**注意:**

*   これらのスクリプトは、FastAPIサーバーが `http://localhost:8000` で動作していることを前提としています。異なるURLで実行している場合は、各 `.py` ファイル内の `FASTAPI_URL` 変数を変更してください。
*   `answer.txt` は `get_test.py` を実行するたびに上書きされます。


## 停止

```bash
docker compose down
