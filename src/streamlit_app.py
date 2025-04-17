# src/streamlit_app.py
import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional # Optional を追加

# --- 環境変数の読み込み ---
# .env ファイルは通常 Docker 環境では docker-compose.yml で読み込まれる
# ローカル実行用に load_dotenv を残しておく
# srcディレクトリの親にある .env を想定
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Streamlit: Loaded environment variables from: {dotenv_path}")
else:
    print(f"Streamlit: .env file not found at {dotenv_path}, relying on system environment variables.")

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- FastAPI エンドポイント設定 ---
API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/upload"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/chat"
COUNT_ENDPOINT = f"{API_BASE_URL}/api/vectorstore/count"
DELETE_ENDPOINT = f"{API_BASE_URL}/api/vectorstore/delete_all"


# --- Helper Functions ---

def get_api_error_message(e: requests.exceptions.RequestException) -> str:
    """Requests 例外から詳細なエラーメッセージを抽出する"""
    if e.response is not None:
        status_code = e.response.status_code
        try:
            error_data = e.response.json()
            # FastAPI の HTTPException detail を取得
            detail = error_data.get("detail", e.response.text)
            # Pydantic のエラーの場合、整形する (オプション)
            if isinstance(detail, list) and detail and isinstance(detail[0], dict) and 'loc' in detail[0] and 'msg' in detail[0]:
                loc = " -> ".join(map(str, detail[0].get('loc', [])))
                msg = detail[0].get('msg', '')
                return f"API Error (Status: {status_code}): {msg} at '{loc}'"
            # 通常の detail 文字列
            return f"API Error (Status: {status_code}): {detail}"
        except requests.exceptions.JSONDecodeError:
            # JSONデコード失敗時
            return f"API Error (Status: {status_code}): {e.response.text[:200]}" # レスポンス本文の一部
    elif isinstance(e, requests.exceptions.ConnectionError):
        return f"Connection Error: Failed to connect to API at {API_BASE_URL}. Is the server running?"
    elif isinstance(e, requests.exceptions.Timeout):
        return "Connection Timeout: The request to the API timed out."
    else:
        # その他の RequestException
        return f"Network Error: An unexpected network error occurred: {e}"

@st.cache_data(ttl=60) # 60秒キャッシュ
def fetch_vector_store_count() -> int:
    """ベクトルストアのドキュメント数を取得する"""
    logger.info(f"Fetching vector store count from {COUNT_ENDPOINT}")
    try:
        response = requests.get(COUNT_ENDPOINT, timeout=10)
        response.raise_for_status()
        count_data = response.json()
        count = count_data.get("count", -1)
        logger.info(f"Vector store count received: {count}")
        return count
    except requests.exceptions.RequestException as e:
        error_msg = get_api_error_message(e)
        st.error(f"Failed to fetch document count: {error_msg}", icon="🚨")
        logger.error(f"Error fetching document count: {error_msg}", exc_info=False if e.response else True)
        return -1 # エラー時は -1
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching document count: {e}", icon="🔥")
        logger.error(f"Unexpected error fetching document count: {e}", exc_info=True)
        return -1

@st.cache_data(ttl=15) # 15秒キャッシュ
def check_api_status() -> Dict[str, Any]:
    """APIのヘルスチェックを行い、状態を返す"""
    logger.info(f"Checking API health at {HEALTH_ENDPOINT}")
    status_info = {"healthy": False, "message": "Checking...", "details": {}}
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and data.get("initialized"):
                status_info["healthy"] = True
                status_info["message"] = "API Ready"
                status_info["details"] = data
                logger.info("API health check: OK")
            else:
                status_info["message"] = "API Running but not fully initialized"
                status_info["details"] = data
                logger.warning(f"API health check: Not fully initialized. Response: {data}")
        else:
            status_info["message"] = f"API Error (Status: {response.status_code})"
            try:
                status_info["details"] = response.json()
            except requests.exceptions.JSONDecodeError:
                status_info["details"] = {"error": response.text[:100]}
            logger.error(f"API health check failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        error_msg = get_api_error_message(e)
        status_info["message"] = f"API Connection Error"
        status_info["details"] = {"error": error_msg}
        logger.error(f"API health check connection error: {error_msg}", exc_info=False if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) else True)

    return status_info

# --- Streamlit アプリケーション UI ---

st.set_page_config(page_title="講義支援 RAG チャットボット", layout="wide")
st.title("🎓 講義支援 RAG チャットボット")
st.caption("アップロードされた講義資料に基づいて質問に回答します。")

# --- セッション状態の初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# if "uploaded_files_info" not in st.session_state: # アップロード成功ファイルの情報は保持しない方針に
#     st.session_state.uploaded_files_info = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = None # None で未取得状態を示す
if "api_status" not in st.session_state:
    st.session_state.api_status = {"healthy": False, "message": "Initializing..."}


# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 設定 & 操作")

    # APIステータス表示
    st.divider()
    st.subheader("API ステータス")
    if st.button("🔄 再確認"):
        st.cache_data.clear() # キャッシュをクリアして再取得
        st.session_state.api_status = check_api_status()
    else:
        # 定期的に確認（初回またはボタンが押されなかった場合）
        if st.session_state.api_status.get("message") == "Initializing...":
             st.session_state.api_status = check_api_status()

    # ステータスメッセージとアイコン表示
    api_healthy = st.session_state.api_status.get("healthy", False)
    api_message = st.session_state.api_status.get("message", "Unknown")
    if api_healthy:
        st.success(f"✅ {api_message}", icon="🔗")
        # 詳細情報（ベクトルDB件数など）を表示
        details = st.session_state.api_status.get("details", {})
        if "vector_store_count" in details:
             st.info(f"Vector DB Count: {details['vector_store_count']}", icon="📊")
    elif "Connection Error" in api_message:
         st.error(f"❌ {api_message}", icon="🚨")
    else:
         st.warning(f"⚠️ {api_message}", icon="⏳")
         details = st.session_state.api_status.get("details", {})
         if details.get("error"):
             st.caption(f"Detail: {details['error']}")


    # ファイルアップロードセクション
    st.divider()
    st.subheader("📚 資料アップロード")
    uploaded_files = st.file_uploader(
        "講義資料 (PDF, DOCX, TXT) を選択",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        disabled=not api_healthy, # APIが unhealthy なら無効化
        key="file_uploader"
    )

    if st.button("📤 選択したファイルを処理", disabled=not uploaded_files or not api_healthy):
        files_to_process = uploaded_files
        total_files = len(files_to_process)
        processed_count = 0
        failed_files_details: List[Dict[str, str]] = []

        progress_bar = st.progress(0, text=f"開始しています...")
        status_placeholder = st.empty() # 個々のファイルステータス用

        for i, uploaded_file in enumerate(files_to_process):
            filename = uploaded_file.name
            progress_text = f"処理中 ({i+1}/{total_files}): {filename}"
            progress_bar.progress((i + 1) / total_files, text=progress_text)
            status_placeholder.info(f"⏳ {filename} をアップロード中...")

            try:
                # ファイルデータとファイル名をAPIに送信
                files = {"file": (filename, uploaded_file.getvalue(), uploaded_file.type)}
                # タイムアウトを長めに設定 (大きなファイルの処理を考慮)
                response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=300)
                response.raise_for_status() # ステータスコード 2xx 以外で例外発生

                result = response.json() # 成功時 (201 Created)
                processed_count += 1
                chunks = result.get("chunks_added", "N/A")
                status_placeholder.success(f"✅ {filename} 処理完了 (Chunks: {chunks})")
                logger.info(f"File '{filename}' uploaded successfully. Chunks added: {chunks}")
                # アップロード成功後、ファイルアップローダーをクリアする (任意)
                # st.session_state.file_uploader = [] # これを行うとUIがリセットされる

            except requests.exceptions.RequestException as e:
                error_msg = get_api_error_message(e)
                failed_files_details.append({"name": filename, "error": error_msg})
                status_placeholder.error(f"❌ {filename}: {error_msg}")
                logger.error(f"Error uploading file '{filename}': {error_msg}", exc_info=False if e.response else True)
            except Exception as e:
                # API呼び出し以外の予期せぬエラー
                error_msg = f"An unexpected error occurred: {e}"
                failed_files_details.append({"name": filename, "error": error_msg})
                status_placeholder.error(f"🔥 {filename}: {error_msg}")
                logger.error(f"Unexpected error processing file '{filename}' locally: {e}", exc_info=True)

        # 最終結果表示
        progress_bar.empty() # プログレスバーを削除
        status_placeholder.empty() # 個々のファイルステータスを削除
        st.success(f"ファイル処理完了: {processed_count} 件成功 / {total_files} 件中")
        if failed_files_details:
             with st.expander("⚠️ 失敗したファイルの詳細", expanded=True):
                 for item in failed_files_details:
                     st.error(f"**{item['name']}**: {item['error']}")
        # DBカウントを更新
        st.session_state.doc_count = None # 再取得を促す


    # ベクトルストア管理セクション
    st.divider()
    st.subheader("📦 ベクトルストア管理")

    # ドキュメント数を表示 (キャッシュされた関数を使用)
    if st.button("🔄 DBドキュメント数 更新", disabled=not api_healthy):
         st.cache_data.clear() # fetch_vector_store_count のキャッシュをクリア
         st.session_state.doc_count = None # リセットして再取得を促す

    # doc_count が None の場合は取得を試みる
    if st.session_state.doc_count is None and api_healthy:
        st.session_state.doc_count = fetch_vector_store_count()

    if st.session_state.doc_count is not None and st.session_state.doc_count >= 0:
        st.metric("DB内のドキュメント数:", st.session_state.doc_count)
    elif api_healthy:
        st.info("ドキュメント数を取得できませんでした。")
    else:
         st.info("APIに接続してドキュメント数を取得してください。")


    # 全削除ボタン (確認付き)
    st.divider()
    if st.button("🗑️ DB全ドキュメント削除", type="secondary", disabled=not api_healthy or st.session_state.doc_count == 0):
        st.session_state.show_delete_confirmation = True

    if st.session_state.get("show_delete_confirmation", False):
        st.warning("🔥 本当にベクトルストア内の全ドキュメントを削除しますか？ この操作は元に戻せません。", icon="⚠️")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ はい、削除します", type="primary"):
                try:
                    with st.spinner("削除処理を実行中..."):
                        # タイムアウトを少し長めに
                        response = requests.delete(DELETE_ENDPOINT, timeout=60)
                        response.raise_for_status() # エラーチェック
                        delete_result = response.json()
                        st.success(f"削除完了: {delete_result.get('message', '成功しました')}")
                        st.session_state.doc_count = 0 # カウントをリセット
                        st.session_state.api_status = check_api_status() # APIステータスも更新
                except requests.exceptions.RequestException as e:
                    error_msg = get_api_error_message(e)
                    st.error(f"削除中にエラーが発生しました: {error_msg}")
                    logger.error(f"Error deleting collection: {error_msg}", exc_info=False if e.response else True)
                except Exception as e:
                    st.error(f"削除中に予期せぬエラーが発生しました: {e}")
                    logger.error(f"Unexpected error deleting collection: {e}", exc_info=True)
                finally:
                    # 確認ダイアログを閉じる
                    st.session_state.show_delete_confirmation = False
                    # 画面を再描画して結果を反映
                    st.rerun()
        with col2:
            if st.button("❌ キャンセル"):
                st.session_state.show_delete_confirmation = False
                st.rerun()


# --- メインコンテンツエリア: チャット ---
st.header("💬 チャット")

# チャット履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # アシスタントのメッセージでソースがあれば表示
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
             unique_sorted_sources = sorted(list(set(message["sources"])))
             with st.expander("📚 参照ソース"):
                 for source in unique_sorted_sources:
                      st.markdown(f"- {source}")

# ユーザー入力
if prompt := st.chat_input("質問を入力してください...", disabled=not api_healthy):
    # ユーザーメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ボットの応答処理
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 回答表示用のプレースホルダ
        full_response_content = ""
        sources_content = ""
        sources: List[str] = []

        with st.spinner("🤖 回答を生成中です..."):
            try:
                # FastAPIエンドポイントにPOSTリクエスト
                payload = {"query": prompt, "k": 3} # kの値は調整可能
                logger.info(f"Calling chat API: {CHAT_ENDPOINT} with payload: {payload}")
                # タイムアウトを長めに設定
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)
                response.raise_for_status() # エラーチェック

                result = response.json()
                full_response_content = result.get("response", "エラー: 回答を取得できませんでした。")
                sources = result.get("sources", [])
                if sources:
                    unique_sorted_sources = sorted(list(set(sources)))
                    # ソースが多い場合は省略表示なども検討
                    sources_content = "\n\n---\n**📚 参照ソース:**\n" + "\n".join([f"- {s}" for s in unique_sorted_sources])

                # 回答とソースを表示
                message_placeholder.markdown(full_response_content + sources_content)

            except requests.exceptions.RequestException as e:
                error_message = get_api_error_message(e)
                st.error(error_message, icon="🔥") # エラーをUIに表示
                full_response_content = f"エラーが発生しました: {error_message}" # エラー情報を履歴に残す
                logger.error(f"Error calling chat endpoint: {error_message}", exc_info=False if e.response else True)
            except Exception as e:
                error_message = f"回答の生成中に予期せぬエラーが発生しました: {e}"
                st.error(error_message, icon="🔥")
                full_response_content = error_message
                logger.error(f"Unexpected error during chat response generation: {e}", exc_info=True)

        # アシスタントメッセージを履歴に追加
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content, # エラーメッセージも含む
            "sources": sources # エラー時でも sources は空リスト
        })
        # 新しいメッセージが表示されるように再実行
        st.rerun()

# --- フッター ---
st.divider()
st.caption("Powered by LangChain, Gemini, ChromaDB, FastAPI, and Streamlit.")