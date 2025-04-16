# src/streamlit_app.py
import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- 環境変数の読み込み ---
# Docker環境外でのローカル実行時などに .env ファイルを読み込む
load_dotenv()

# --- ロギング設定 ---
# フォーマットを改善
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI エンドポイント設定 ---
# 環境変数から FastAPI のベース URL を取得、なければデフォルト値を使用
API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/upload"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/chat"
COUNT_ENDPOINT = f"{API_BASE_URL}/api/vectorstore/count"
DELETE_ENDPOINT = f"{API_BASE_URL}/api/vectorstore/delete_all"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health" # ヘルスチェック用追加

# --- Streamlit アプリケーション ---

st.set_page_config(page_title="講義支援 RAG チャットボット", layout="wide")
st.title("🎓 講義支援 RAG チャットボット")
st.caption("アップロードされた講義資料に基づいて質問に回答します。")

# --- セッション状態の初期化 ---
# チャット履歴
if "messages" not in st.session_state:
    st.session_state.messages = []
# アップロード状態
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []
# ベクトルDBのドキュメント数
if "doc_count" not in st.session_state:
    st.session_state.doc_count = -1 # 未取得状態

# --- ヘルスチェック関数 ---
def check_api_health() -> bool:
    """FastAPIサーバーのヘルスチェックを行う"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and data.get("initialized"):
                logger.info("API health check: OK")
                return True
            else:
                logger.warning(f"API health check: Service might not be fully initialized. Response: {data}")
                st.sidebar.warning("⚠️ APIは起動していますが、初期化が未完了の可能性があります。", icon="🚨")
                return False
        else:
            logger.error(f"API health check failed with status code: {response.status_code}")
            st.sidebar.error(f"❌ APIサーバー({API_BASE_URL})に接続できません (ステータス: {response.status_code})", icon="🚨")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check connection error: {e}")
        st.sidebar.error(f"❌ APIサーバー({API_BASE_URL})に接続できません。サーバーが起動しているか確認してください。", icon="🚨")
        return False

# --- サイドバー ---
with st.sidebar:
    st.header("設定 & 操作")

    # APIヘルスチェック
    st.divider()
    st.subheader("APIステータス")
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API接続済み", icon="🔗")
    st.button("再確認", key="health_check_button", on_click=check_api_health, disabled=not api_healthy and st.session_state.get("health_check_button", False)) # 連続クリック防止

    # ファイルアップロードセクション
    st.divider()
    st.subheader("資料アップロード")
    uploaded_files = st.file_uploader(
        "講義資料 (PDF, DOCX, TXT) を選択",
        accept_multiple_files=True, # 複数ファイル対応
        type=["pdf", "docx", "txt"],
        disabled=not api_healthy # APIが unhealthy なら無効化
    )

    if st.button("選択したファイルを処理", disabled=not uploaded_files or not api_healthy):
        st.session_state.upload_processing = True # 処理中フラグ
        files_to_process = uploaded_files
        total_files = len(files_to_process)
        processed_files = 0
        failed_files = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(files_to_process):
            filename = uploaded_file.name
            status_text.info(f"処理中 ({i+1}/{total_files}): {filename}")
            try:
                files = {"file": (filename, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=180)
                response.raise_for_status() # エラーチェック

                result = response.json()
                if result.get("status") == "success":
                    processed_files += 1
                    # 成功したファイル情報を追加 (必要に応じて)
                    st.session_state.uploaded_files_info.append({
                        "name": filename,
                        "chunks": result.get("chunks_added", 0)
                    })
                    logger.info(f"File '{filename}' uploaded successfully. Chunks added: {result.get('chunks_added')}")
                else:
                    failed_files.append(f"{filename} ({result.get('message', '不明なエラー')})")
                    logger.error(f"Failed to process file '{filename}': {result.get('message')}")

            except requests.exceptions.RequestException as e:
                failed_files.append(f"{filename} (ネットワークエラー: {e})")
                logger.error(f"Network error uploading file '{filename}': {e}", exc_info=True)
                st.error(f"{filename} のアップロード中にネットワークエラーが発生しました。")
            except Exception as e:
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                     try:
                          error_detail = e.response.json().get('detail', str(e))
                     except requests.exceptions.JSONDecodeError:
                          error_detail = e.response.text[:200] # レスポンス本文の一部
                failed_files.append(f"{filename} (サーバーエラー: {error_detail})")
                logger.error(f"Server error uploading file '{filename}': {e}", exc_info=True)
                st.error(f"{filename} のアップロード中にサーバーエラーが発生しました。")

            progress_bar.progress((i + 1) / total_files)

        status_text.empty() # "処理中..." メッセージを消去
        if processed_files > 0:
             st.success(f"{processed_files} / {total_files} 件のファイルの処理が完了しました。")
        if failed_files:
             st.error("以下のファイルの処理に失敗しました:")
             for failed in failed_files:
                 st.markdown(f"- {failed}")
        st.session_state.upload_processing = False # 処理中フラグ解除
        st.rerun() # 画面を再描画して結果を反映

    # ベクトルストア管理セクション
    st.divider()
    st.subheader("ベクトルストア管理")

    # ドキュメント数を表示
    if st.button("DBドキュメント数 更新", disabled=not api_healthy):
        try:
            response = requests.get(COUNT_ENDPOINT, timeout=10)
            response.raise_for_status()
            count_data = response.json()
            st.session_state.doc_count = count_data.get("count", -1)
        except requests.exceptions.RequestException as e:
            st.error(f"ドキュメント数取得中にエラー: {e}")
            st.session_state.doc_count = -1
        except Exception as e:
            st.error(f"ドキュメント数取得中に予期せぬエラー: {e}")
            st.session_state.doc_count = -1

    if st.session_state.doc_count != -1:
        st.metric("DB内のドキュメント数:", st.session_state.doc_count)
    else:
        st.info("ドキュメント数を取得してください。")

    # 全削除ボタン (確認付き)
    st.divider()
    if st.button("DB全ドキュメント削除", type="secondary", disabled=not api_healthy):
        st.session_state.show_delete_confirmation = True

    if st.session_state.get("show_delete_confirmation", False):
        st.warning("⚠️ 本当にベクトルストア内の全ドキュメントを削除しますか？この操作は元に戻せません。", icon="🔥")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("はい、削除します", type="primary"):
                try:
                    with st.spinner("削除処理を実行中..."):
                        response = requests.delete(DELETE_ENDPOINT, timeout=60)
                        response.raise_for_status()
                        delete_result = response.json()
                        st.success(f"削除完了: {delete_result.get('message', '成功')}")
                        st.session_state.doc_count = 0 # カウントをリセット
                        st.session_state.uploaded_files_info = [] # アップロード情報もクリア
                except requests.exceptions.RequestException as e:
                    st.error(f"削除中にネットワークエラー: {e}")
                except Exception as e:
                    error_detail = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json().get('detail', str(e))
                        except requests.exceptions.JSONDecodeError:
                            error_detail = e.response.text[:200]
                    st.error(f"削除中にサーバーエラー: {error_detail}")
                finally:
                    st.session_state.show_delete_confirmation = False
                    st.rerun()
        with col2:
            if st.button("キャンセル"):
                st.session_state.show_delete_confirmation = False
                st.rerun()

# --- メインコンテンツエリア ---
# チャットインターフェース
st.header("💬 チャット")

# チャット履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # アシスタントのメッセージでソースがあれば表示
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
             # 重複を除外しソートして表示
             unique_sorted_sources = sorted(list(set(message["sources"])))
             with st.expander("参照ソース"):
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
        sources: List[str] = [] # スコープを広げる

        with st.spinner("回答を生成中です..."):
            try:
                # FastAPIエンドポイントにPOSTリクエスト (jsonパラメータを使用)
                payload = {"query": prompt, "k": 3} # kの値は調整可能
                logger.info(f"Calling chat API: {CHAT_ENDPOINT} with payload: {payload}")
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)

                logger.info(f"Chat API response status code: {response.status_code}")
                response.raise_for_status() # エラーチェック

                result = response.json()
                full_response_content = result.get("response", "エラー: 回答を取得できませんでした。")
                sources = result.get("sources", [])
                if sources:
                    # 重複を除外しソート
                    unique_sorted_sources = sorted(list(set(sources)))
                    sources_content = "\n\n---\n**参照ソース:**\n" + "\n".join([f"- {s}" for s in unique_sorted_sources])

                # 回答とソースを表示 (ストリーミング風に更新も可能だが、ここでは一括表示)
                message_placeholder.markdown(full_response_content + sources_content)

            except requests.exceptions.RequestException as e:
                error_message = f"API呼び出し中にネットワークエラーが発生しました: {e}"
                st.error(error_message)
                full_response_content = error_message
                logger.error(f"Error calling chat endpoint: {e}", exc_info=True)
            except Exception as e:
                error_message = f"回答の生成中にエラーが発生しました: {e}"
                error_details_for_display = ""
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    try:
                        error_details = e.response.json()
                        if isinstance(error_details, dict) and 'detail' in error_details:
                            if isinstance(error_details['detail'], list) and error_details['detail']:
                                 # Pydanticのエラーの場合
                                 loc = " -> ".join(map(str, error_details['detail'][0].get('loc', [])))
                                 msg = error_details['detail'][0].get('msg', '')
                                 error_details_for_display = f"(ステータス: {status_code}, 詳細: {msg} at {loc})"
                            else:
                                 error_details_for_display = f"(ステータス: {status_code}, 詳細: {error_details['detail']})"
                        else:
                             error_details_for_display = f"(ステータス: {status_code}, 詳細: {str(error_details)[:200]})"
                    except requests.exceptions.JSONDecodeError:
                        error_details = e.response.text
                        error_details_for_display = f"(ステータス: {status_code}, 詳細: {error_details[:200]})"
                    logger.error(f"Error calling chat endpoint (Status: {status_code}, Detail: {error_details})", exc_info=True)
                else:
                    logger.error(f"Error generating response: {e}", exc_info=True)

                error_message_display = f"回答の生成中にエラーが発生しました {error_details_for_display}"
                st.error(error_message_display)
                full_response_content = error_message_display # エラーを履歴に残す

        # アシスタントメッセージを履歴に追加
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content,
            "sources": sources # 取得したソース情報を履歴に保存
        })
        # 新しいメッセージが追加されたら再実行して表示を更新
        st.rerun()

# フッター等 (オプション)
st.divider()
st.markdown("---")
st.caption("Powered by LangChain, Gemini, ChromaDB, FastAPI, and Streamlit.")