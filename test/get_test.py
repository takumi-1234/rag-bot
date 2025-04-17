# get_test.py
import requests
import os
import logging
from tqdm import tqdm  # 進捗表示用

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- スクリプトのディレクトリを取得 ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 設定 ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{FASTAPI_URL}/api/chat"
# ファイルパスをスクリプトの場所からの相対パスで指定
QUESTION_FILE = os.path.join(script_dir, "question.txt")
ANSWER_FILE = os.path.join(script_dir, "answer.txt")
REQUEST_TIMEOUT = 180 # APIのタイムアウト（秒）
DEFAULT_K = 3       # 検索するチャンク数

# --- 関数 ---
def test_chat_endpoint():
    """question.txt から質問を読み込み、APIに送信して結果を answer.txt に保存する"""
    if not os.path.exists(QUESTION_FILE):
        logger.error(f"質問ファイルが見つかりません: {QUESTION_FILE}")
        return

    try:
        with open(QUESTION_FILE, 'r', encoding='utf-8') as qf:
            questions = [line.strip() for line in qf if line.strip()]
    except Exception as e:
        logger.error(f"{QUESTION_FILE} の読み込み中にエラーが発生しました: {e}")
        return

    if not questions:
        logger.warning(f"{QUESTION_FILE} に有効な質問が見つかりませんでした。")
        return

    logger.info(f"{len(questions)} 件の質問を {QUESTION_FILE} から読み込みました。")
    logger.info(f"回答を {ANSWER_FILE} に書き込みます...")

    # 回答ファイルを書き込みモード ('w') で開く (実行ごとに上書き)
    try:
        with open(ANSWER_FILE, 'w', encoding='utf-8') as af:
            # tqdm を使って進捗バーを表示
            for question in tqdm(questions, desc="質問処理中"):
                logger.debug(f"質問送信中: {question}")
                payload = {"query": question, "k": DEFAULT_K}
                try:
                    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status() # HTTPエラーがあれば例外発生

                    response_data = response.json()
                    answer = response_data.get("response", "エラー: 回答キー 'response' が見つかりません")
                    sources = response_data.get("sources", [])

                    # 回答をファイルに書き込み
                    af.write(f"Q: {question}\n")
                    af.write(f"A: {answer}\n")
                    if sources:
                        # ソースリストを整形して書き込み
                        unique_sorted_sources = sorted(list(set(sources)))
                        af.write(f"Sources: {', '.join(unique_sorted_sources)}\n")
                    af.write("-" * 20 + "\n") # 区切り線
                    logger.debug(f"回答受信: {answer[:50]}...")

                except requests.exceptions.Timeout:
                    error_msg = f"エラー: APIリクエストがタイムアウトしました (質問: {question})"
                    logger.error(error_msg)
                    af.write(f"Q: {question}\n")
                    af.write(f"A: {error_msg}\n")
                    af.write("-" * 20 + "\n")
                except requests.exceptions.RequestException as e:
                    status_code = e.response.status_code if e.response is not None else "N/A"
                    detail = "N/A"
                    if e.response is not None:
                        try:
                            detail = e.response.json().get('detail', e.response.text)
                        except ValueError: # JSONデコードエラーの場合
                            detail = e.response.text[:200] # レスポンスの一部
                    error_msg = f"エラー: APIリクエストに失敗しました (ステータス: {status_code}, 詳細: {detail}, 質問: {question})"
                    logger.error(error_msg)
                    af.write(f"Q: {question}\n")
                    af.write(f"A: {error_msg}\n")
                    af.write("-" * 20 + "\n")
                except Exception as e:
                    error_msg = f"エラー: 予期せぬエラーが発生しました (質問: {question}): {e}"
                    logger.exception(error_msg) # スタックトレースも出力
                    af.write(f"Q: {question}\n")
                    af.write(f"A: {error_msg}\n")
                    af.write("-" * 20 + "\n")

        logger.info(f"{ANSWER_FILE} への書き込みが完了しました。")

    except Exception as e:
        logger.error(f"{ANSWER_FILE} への書き込み中にエラーが発生しました: {e}")


# --- メイン処理 ---
if __name__ == "__main__":
    logger.info("GETテスト スクリプトを開始します...")
    test_chat_endpoint()
    logger.info("GETテスト スクリプトを終了します。")