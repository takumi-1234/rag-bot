# post_test.py
import requests
import os
import logging
from tqdm import tqdm # 進捗表示用

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
UPLOAD_ENDPOINT = f"{FASTAPI_URL}/api/upload"
# teacher ディレクトリのパスをスクリプトの場所からの相対パスで指定
TEACHER_DIR = os.path.join(script_dir, "teacher")
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")
REQUEST_TIMEOUT = 300 # アップロードは時間がかかる可能性があるので長めに設定

# --- 関数 ---
def test_upload_endpoint():
    """teacher ディレクトリ内のファイルを API にアップロードする"""
    if not os.path.isdir(TEACHER_DIR):
        logger.error(f"学習資料ディレクトリが見つかりません: {TEACHER_DIR}")
        return

    files_to_upload = []
    try:
        # teacher ディレクトリ直下のファイルのみを対象とする
        for filename in os.listdir(TEACHER_DIR):
            filepath = os.path.join(TEACHER_DIR, filename)
            # ファイルであり、かつサポートされている拡張子を持つか確認
            if os.path.isfile(filepath) and filename.lower().endswith(SUPPORTED_EXTENSIONS):
                files_to_upload.append(filepath)
    except Exception as e:
        logger.error(f"{TEACHER_DIR} のファイルリスト取得中にエラーが発生しました: {e}")
        return

    if not files_to_upload:
        logger.warning(f"{TEACHER_DIR} 内にサポートされている形式のファイルが見つかりませんでした。")
        return

    logger.info(f"{len(files_to_upload)} 件のファイルを {TEACHER_DIR} から検出しました。")
    logger.info("ファイルのアップロードを開始します...")

    success_count = 0
    failure_count = 0

    # tqdm を使って進捗バーを表示
    for filepath in tqdm(files_to_upload, desc="ファイルアップロード中"):
        filename = os.path.basename(filepath)
        logger.debug(f"アップロード試行中: {filename}")
        try:
            # ファイルをバイナリ読み込みモードで開く
            with open(filepath, 'rb') as f:
                # requests に渡すファイル形式をタプルで指定: (ファイル名, ファイルオブジェクト)
                files = {'file': (filename, f)}
                response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=REQUEST_TIMEOUT)
                response.raise_for_status() # HTTPステータスコードが 4xx または 5xx の場合に例外を発生

                result = response.json()
                if result.get("status") == "success":
                    chunks = result.get("chunks_added", "N/A") # chunks_added が存在しない場合に備える
                    logger.info(f"成功: {filename} (追加されたチャンク数: {chunks})")
                    success_count += 1
                else:
                    message = result.get("message", "不明なエラー")
                    logger.error(f"失敗: {filename} - APIからのエラーメッセージ: {message}")
                    failure_count += 1

        except FileNotFoundError:
            logger.error(f"失敗: ローカルファイルが見つかりません - {filepath}")
            failure_count += 1
        except requests.exceptions.Timeout:
            logger.error(f"失敗: APIリクエストがタイムアウトしました ({REQUEST_TIMEOUT}秒超過) - {filename}")
            failure_count += 1
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            detail = "N/A"
            if e.response is not None:
                try:
                    # FastAPIからの詳細なエラーメッセージを取得試行
                    detail = e.response.json().get('detail', e.response.text)
                except ValueError: # JSONデコードに失敗した場合
                    detail = e.response.text[:200] # レスポンスの一部
            logger.error(f"失敗: APIリクエストエラー (ステータス: {status_code}, 詳細: {detail}) - {filename}")
            failure_count += 1
        except Exception as e:
            # その他の予期せぬエラー
            logger.exception(f"失敗: 予期せぬエラーが発生しました - {filename}: {e}") # スタックトレースも記録
            failure_count += 1

    logger.info("-" * 30)
    logger.info("アップロード処理完了")
    logger.info(f"成功: {success_count} 件")
    logger.info(f"失敗: {failure_count} 件")
    logger.info("-" * 30)

# --- メイン処理 ---
if __name__ == "__main__":
    logger.info("POSTテスト スクリプトを開始します...")
    test_upload_endpoint()
    logger.info("POSTテスト スクリプトを終了します。")