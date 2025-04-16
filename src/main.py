# src/main.py
import os
import logging
import shutil
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from werkzeug.utils import secure_filename # ファイル名サニタイズのために追加

# --- 自作モジュール ---
# (これらのファイルが src/rag ディレクトリに存在することを確認してください)
try:
    from src.rag.chroma_manager import ChromaManager
    from src.rag.document_processor import process_documents, SUPPORTED_EXTENSIONS
    from src.rag.llm_gemini import GeminiChat
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'chroma_manager.py', 'document_processor.py', and 'llm_gemini.py' exist in the 'src/rag' directory.")
    import sys
    sys.exit(1)

# --- 環境変数の読み込み ---
load_dotenv()

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI アプリケーション設定 ---
# lifespan を使うために Depends からグローバル変数へ戻すアプローチも検討
# または、lifespan内で依存関係を解決するファクトリパターンなど
# ここではシンプルにするため、アプリケーションスコープのシングルトンとして定義
chroma_manager: Optional[ChromaManager] = None
gemini_chat: Optional[GeminiChat] = None
upload_dir: Optional[str] = None
initialized: bool = False # 初期化成功フラグ

app = FastAPI(
    title="大学講義支援 RAG チャットボット API",
    description="講義資料のアップロードと、それに基づいた質問応答 (RAG) を行う API",
    version="0.2.2", # バージョン更新
)

# --- CORS 設定 (ローカル開発用に許可) ---
origins = [
    "http://localhost:8501", # Streamlit default port
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- アプリケーション起動時の初期化処理 ---
@app.on_event("startup")
async def startup_event():
    global chroma_manager, gemini_chat, upload_dir, initialized
    logger.info("Starting application initialization...")
    try:
        # 環境変数から設定値を取得
        chroma_db_path = os.getenv("CHROMA_DB_PATH")
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        upload_dir_env = os.getenv("UPLOAD_DIR")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

        # 必須の環境変数が設定されているかチェック
        required_vars = {
            "CHROMA_DB_PATH": chroma_db_path,
            "EMBEDDING_MODEL_NAME": embedding_model_name,
            "UPLOAD_DIR": upload_dir_env,
            "GEMINI_API_KEY": gemini_api_key
        }
        missing_vars = [k for k, v in required_vars.items() if not v]
        if missing_vars:
            raise ValueError(f"以下の必須環境変数が設定されていません: {', '.join(missing_vars)}。 .env ファイルを確認してください。")

        upload_dir = upload_dir_env

        # ディレクトリが存在しない場合は作成
        os.makedirs(chroma_db_path, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"Upload directory set to: {os.path.abspath(upload_dir)}")
        logger.info(f"ChromaDB persist directory set to: {os.path.abspath(chroma_db_path)}")

        # ChromaManager の初期化
        chroma_manager = ChromaManager(
            persist_directory=chroma_db_path,
            embedding_model_name=embedding_model_name
        )
        logger.info(f"ChromaManager initialized successfully.")

        # GeminiChat の初期化
        gemini_chat = GeminiChat(api_key=gemini_api_key, model_name=gemini_model_name)
        logger.info(f"GeminiChat initialized. Model: {gemini_model_name}")

        initialized = True # 全ての初期化が成功
        logger.info("Application initialized successfully.")

    except ValueError as ve:
        logger.error(f"Initialization error (environment variables): {ve}")
    except ImportError as ie:
        logger.error(f"Initialization error (module import): {ie}")
    except RuntimeError as re:
        logger.error(f"Initialization error (service initialization): {re}")
    except Exception as e:
        logger.error(f"Unexpected error during application startup: {e}", exc_info=True)
        # initialized は False のまま

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    # 必要であればリソース解放処理などをここに追加
    # 例: chroma_manager.client.close() など (ただし ChromaDB クライアントは通常不要)

# --- リクエスト/レスポンスモデル ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="ユーザーからの質問テキスト", min_length=1) # min_length追加
    k: int = Field(3, description="検索する関連文書（チャンク）の数", ge=1, le=10)

class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM によって生成された回答")
    sources: List[str] = Field([], description="回答の根拠となった資料のソース（ファイル名など）")

class UploadResponse(BaseModel):
    status: str = Field(..., description="処理結果 (success または error)")
    file: Optional[str] = Field(None, description="処理されたファイル名")
    chunks_added: Optional[int] = Field(None, description="追加/更新されたチャンク数") # add -> upsert に伴い修正
    message: Optional[str] = Field(None, description="エラーまたは成功メッセージ")

class VectorStoreCountResponse(BaseModel):
    count: int = Field(..., description="ベクトルストア内のドキュメント（チャンク）総数")

class DeleteResponse(BaseModel):
    status: str = Field(..., description="処理結果 (success または error)")
    message: str = Field(..., description="処理結果メッセージ")

# --- 依存性注入関数 (初期化チェック用) ---
async def get_initialized_services() -> Dict[str, object]:
    """
    アプリケーションが正常に初期化されているか確認し、
    初期化済みのサービスオブジェクトを返します。
    初期化されていない場合は 503 Service Unavailable エラーを発生させます。
    """
    # グローバル変数を参照
    if not initialized or chroma_manager is None or gemini_chat is None or upload_dir is None:
        logger.error("Service not initialized request received.")
        raise HTTPException(status_code=503, detail="サービスは現在利用できません (初期化失敗)。サーバー管理者にご連絡ください。")
    return {
        "chroma_manager": chroma_manager,
        "gemini_chat": gemini_chat,
        "upload_dir": upload_dir
    }

# --- API エンドポイント ---

@app.get("/health", summary="ヘルスチェック", tags=["システム"])
async def health_check():
    """アプリケーションが正常に起動し、初期化されているか確認します。"""
    # グローバル変数を参照
    if not initialized:
         raise HTTPException(status_code=503, detail="サービス初期化失敗")
    logger.info("Health check requested: Service is healthy.")
    return {"status": "ok", "initialized": initialized}

@app.post("/api/upload", response_model=UploadResponse, summary="講義資料のアップロード", tags=["資料管理"])
async def upload_lecture_document(
    file: UploadFile = File(..., description=f"アップロードするファイル。対応形式: {', '.join(SUPPORTED_EXTENSIONS)}"),
    services: dict = Depends(get_initialized_services)
    ) -> UploadResponse:
    """
    講義資料ファイル (PDF, DOCX, TXT) をアップロードし、テキストを抽出してベクトル化し、
    ベクトルデータベースに保存（または更新）します。
    """
    chroma: ChromaManager = services["chroma_manager"]
    upload_path: str = services["upload_dir"]

    # ファイル名をサニタイズ
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイル名がありません。")
    # secure_filename を使用して安全なファイル名を取得
    safe_filename = secure_filename(file.filename)
    if not safe_filename: # サニタイズの結果、空になった場合など
        safe_filename = "uploaded_file" # デフォルト名
    file_path = os.path.join(upload_path, safe_filename)

    # ファイル拡張子の検証
    file_ext = os.path.splitext(safe_filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type uploaded: {safe_filename}")
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないファイル形式です: '{file_ext}'。サポート形式: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"ファイル受信開始: {safe_filename}")

    try:
        # ファイルを一時保存
        with open(file_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.info(f"ファイル保存完了: {file_path}")

        # ドキュメント処理とベクトルDBへの追加/更新
        documents = process_documents(file_path)
        if documents:
            chunks_count = len(documents)
            # add_documents は upsert 動作 (chroma_manager.py で修正済みと仮定)
            chroma.add_documents(documents)
            logger.info(f"{safe_filename} から {chunks_count} 個のチャンクを処理し、ベクトルDBに追加/更新しました。")
            return UploadResponse(status="success", file=safe_filename, chunks_added=chunks_count, message="ファイルの処理とベクトルDBへの追加/更新が完了しました。")
        else:
            logger.warning(f"ファイルからテキストを抽出または分割できませんでした: {safe_filename}")
            # 一時ファイルが存在すれば削除
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"テキスト抽出失敗のため一時ファイルを削除しました: {file_path}")
                except OSError as e:
                    logger.error(f"一時ファイルの削除に失敗しました {file_path}: {e}")
            raise HTTPException(status_code=400, detail="ファイルからテキストを抽出または分割できませんでした。空のファイルか、非対応のフォーマット、または処理中にエラーが発生した可能性があります。")

    except HTTPException as http_exc:
         # 既に HTTPException が発生している場合はそのまま再送出
         raise http_exc
    except Exception as e:
        logger.error(f"ファイル処理中に予期せぬエラーが発生しました ({safe_filename}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"サーバー内部エラーが発生しました。ファイル処理に失敗しました: {e}")
    finally:
        # 正常終了・異常終了問わず、一時ファイルを削除 (ただし抽出失敗時は上記で削除済みの場合あり)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"処理完了のため一時ファイルを削除しました: {file_path}")
            except OSError as e:
                logger.error(f"一時ファイルの削除に失敗しました {file_path}: {e}")
        # アップロードされたファイルのストリームを閉じる
        await file.close()
        logger.info(f"ファイルストリームを閉じました: {safe_filename}")


@app.post("/api/chat", response_model=ChatResponse, summary="RAG チャットボットとの対話", tags=["チャット"])
async def chat_with_rag_bot(
    request: ChatRequest,
    services: dict = Depends(get_initialized_services)
    ) -> ChatResponse:
    """
    ユーザーからの質問を受け取り、ベクトルストアから関連文書を検索し、
    その情報をコンテキストとして大規模言語モデル (LLM) に渡し、回答を生成させます。
    """
    chroma: ChromaManager = services["chroma_manager"]
    llm: GeminiChat = services["gemini_chat"]
    query = request.query
    k = request.k

    logger.info(f"チャットリクエスト受信: query='{query[:100]}...', k={k}") # クエリが長い場合を考慮してログを抑制

    try:
        # 1. ベクトルストアで関連文書を検索
        logger.info(f"ベクトルストアで類似文書を検索中 (k={k})...")
        search_results: List[Document] = chroma.search(query, k=k)

        sources: List[str] = []
        if not search_results:
            logger.info("関連文書が見つかりませんでした。")
        else:
            # 取得したDocumentオブジェクトからソースファイル名を取得 (重複除去)
            sources = sorted(list(set([
                doc.metadata.get('source', '不明なソース')
                for doc in search_results if doc.metadata and doc.metadata.get('source')
            ])))
            logger.info(f"{len(search_results)} 件の関連コンテキストを取得しました。ソース: {sources}")

        # 2. LLM に回答生成をリクエスト (引数名を context_docs に修正)
        logger.info("LLM に回答生成をリクエストします...")
        response_text = llm.generate_response(query=query, context_docs=search_results) # ここを修正！
        logger.info("LLM から回答を受信しました。")

        # 3. レスポンスを返す
        return ChatResponse(response=response_text, sources=sources)

    except HTTPException as http_exc:
        # search や generate_response 内で HTTPException が発生した場合
        raise http_exc
    except Exception as e:
        logger.error(f"チャット処理中に予期せぬエラーが発生しました (Query: '{query[:100]}...'): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"チャット処理中に予期せぬサーバーエラーが発生しました。")

@app.get("/api/vectorstore/count", response_model=VectorStoreCountResponse, summary="ベクトルストアのドキュメント数取得", tags=["ベクトルストア管理"])
async def get_vector_store_document_count(services: dict = Depends(get_initialized_services)) -> VectorStoreCountResponse:
    """ベクトルストアに格納されているドキュメント（チャンク）の総数を取得します。"""
    chroma: ChromaManager = services["chroma_manager"]
    try:
        count = chroma.count_documents()
        logger.info(f"ベクトルストアのドキュメント数を取得しました: {count}")
        return VectorStoreCountResponse(count=count)
    except Exception as e:
        logger.error(f"ベクトルストアのカウント取得中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"カウント取得中にサーバーエラーが発生しました。")

@app.delete("/api/vectorstore/delete_all", response_model=DeleteResponse, summary="ベクトルストアのコレクション削除", tags=["ベクトルストア管理"])
async def delete_all_documents_from_vector_store(services: dict = Depends(get_initialized_services)) -> DeleteResponse:
    """
    ベクトルストアのコレクション全体を削除します。
    **注意:** この操作は元に戻せません。本番環境での使用は特に注意してください。
    """
    chroma: ChromaManager = services["chroma_manager"]
    collection_name_to_delete = chroma.collection_name # 削除するコレクション名を取得
    try:
        logger.warning(f"ベクトルストアのコレクション '{collection_name_to_delete}' の削除リクエストを受け付けました。")
        count_before = chroma.count_documents() # 削除前の件数をログに残す

        chroma.delete_collection() # ChromaManagerのメソッドを使用

        message = f"ベクトルストアのコレクション '{collection_name_to_delete}' を削除しました (以前の件数: {count_before})。次回起動時に空のコレクションが再作成されます。"
        logger.info(message)
        return DeleteResponse(status="success", message=message)
    except RuntimeError as re:
         logger.error(f"コレクション '{collection_name_to_delete}' の削除中にエラーが発生しました: {re}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"コレクション削除中にエラーが発生しました: {re}")
    except Exception as e:
        logger.error(f"コレクション '{collection_name_to_delete}' の削除中に予期せぬエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"コレクション削除中に予期せぬサーバーエラーが発生しました。")

# --- Uvicornでの直接実行用 (主にデバッグ目的) ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    logger.info(f"Starting Uvicorn server directly on {host}:{port} with reload={reload}")
    # 初期化が失敗していても Uvicorn は起動するが、API は 503 を返す
    uvicorn.run("main:app", host=host, port=port, reload=reload)