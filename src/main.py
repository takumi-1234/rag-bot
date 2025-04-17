# src/main.py
import os
import logging
import shutil
from typing import List, Dict, Optional, Tuple, Any # Any を追加
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import google.api_core.exceptions

# --- 自作モジュール ---
try:
    from src.rag.chroma_manager import ChromaManager
    from src.rag.document_processor import process_documents, SUPPORTED_EXTENSIONS
    from src.rag.llm_gemini import GeminiChat
except ImportError as e:
    # このエラーは Docker ビルド時またはパス設定の問題を示す
    print(f"[FATAL] Error importing custom modules: {e}")
    print("Ensure 'chroma_manager.py', 'document_processor.py', and 'llm_gemini.py' exist in 'src/rag' directory.")
    print("Check PYTHONPATH if running outside Docker.")
    import sys
    sys.exit(1) # 起動不可なので終了

# --- 環境変数の読み込み ---
# .env ファイルが存在する場合に読み込む
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # srcディレクトリの親にある .env を想定
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from: {dotenv_path}")
else:
    print(f".env file not found at {dotenv_path}, relying on system environment variables.")


# --- ロギング設定 ---
# コンテナログでタイムスタンプが重複しないように basicConfig を設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Uvicorn のロガーにも設定を適用する場合（オプション）
# logging.getLogger("uvicorn.error").propagate = False
# logging.getLogger("uvicorn.access").propagate = False

logger = logging.getLogger(__name__)

# --- アプリケーションスコープのシングルトン ---
# Optional を使うことで、初期化前は None であることを明示
app_state: Dict[str, Any] = {
    "chroma_manager": None,
    "gemini_chat": None,
    "upload_dir": None,
    "initialized": False,
    "initialization_error": None # 初期化失敗時のエラーメッセージ
}

# --- FastAPI アプリケーションインスタンス ---
app = FastAPI(
    title="大学講義支援 RAG チャットボット API",
    description="講義資料のアップロードと、それに基づいた質問応答 (RAG) を行う API",
    version="0.3.0", # バージョン更新
    # lifespan=lifespan # FastAPI 0.10 lifespan context manager (推奨)
)

# --- CORS 設定 ---
origins = [
    "http://localhost:8501", # Streamlit default port
    "http://127.0.0.1:8501",
    # 必要に応じてデプロイ先のオリジンを追加
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- アプリケーション起動/終了イベント (Lifespan) ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application initialization...")
    global app_state
    try:
        # 環境変数取得と検証
        chroma_db_path = os.getenv("CHROMA_DB_PATH")
        embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu") # デフォルト'cpu'
        upload_dir_env = os.getenv("UPLOAD_DIR")
        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL_NAME") # llm_geminiでデフォルト設定あり

        required_vars = {
            "CHROMA_DB_PATH": chroma_db_path,
            "EMBEDDING_MODEL_NAME": embedding_model,
            "UPLOAD_DIR": upload_dir_env,
            "GEMINI_API_KEY": gemini_key
        }
        missing_vars = [k for k, v in required_vars.items() if not v]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Check .env file.")

        # ディレクトリ作成
        os.makedirs(chroma_db_path, exist_ok=True)
        os.makedirs(upload_dir_env, exist_ok=True)
        app_state["upload_dir"] = upload_dir_env
        logger.info(f"Upload directory: {os.path.abspath(upload_dir_env)}")
        logger.info(f"ChromaDB persist directory: {os.path.abspath(chroma_db_path)}")

        # ChromaManager 初期化
        app_state["chroma_manager"] = ChromaManager(
            persist_directory=chroma_db_path,
            embedding_model_name=embedding_model,
            embedding_device=embedding_device
        )
        logger.info(f"ChromaManager initialized.")

        # GeminiChat 初期化
        app_state["gemini_chat"] = GeminiChat(
            api_key=gemini_key,
            model_name=gemini_model # NoneでもOK
        )
        logger.info(f"GeminiChat initialized. Using model: {app_state['gemini_chat'].model_name}")

        app_state["initialized"] = True
        logger.info("Application initialized successfully.")

    except Exception as e:
        # 初期化失敗時のエラーを記録
        error_msg = f"Application initialization failed: {e}"
        logger.error(error_msg, exc_info=True)
        app_state["initialization_error"] = error_msg
        # ここでアプリケーションを停止させるか、エラー状態で起動するか選択
        # 例: エラー状態で起動し、ヘルスチェックでエラーを返す
        # raise RuntimeError(error_msg) from e # => アプリケーションが起動しない

    yield # ここでアプリケーションがリクエストを受け付ける

    # Shutdown
    logger.info("Application shutting down...")
    # 必要に応じてクリーンアップ処理をここに追加
    # 例: chroma_manager のクリーンアップメソッド呼び出しなど (通常は不要)
    logger.info("Shutdown complete.")

# FastAPI 0.10.x以降のlifespanを使用
app.router.lifespan_context = lifespan


# --- リクエスト/レスポンスモデル (変更なし) ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="ユーザーからの質問テキスト", min_length=1)
    k: int = Field(3, description="検索する関連文書（チャンク）の数", ge=1, le=10)

class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM によって生成された回答")
    sources: List[str] = Field([], description="回答の根拠となった資料のソース（ファイル名など）")

class UploadResponse(BaseModel):
    status: str = Field(..., description="処理結果 (success または error)")
    file: Optional[str] = Field(None, description="処理されたファイル名")
    chunks_added: Optional[int] = Field(None, description="追加/更新されたチャンク数")
    message: Optional[str] = Field(None, description="エラーまたは成功メッセージ")

class VectorStoreCountResponse(BaseModel):
    count: int = Field(..., description="ベクトルストア内のドキュメント（チャンク）総数")

class DeleteResponse(BaseModel):
    status: str = Field(..., description="処理結果 (success または error)")
    message: str = Field(..., description="処理結果メッセージ")

# --- 依存性注入関数 (初期化チェックとサービス取得) ---
# 型ヒントをより具体的に
def get_chroma_manager() -> ChromaManager:
    if not app_state["initialized"] or app_state["chroma_manager"] is None:
        logger.error("ChromaManager requested but not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not available (Initialization Error: {app_state.get('initialization_error', 'Unknown error')})"
        )
    return app_state["chroma_manager"]

def get_gemini_chat() -> GeminiChat:
    if not app_state["initialized"] or app_state["gemini_chat"] is None:
        logger.error("GeminiChat requested but not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not available (Initialization Error: {app_state.get('initialization_error', 'Unknown error')})"
        )
    return app_state["gemini_chat"]

def get_upload_dir() -> str:
    if not app_state["initialized"] or app_state["upload_dir"] is None:
        logger.error("Upload directory requested but not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not available (Initialization Error: {app_state.get('initialization_error', 'Unknown error')})"
        )
    return app_state["upload_dir"]


# --- API エンドポイント ---

@app.get("/health", summary="ヘルスチェック", tags=["システム"])
async def health_check():
    """アプリケーションの初期化状態と基本的な動作を確認します。"""
    if not app_state["initialized"]:
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail=f"Service initialization failed: {app_state.get('initialization_error', 'Unknown error')}"
         )
    # オプション: ChromaDBへの接続確認など、より詳細なチェックを追加
    try:
        count = app_state["chroma_manager"].count_documents() # 簡単なDB操作を試す
        return {"status": "ok", "initialized": True, "vector_store_count": count}
    except Exception as e:
        logger.error(f"Health check dependency error (ChromaDB?): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service dependency check failed: {e}"
        )


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    summary="講義資料のアップロード",
    status_code=status.HTTP_201_CREATED, # 成功時のステータスコード
    tags=["資料管理"]
)
async def upload_lecture_document(
    file: UploadFile = File(..., description=f"アップロードするファイル。対応形式: {', '.join(SUPPORTED_EXTENSIONS.keys())}"),
    chroma: ChromaManager = Depends(get_chroma_manager),
    upload_path: str = Depends(get_upload_dir)
    ) -> UploadResponse:
    """アップロードされたファイルを処理し、ベクトルDBに追加します。"""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename cannot be empty.")

    # 安全なファイル名を取得し、空の場合はデフォルト名を生成
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        base, ext = os.path.splitext(file.filename) # 元の拡張子を保持しようと試みる
        safe_filename = "uploaded_file_" + os.urandom(4).hex() + (ext if ext else "")
        logger.warning(f"Original filename '{file.filename}' sanitized, using default: {safe_filename}")

    file_path = os.path.join(upload_path, safe_filename)
    file_ext = os.path.splitext(safe_filename)[1].lower()

    logger.info(f"Received file upload request: {safe_filename} (Type: {file.content_type}, Extension: '{file_ext}')")

    # サポートされている拡張子かチェック
    if file_ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type attempt: {safe_filename} (Extension: '{file_ext}')")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file_ext}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    # 一時ファイルに保存
    try:
        with open(file_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved temporarily to: {file_path}")

        # ドキュメント処理 (ロード、分割)
        documents = process_documents(file_path)

        if documents:
            chunks_count = len(documents)
            # ベクトルDBに追加
            chroma.add_documents(documents)
            logger.info(f"Successfully processed and added {chunks_count} chunks from '{safe_filename}' to vector store.")
            return UploadResponse(
                status="success",
                file=safe_filename,
                chunks_added=chunks_count,
                message="File processed and added to vector store successfully."
            )
        else:
            # process_documents が空リストを返した場合 (サポート対象だが内容がない、または処理失敗)
            logger.warning(f"No processable content found or processing failed for file: {safe_filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not extract or process text from the file '{safe_filename}'. It might be empty, corrupted, password-protected, or an internal error occurred during processing."
            )

    except HTTPException as http_exc:
         # 既に HTTP 例外の場合はそのまま送出
         raise http_exc
    except Exception as e:
        # その他の予期せぬエラー
        logger.error(f"Unexpected error during file upload processing ({safe_filename}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred while processing the file: {e}"
        )
    finally:
        # 一時ファイルを削除
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Temporary file deleted: {file_path}")
            except OSError as e_remove:
                logger.error(f"Failed to delete temporary file {file_path}: {e_remove}")
        # ファイルストリームを閉じる (FastAPIが通常管理するが念のため)
        await file.close()
        logger.info(f"File stream closed for: {safe_filename}")


@app.post("/api/chat", response_model=ChatResponse, summary="RAG チャットボットとの対話", tags=["チャット"])
async def chat_with_rag_bot(
    request: ChatRequest,
    chroma: ChromaManager = Depends(get_chroma_manager),
    llm: GeminiChat = Depends(get_gemini_chat)
    ) -> ChatResponse:
    """ユーザーの質問に基づき、関連文書を検索し、LLMで回答を生成します。"""
    query = request.query
    k = request.k
    logger.info(f"Chat request received: query='{query[:100]}...', k={k}")

    try:
        # 1. ベクトルストアで類似文書を検索
        logger.info(f"Searching vector store for relevant documents (k={k})...")
        search_results: List[Document] = chroma.search(query, k=k)

        # 検索結果からソース情報を抽出
        sources: List[str] = []
        if not search_results:
            logger.info("No relevant documents found in vector store.")
        else:
            sources = sorted(list(set(
                doc.metadata.get('source')
                for doc in search_results if doc.metadata and doc.metadata.get('source')
            )))
            logger.info(f"Retrieved {len(search_results)} context documents. Sources: {sources}")

        # 2. LLM に回答生成をリクエスト
        logger.info("Generating response using LLM...")
        response_text = llm.generate_response(query=query, context_docs=search_results)
        logger.info("LLM generated response successfully.")

        return ChatResponse(response=response_text, sources=sources)

    # --- エラーハンドリング ---
    except ValueError as ve:
        # GeminiChatでの入力バリデーションエラーなど
        logger.warning(f"Chat request validation error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except google.api_core.exceptions.PermissionDenied as e_perm:
        logger.error(f"Gemini API Permission Denied: {e_perm}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied for Gemini API. Check API key or project settings.")
    except google.api_core.exceptions.ResourceExhausted as e_res:
        logger.error(f"Gemini API Resource Exhausted: {e_res}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Gemini API quota exceeded. Please try again later.")
    except google.api_core.exceptions.DeadlineExceeded as e_timeout:
        logger.error(f"Gemini API Deadline Exceeded: {e_timeout}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Gemini API request timed out.")
    except google.api_core.exceptions.InternalServerError as e_internal:
         logger.error(f"Gemini API Internal Server Error: {e_internal}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Gemini API encountered an internal server error.")
    except google.api_core.exceptions.InvalidArgument as e_invalid:
         logger.error(f"Gemini API Invalid Argument: {e_invalid}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid argument sent to Gemini API: {e_invalid}")
    except RuntimeError as re:
        # llm_gemini.py 内でのブロックや ChromaManager のエラーなど
        logger.error(f"Runtime error during chat processing: {re}", exc_info=True)
        # エラーメッセージをそのままクライアントに返すのはセキュリティリスクの可能性あり
        # 詳細なエラーはログに記録し、クライアントには一般的なエラーメッセージを返すのが良い場合も
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        # その他の予期せぬエラー
        logger.error(f"Unexpected error during chat processing (Query: '{query[:100]}...'): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred during chat processing."
        )


@app.get(
    "/api/vectorstore/count",
    response_model=VectorStoreCountResponse,
    summary="ベクトルストアのドキュメント数取得",
    tags=["ベクトルストア管理"]
)
async def get_vector_store_document_count(chroma: ChromaManager = Depends(get_chroma_manager)) -> VectorStoreCountResponse:
    """ベクトルストア内のドキュメント（チャンク）総数を取得します。"""
    try:
        count = chroma.count_documents()
        logger.info(f"Retrieved vector store document count: {count}")
        return VectorStoreCountResponse(count=count)
    except Exception as e:
        logger.error(f"Error retrieving vector store count: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve vector store count: {e}"
        )

@app.delete(
    "/api/vectorstore/delete_all",
    response_model=DeleteResponse,
    summary="ベクトルストアのコレクション削除",
    tags=["ベクトルストア管理"]
)
async def delete_all_documents_from_vector_store(chroma: ChromaManager = Depends(get_chroma_manager)) -> DeleteResponse:
    """ベクトルストア内の全ドキュメント（コレクション自体）を削除します。"""
    collection_name_to_delete = chroma.collection_name
    try:
        logger.warning(f"Request received to delete vector store collection: '{collection_name_to_delete}'")
        count_before = chroma.count_documents() # 削除前の件数を記録

        chroma.delete_collection() # 削除実行

        message = f"Vector store collection '{collection_name_to_delete}' deleted successfully (contained {count_before} documents). A new empty collection will be created on the next operation or restart."
        logger.info(message)
        return DeleteResponse(status="success", message=message)
    except RuntimeError as re:
         logger.error(f"Error deleting collection '{collection_name_to_delete}': {re}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete collection: {re}")
    except Exception as e:
        logger.error(f"Unexpected error deleting collection '{collection_name_to_delete}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during collection deletion.")

# --- Uvicornでの直接実行用 ---
if __name__ == "__main__":
    import uvicorn
    # 環境変数からポート、ホスト、リロード設定を取得
    port = int(os.getenv("PORT", 8000))
    # デフォルトのホストを 0.0.0.0 に変更し、コンテナ外部からのアクセスを許可
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("UVICORN_RELOAD", "false").lower() in ["true", "1", "t"]

    print(f"Starting Uvicorn server directly on {host}:{port} with reload={reload}")
    uvicorn.run("main:app", host=host, port=port, reload=reload)