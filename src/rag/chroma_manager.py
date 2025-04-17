# src/rag/chroma_manager.py
import chromadb
import logging
from typing import List, Optional, Dict
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
# from chromadb.utils import embedding_functions # 不要

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "university_lecture_docs"
DEFAULT_EMBEDDING_DEVICE = 'cpu' # デフォルトのデバイス

class ChromaManager:
    """ChromaDBとのインタラクションを管理するクラス (SentenceTransformer直接利用)"""

    def __init__(self, persist_directory: str, embedding_model_name: str, embedding_device: str = DEFAULT_EMBEDDING_DEVICE, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        ChromaManagerを初期化します。

        Args:
            persist_directory (str): ChromaDBデータを永続化するディレクトリ。
            embedding_model_name (str): 使用するSentenceTransformerモデル名。
            embedding_device (str): Embedding計算に使用するデバイス ('cpu' または 'cuda')。
            collection_name (str): 使用するChromaDBコレクション名。
        """
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        logger.info(f"Using embedding model: {embedding_model_name} on device: {embedding_device}")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.embedding_model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Collection] = None

        try:
            # 1. SentenceTransformer モデルを初期化
            try:
                logger.info(f"Initializing SentenceTransformer with model: {self.embedding_model_name}")
                # trust_remote_code は必要に応じて True にするが、E5モデルでは通常不要
                self.embedding_model = SentenceTransformer(
                    model_name_or_path=self.embedding_model_name,
                    device=self.embedding_device,
                    trust_remote_code=False # E5モデルでは通常 False でOK
                    # cache_folder=... # 必要ならキャッシュディレクトリ指定
                )
                logger.info("SentenceTransformer model initialized successfully.")
            except Exception as e_st_init:
                logger.error(f"Failed to initialize SentenceTransformer model: {e_st_init}", exc_info=True)
                # エラーメッセージを一般化
                raise RuntimeError(f"Failed to initialize Embedding Model '{self.embedding_model_name}': {e_st_init}") from e_st_init

            # 2. ChromaDBクライアントを初期化
            logger.info(f"Initializing ChromaDB PersistentClient at path: {self.persist_directory}")
            try:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info("ChromaDB PersistentClient initialized successfully.")
            except Exception as e_client:
                logger.error(f"Failed to initialize ChromaDB PersistentClient: {e_client}", exc_info=True)
                raise RuntimeError(f"Failed to initialize ChromaDB Client: {e_client}") from e_client

            # 3. コレクション取得または作成
            # embedding_function を指定せずにコレクションを作成
            logger.info(f"Attempting to get or create ChromaDB collection: '{self.collection_name}' without default embedding function")
            try:
                # コレクション作成時にメタデータで距離空間を指定することを推奨
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"} # or "l2" for E5 models, check model card
                )
                logger.info(f"Successfully retrieved or created ChromaDB collection '{self.collection.name}'.")
            except Exception as e_coll:
                logger.error(f"Failed to get or create ChromaDB collection '{self.collection_name}': {e_coll}", exc_info=True)
                raise RuntimeError(f"Failed to initialize ChromaDB collection: {e_coll}") from e_coll

            if self.collection is None:
                logger.error(f"ChromaDB collection '{self.collection_name}' is None after get_or_create_collection.")
                raise RuntimeError(f"Failed to obtain a valid ChromaDB collection object for '{self.collection_name}'.")

        # --- 初期化中のその他のエラーハンドリング ---
        except ImportError as ie:
            logger.error(f"Failed to import necessary libraries: {ie}", exc_info=True)
            raise RuntimeError(f"Initialization failed due to missing libraries: {ie}") from ie
        except Exception as e: # ValueError, RuntimeError などを含む
            logger.error(f"Failed to initialize ChromaManager: {e}", exc_info=True)
            # 既にRuntimeErrorにラップされている場合はそのまま送出、そうでなければラップ
            if isinstance(e, RuntimeError):
                raise e
            else:
                raise RuntimeError(f"Failed to initialize ChromaManager: {e}") from e

        # --- 初期化完了確認 ---
        if self.collection:
             try:
                 count = self.collection.count()
                 logger.info(f"ChromaManager initialized successfully with collection '{self.collection.name}'. Initial document count: {count}")
             except Exception as e_count:
                 logger.warning(f"ChromaManager initialized collection '{self.collection.name}', but failed to get initial count: {e_count}", exc_info=True)
        else:
             logger.error("ChromaManager initialization finished, but self.collection is unexpectedly None.")

    def _generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """テキストリストの埋め込みベクトルを生成"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model is not initialized.")
        if not texts:
            return []
        logger.debug(f"Generating embeddings for {len(texts)} texts using {self.embedding_model_name}...")
        try:
            # E5モデルなど、特定のプレフィックスが必要な場合があるかもしれないが、
            # SentenceTransformerのデフォルト動作に任せる
            # query: "query: " + text
            # passage: "passage: " + text
            # multilingual-e5-large はプレフィックス不要なことが多い
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True, # 正規化を推奨 (特にコサイン類似度を使う場合)
                batch_size=32 # バッチサイズ調整可能
                # show_progress_bar=True # 進捗表示が必要な場合
            )
            return embeddings.tolist() if embeddings is not None else []
        except Exception as e:
            logger.error(f"Error during text embedding: {e}", exc_info=True)
            return None # エラー時はNoneを返す

    def add_documents(self, documents: List[Document]):
        """ドキュメントリストをChromaDBコレクションに追加します。"""
        if not self.collection:
            logger.error("Collection is not initialized. Cannot add documents.")
            raise RuntimeError("ChromaDB collection is not available.")
        if not self.embedding_model:
             logger.error("Embedding model is not initialized. Cannot add documents.")
             raise RuntimeError("Embedding model is not available.")
        if not documents:
            logger.warning("No documents provided to add.")
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # ID生成ロジック (以前と同じ、衝突の可能性は低いが一意性を保証するものではない)
        ids = []
        doc_counts: Dict[str, int] = {}
        for i, doc in enumerate(documents):
            source = str(doc.metadata.get('source', 'unknown_source')) # 念のため文字列に
            count = doc_counts.get(source, 0)
            # ID にハッシュを含めることで内容の僅かな違いも区別する
            ids.append(f"{source}_{count}_{i}_{hash(doc.page_content)}")
            doc_counts[source] = count + 1

        try:
            # 内部メソッドでベクトルを計算
            logger.info(f"Generating embeddings for {len(texts)} document chunks...")
            embeddings = self._generate_embeddings(texts)

            if embeddings is None or len(embeddings) != len(texts):
                 error_msg = f"Mismatch or failure in generating embeddings. Texts: {len(texts)}, Embeddings generated: {len(embeddings) if embeddings is not None else 'None'}."
                 logger.error(error_msg)
                 raise RuntimeError(f"Failed to generate embeddings correctly for all documents. {error_msg}")

            logger.info(f"Embeddings generated successfully for {len(embeddings)} chunks.")
            logger.info(f"Adding {len(documents)} documents with pre-generated embeddings to collection '{self.collection_name}'...")

            # add メソッドには embeddings を渡す
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts, # ドキュメントテキストも保存
                ids=ids
            )
            logger.info(f"Successfully added/updated {len(documents)} documents in ChromaDB collection '{self.collection_name}'.")
            # DBへの永続化を明示的に行う必要はない (PersistentClientが管理)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            # 重複IDエラーに関するヒント
            if "UniqueConstraintError" in str(e) or "ID already exists" in str(e):
                 logger.error("Hint: Duplicate IDs might be generated. Check ID generation logic or consider if content is truly identical.")
            raise RuntimeError(f"Failed to add documents to ChromaDB: {e}") from e

    def search(self, query: str, k: int = 3) -> List[Document]:
        """クエリテキストに基づいて類似ドキュメントを検索します。"""
        if not self.collection:
             logger.error("Collection is not initialized. Cannot search.")
             raise RuntimeError("ChromaDB collection is not available.")
        if not self.embedding_model:
             logger.error("Embedding model is not initialized. Cannot search.")
             raise RuntimeError("Embedding model is not available.")

        try:
            # クエリベクトルを計算
            logger.info(f"Embedding query using {self.embedding_model_name}: '{query[:50]}...'")
            # _generate_embeddingsはリストを受け取るので、クエリをリストに入れる
            query_embedding_list = self._generate_embeddings([query])

            if query_embedding_list is None or not query_embedding_list:
                logger.error("Failed to generate query embedding.")
                return []
            query_embedding = query_embedding_list[0] # 最初の要素（クエリのベクトル）を取得

            logger.info(f"Searching ChromaDB collection '{self.collection_name}' with query embedding (k={k})...")
            results = self.collection.query(
                query_embeddings=[query_embedding], # リストとして渡す
                n_results=k,
                include=['documents', 'metadatas', 'distances'] # 必要な情報を指定
            )

            # --- 検索結果をLangChain Document形式に変換 (変更なし) ---
            retrieved_docs: List[Document] = []
            if results and results.get('ids') and results['ids'] and results['ids'][0]:
                num_results = len(results['ids'][0])
                logger.info(f"Found {num_results} similar document(s) from ChromaDB query.")
                for i in range(num_results):
                    # 安全なアクセス (インデックスが存在するか確認)
                    doc_id = results['ids'][0][i] if results['ids'][0] and i < len(results['ids'][0]) else 'N/A'
                    content = results['documents'][0][i] if results.get('documents') and results['documents'][0] and i < len(results['documents'][0]) else ""
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]) else {}
                    distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] and i < len(results['distances'][0]) else None

                    if metadata is None: metadata = {} # メタデータがNoneの場合空辞書に
                    if distance is not None:
                        metadata['distance'] = distance # メタデータに距離を追加
                        distance_str = f"{distance:.4f}"
                    else:
                        distance_str = "N/A"

                    logger.debug(f"Retrieved doc {i+1}: ID={doc_id}, Distance={distance_str}, Source={metadata.get('source', 'N/A')}")
                    retrieved_docs.append(Document(page_content=content, metadata=metadata))
            else:
                logger.info("No similar documents found from ChromaDB query.")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during similarity search in ChromaDB: {e}", exc_info=True)
            return []

    def count_documents(self) -> int:
        """コレクション内のドキュメント数を返します。"""
        if not self.collection:
            logger.warning("Collection is not initialized. Returning count 0.")
            return 0
        try:
            count = self.collection.count()
            logger.info(f"Current document count in collection '{self.collection_name}': {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting documents in ChromaDB: {e}", exc_info=True)
            return 0 # エラー時は 0 を返す

    def delete_collection(self):
        """現在のコレクションを削除します。"""
        if not self.client:
             logger.error("Client not initialized. Cannot delete collection.")
             raise RuntimeError("ChromaDB client not available.")
        if not self.collection_name:
             logger.error("Collection name is not set. Cannot delete.")
             return # 何もせず終了

        try:
             logger.warning(f"Attempting to delete ChromaDB collection: '{self.collection_name}'")
             # コレクションが存在するか確認してから削除 (エラーを避ける)
             try:
                 existing_collections = [c.name for c in self.client.list_collections()]
                 if self.collection_name not in existing_collections:
                     logger.warning(f"Collection '{self.collection_name}' does not exist, cannot delete.")
                     self.collection = None # ローカルの参照もクリア
                     return # 存在しない場合は削除処理不要
             except Exception as e_list:
                 # リスト取得に失敗しても削除を試みる (警告ログのみ)
                 logger.warning(f"Could not verify existence of collection before deletion: {e_list}")

             self.client.delete_collection(name=self.collection_name)
             self.collection = None # 削除成功後、ローカルの参照もクリア
             logger.info(f"Successfully deleted collection: '{self.collection_name}'")
        except ValueError as ve: # delete_collection が存在しない場合に発生することもある
             logger.warning(f"Collection '{self.collection_name}' likely does not exist or already deleted: {ve}")
             self.collection = None # 念のためクリア
        except Exception as e:
             logger.error(f"Error deleting collection '{self.collection_name}': {e}", exc_info=True)
             # エラーが発生しても、上位にエラーを伝える
             raise RuntimeError(f"Failed to delete collection '{self.collection_name}': {e}") from e