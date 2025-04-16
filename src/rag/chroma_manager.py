# src/rag/chroma_manager.py
import chromadb
import logging
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings # 新しい推奨
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# コレクション名を定数化
DEFAULT_COLLECTION_NAME = "university_lecture_docs"

class ChromaManager:
    """ChromaDBとのインタラクションを管理するクラス"""

    def __init__(self, persist_directory: str, embedding_model_name: str, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        ChromaDBクライアントとコレクションを初期化します。
        コレクションが存在しない場合は作成します。
        """
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        logger.info(f"Using embedding model: {embedding_model_name}")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.langchain_embedding_function: Optional[HuggingFaceEmbeddings] = None # LangChainインターフェース用
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Collection] = None # ChromaDB Collectionオブジェクト

        try:
            # LangChain用のEmbedding関数を初期化
            self.langchain_embedding_function = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cpu'}, # 環境に合わせて 'cuda' or 'cpu'
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("HuggingFaceEmbeddings (for LangChain) initialized.")

            # ChromaDBクライアントを初期化
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("ChromaDB PersistentClient initialized.")

            # ChromaDBネイティブのEmbedding関数を初期化 (get_or_create_collection用)
            chromadb_embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
                # device='cpu' # 必要に応じてデバイス指定
            )
            logger.info("ChromaDB SentenceTransformerEmbeddingFunction (for collection) initialized.")

            # コレクションを取得または作成
            # ここで embedding_function を渡すことで、コレクションが正しい関数で作成される
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=chromadb_embedding_function
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' retrieved or created successfully.")

        except ImportError as ie:
            logger.error(f"Failed to import necessary libraries: {ie}", exc_info=True)
            raise RuntimeError(f"Initialization failed due to missing libraries: {ie}") from ie
        except Exception as e:
            logger.error(f"Failed to initialize ChromaManager: {e}", exc_info=True)
            # エラー内容をラップして再送出
            raise RuntimeError(f"Failed to initialize ChromaManager: {e}") from e

    def add_documents(self, documents: List[Document]):
        """ドキュメントリストをChromaDBコレクションに追加または更新します (upsert)。"""
        if not self.collection:
            logger.error("Collection is not initialized. Cannot upsert documents.")
            raise RuntimeError("ChromaDB collection is not available.")
        if not documents:
            logger.warning("No documents provided to upsert.")
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # ID生成: source と チャンクインデックスで一意なIDを生成
        ids = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'unknown_source')
            # IDに使用できない文字を置換/削除 (より安全な方法)
            safe_source = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in source)
            # ページ番号があればIDに含める (PyPDFLoaderなどが付与する場合)
            page_num = doc.metadata.get('page', None)
            if page_num is not None:
                 # チャンクインデックスは、ファイル内で一意であれば良いので i を使う
                 ids.append(f"doc_{safe_source}_p{page_num}_c{i}")
            else:
                 ids.append(f"doc_{safe_source}_c{i}")

        if not ids:
             logger.warning("Generated empty IDs list, cannot upsert.")
             return

        try:
            logger.info(f"Upserting {len(documents)} documents/chunks into collection '{self.collection_name}'...")
            # add の代わりに upsert を使用
            self.collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully upserted {len(documents)} documents/chunks.")
        except Exception as e:
            logger.error(f"Error upserting documents into ChromaDB: {e}", exc_info=True)
            # エラーをラップして再送出
            raise RuntimeError(f"Failed to upsert documents into ChromaDB: {e}") from e

    def search(self, query: str, k: int = 3) -> List[Document]:
        """クエリテキストに基づいて類似ドキュメントを検索します。"""
        if not self.collection or not self.langchain_embedding_function:
            logger.error("Collection or embedding function is not initialized. Cannot search.")
            raise RuntimeError("ChromaDB collection or embedding function is not available.")

        try:
            logger.info(f"Searching for documents similar to query: '{query[:50]}...' (k={k})")
            # LangChainのEmbedding関数でクエリをベクトル化
            query_embedding = self.langchain_embedding_function.embed_query(query)

            # ChromaDBコレクションで検索
            # include で 'embeddings' は通常不要なので削除
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # 結果をLangchainのDocument形式に変換
            retrieved_docs: List[Document] = []
            # 結果が存在するか、より安全にチェック
            if results and results.get('ids') and results['ids'] and results['ids'][0]:
                num_results = len(results['ids'][0])
                logger.debug(f"Raw search results count: {num_results}")
                for i in range(num_results):
                    # 各要素が存在するか確認しながらアクセス
                    content = results['documents'][0][i] if results.get('documents') and results['documents'][0] and len(results['documents'][0]) > i else ""
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] and len(results['metadatas'][0]) > i else {}
                    distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] and len(results['distances'][0]) > i else None

                    # メタデータがNoneでないことを保証
                    if metadata is None:
                        metadata = {}
                    if distance is not None:
                        metadata['distance'] = distance # 距離情報もメタデータに追加

                    retrieved_docs.append(Document(page_content=content, metadata=metadata))

                logger.info(f"Found {len(retrieved_docs)} similar documents.")
            else:
                logger.info("No similar documents found.")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during similarity search in ChromaDB: {e}", exc_info=True)
            return [] # エラー時は空リストを返す

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
        """現在のコレクションをChromaDBから削除します。"""
        if not self.client:
             logger.error("Client not initialized. Cannot delete collection.")
             raise RuntimeError("ChromaDB client not available.")
        if not self.collection_name:
             logger.error("Collection name is not set. Cannot delete.")
             return

        try:
             logger.warning(f"Attempting to delete ChromaDB collection: '{self.collection_name}'")
             self.client.delete_collection(name=self.collection_name)
             self.collection = None # 削除されたのでNoneに更新
             logger.info(f"Successfully deleted collection: '{self.collection_name}'")
             # 再度 get_or_create_collection を呼び出すことで空のコレクションを再作成する
             # (削除後にすぐ必要になる場合)
             # self.collection = self.client.get_or_create_collection(name=self.collection_name, ...)
        except ValueError:
             # ChromaDB < 0.4.14 では存在しないコレクションを削除しようとすると ValueError
             logger.warning(f"Collection '{self.collection_name}' does not exist or already deleted, cannot delete.")
             self.collection = None # 存在しない場合もNoneに更新
        except Exception as e:
             logger.error(f"Error deleting collection '{self.collection_name}': {e}", exc_info=True)
             # 削除に失敗した場合でも、アクセスできないようにNoneにしておく
             self.collection = None
             raise RuntimeError(f"Failed to delete collection '{self.collection_name}': {e}") from e