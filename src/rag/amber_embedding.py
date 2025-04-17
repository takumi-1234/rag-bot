
# src/rag/amber_embedding.py
import logging
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class AmberHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    AMBERモデル用にカスタマイズされたHuggingFaceEmbeddingsクラス。
    クエリとドキュメントで異なる prompt_name を使用します。
    """
    def __init__(self, **kwargs: Any):
        """
        通常の HuggingFaceEmbeddings と同様に初期化します。
        内部の SentenceTransformer クライアントを使用します。
        """
        super().__init__(**kwargs)
        if not isinstance(self.client, SentenceTransformer):
             # 通常 HuggingFaceEmbeddings は内部で SentenceTransformer を client として持つ
             logger.warning(f"Expected self.client to be SentenceTransformer, but got {type(self.client)}. Prompt name handling might not work as expected.")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        ドキュメントリストの埋め込みベクトルを生成します。
        prompt_name="Retrieval-passage" を使用します。
        """
        if not texts:
             return []
        logger.debug(f"Embedding {len(texts)} documents using AMBER with prompt_name='Retrieval-passage'")
        # HuggingFaceEmbeddings の encode_kwargs を取得し、prompt_name を追加（または上書き）
        encode_kwargs = self.encode_kwargs.copy()
        encode_kwargs["prompt_name"] = "Retrieval-passage"

        # SentenceTransformerのencodeメソッドを使用
        try:
            embeddings = self.client.encode(texts, **encode_kwargs)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during AMBER document embedding: {e}", exc_info=True)
            # エラー発生時は空リストやNoneを返すか、例外を再送出するか検討
            # ここでは空リストを返す例
            return []

    def embed_query(self, text: str) -> List[float]:
        """
        単一クエリの埋め込みベクトルを生成します。
        prompt_name="Retrieval-query" を使用します。
        """
        logger.debug(f"Embedding query using AMBER with prompt_name='Retrieval-query'")
        # HuggingFaceEmbeddings の encode_kwargs を取得し、prompt_name を追加（または上書き）
        encode_kwargs = self.encode_kwargs.copy()
        encode_kwargs["prompt_name"] = "Retrieval-query"

        # SentenceTransformerのencodeメソッドを使用
        try:
            embedding = self.client.encode(text, **encode_kwargs)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error during AMBER query embedding: {e}", exc_info=True)
            # エラー発生時の処理
            return [] # または例外を送出