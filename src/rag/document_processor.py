# src/rag/document_processor.py
import os
import logging
from typing import List, Dict, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- ファイルローダー (必要に応じて追加・変更) ---
# pypdf と python-docx が requirements.txt にあることを確認
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
except ImportError:
    print("Required loaders (PyPDFLoader, TextLoader, Docx2txtLoader) not found.")
    print("Please ensure 'langchain-community', 'pypdf', 'docx2txt' are installed.")
    # エラーにするか、代替手段を考える
    PyPDFLoader = TextLoader = Docx2txtLoader = None # 見つからない場合は None に

logger = logging.getLogger(__name__)

# --- 定数 ---
# サポートするファイル拡張子と対応するローダーのマッピング
# TextLoader は encoding を指定した方が良い場合がある
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

# テキスト分割の設定 (調整可能)
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,        # 各チャンクのおおよそのサイズ（文字数）
    "chunk_overlap": 200,       # チャンク間のオーバーラップ（文字数）
    # 優先度の高い順にセパレータを指定
    "separators": ["\n\n", "\n", "。", "、", " ", ""],
    "length_function": len,     # 文字数の計算方法
    "add_start_index": True,    # チャンクの開始位置をメタデータに追加するか
}

# --- 関数 ---

def load_document(file_path: str) -> List[Document]:
    """ファイルパスから適切なローダーを使ってドキュメントを読み込む"""
    file_ext = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_EXTENSIONS.get(file_ext)

    if loader_class is None:
        logger.warning(f"Unsupported file type: {file_ext} for file {file_path}")
        return [] # 空リストを返す

    if loader_class is None: # ImportError で None になっている場合
        logger.error(f"Loader for {file_ext} is not available due to missing dependencies.")
        return []

    try:
        # encoding を指定できるローダーの場合は指定する (例: TextLoader)
        if loader_class == TextLoader:
            # UTF-8 で試みる。他のエンコーディングが必要な場合は変更。
            loader = loader_class(file_path, encoding='utf-8')
        else:
            loader = loader_class(file_path)

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path}")
        return documents
    except FileNotFoundError:
        logger.error(f"File not found during loading: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
        return []

def split_documents(documents: List[Document]) -> List[Document]:
    """ドキュメントを指定された設定でチャンクに分割する"""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(split_docs)} chunks.")
        # 各チャンクの文字数などの統計情報をログ出力 (デバッグ用)
        # for i, doc in enumerate(split_docs):
        #    logger.debug(f"Chunk {i+1}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
        return split_docs
    except Exception as e:
        logger.error(f"Error splitting documents: {e}", exc_info=True)
        return [] # エラー発生時は空リストを返す

def process_documents(file_path: str) -> List[Document]:
    """
    単一のファイルをロードし、チャンクに分割して、メタデータ（ソースファイル名）を付与する。
    :param file_path: 処理対象のファイルパス
    :return: チャンクに分割された Document オブジェクトのリスト
    """
    logger.info(f"Processing document: {file_path}")

    # 1. ドキュメントをロード
    loaded_docs = load_document(file_path)
    if not loaded_docs:
        logger.warning(f"No documents were loaded from {file_path}.")
        return []

    # 2. ドキュメントをチャンクに分割
    split_docs = split_documents(loaded_docs)
    if not split_docs:
        logger.warning(f"Failed to split documents from {file_path} into chunks.")
        return []

    # 3. 各チャンクにソースファイル名のメタデータを付与 (既存のメタデータに追加/上書き)
    source_filename = os.path.basename(file_path)
    for doc in split_docs:
        # doc.metadata が存在しない場合や辞書でない場合に初期化
        if not isinstance(doc.metadata, dict):
             doc.metadata = {}
        doc.metadata["source"] = source_filename
        # 必要であれば他のメタデータも追加 (例: ページ番号など)
        # 'page' メタデータは PyPDFLoader などが自動で追加することがある

    logger.info(f"Finished processing {file_path}. Generated {len(split_docs)} chunks.")
    return split_docs

# --- 直接実行時のテスト用コード (オプション) ---
if __name__ == "__main__":
    # このファイルが直接実行された場合のテストコード
    logging.basicConfig(level=logging.INFO) # ロギングを有効化
    test_file_pdf = "example.pdf" # テスト用のファイルパスを指定
    test_file_txt = "example.txt"

    # --- PDFファイルのテスト ---
    if os.path.exists(test_file_pdf):
        print(f"\n--- Testing PDF Processing ({test_file_pdf}) ---")
        pdf_chunks = process_documents(test_file_pdf)
        if pdf_chunks:
            print(f"Number of chunks generated from PDF: {len(pdf_chunks)}")
            print("First chunk metadata:", pdf_chunks[0].metadata)
            print("First chunk content (first 100 chars):", pdf_chunks[0].page_content[:100])
        else:
            print("PDF processing failed.")
    else:
        print(f"Test PDF file not found: {test_file_pdf}")

    # --- TXTファイルのテスト ---
    if os.path.exists(test_file_txt):
         # テスト用テキストファイルを作成
         with open(test_file_txt, "w", encoding="utf-8") as f:
              f.write("これはテスト用のテキストファイルです。\n複数行のテスト。\n\n段落もテストします。")

         print(f"\n--- Testing TXT Processing ({test_file_txt}) ---")
         txt_chunks = process_documents(test_file_txt)
         if txt_chunks:
             print(f"Number of chunks generated from TXT: {len(txt_chunks)}")
             print("First chunk metadata:", txt_chunks[0].metadata)
             print("First chunk content:", txt_chunks[0].page_content)
         else:
             print("TXT processing failed.")
         # テスト用ファイルを削除
         # os.remove(test_file_txt)
    else:
         print(f"Test TXT file could not be created/found: {test_file_txt}")