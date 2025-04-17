# src/rag/document_processor.py
import os
import logging
from typing import List, Dict, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- ファイルローダー (docx2txt を使用) ---
try:
    # langchain_community からインポート
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
except ImportError:
    print("Required loaders (PyPDFLoader, TextLoader, Docx2txtLoader) not found.")
    print("Please ensure 'langchain-community', 'pypdf', 'docx2txt' are installed.")
    PyPDFLoader = TextLoader = Docx2txtLoader = None

logger = logging.getLogger(__name__)

# --- 定数 ---
# サポートするファイル拡張子と対応するローダーのマッピング
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader, # Docx2txtLoader を使用
    ".txt": TextLoader,
}

# テキスト分割の設定 (変更なし)
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", "。", "、", " ", ""],
    "length_function": len,
    "add_start_index": True,
}

# --- 関数 ---

def load_document(file_path: str) -> List[Document]:
    """ファイルパスから適切なローダーを使ってドキュメントを読み込む"""
    file_ext = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_EXTENSIONS.get(file_ext)

    if loader_class is None:
        # サポートされていない拡張子の場合
        if file_ext:
            logger.warning(f"Unsupported file type: '{file_ext}' for file {os.path.basename(file_path)}. Skipping.")
        else:
            logger.warning(f"File has no extension: {os.path.basename(file_path)}. Skipping.")
        return []

    if PyPDFLoader is None or TextLoader is None or Docx2txtLoader is None: # 依存ライブラリ不足の場合
        logger.error(f"Loader for '{file_ext}' is not available due to missing dependencies. Check installation.")
        return []

    try:
        logger.info(f"Loading document: {os.path.basename(file_path)} using {loader_class.__name__}")
        # encoding を指定できるローダーの場合は指定する
        if loader_class == TextLoader:
            # UTF-8 で試みる。他のエンコーディングが必要な場合は試行錯誤が必要
            try:
                loader = loader_class(file_path, encoding='utf-8')
                documents = loader.load()
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode {os.path.basename(file_path)} with UTF-8. Trying 'cp932' (Shift-JIS).")
                try:
                    loader = loader_class(file_path, encoding='cp932')
                    documents = loader.load()
                except Exception as e_enc:
                    logger.error(f"Failed to load {os.path.basename(file_path)} with multiple encodings: {e_enc}", exc_info=True)
                    return []
            except Exception as e_load: # その他の TextLoader エラー
                 logger.error(f"Error loading text file {os.path.basename(file_path)}: {e_load}", exc_info=True)
                 return []
        else:
             # PyPDFLoader, Docx2txtLoader など
             loader = loader_class(file_path)
             documents = loader.load()

        # 読み込んだドキュメントがリスト形式であることを確認
        if not isinstance(documents, list):
            logger.warning(f"Loader {loader_class.__name__} did not return a list for file {os.path.basename(file_path)}. Got {type(documents)}. Skipping.")
            return []

        # Document オブジェクトでない要素を除外 (念のため)
        valid_documents = [doc for doc in documents if isinstance(doc, Document)]
        if len(valid_documents) != len(documents):
            logger.warning(f"Some elements loaded from {os.path.basename(file_path)} were not Document objects.")

        if not valid_documents:
             logger.warning(f"No valid Document objects were loaded from {os.path.basename(file_path)}.")
             return []

        logger.info(f"Successfully loaded {len(valid_documents)} document page(s)/section(s) from {os.path.basename(file_path)}")
        return valid_documents
    except FileNotFoundError:
        logger.error(f"File not found during loading: {file_path}")
        return []
    except Exception as e:
        # ローダー固有のエラー（例: PDFが破損している、DOCXがパスワード保護されているなど）
        logger.error(f"Error loading document {os.path.basename(file_path)} with {loader_class.__name__}: {e}", exc_info=True)
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """ドキュメントを指定された設定でチャンクに分割する"""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document page(s)/section(s) into {len(split_docs)} chunks.")
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
    logger.info(f"Processing document: {os.path.basename(file_path)}")

    # 1. ドキュメントをロード
    loaded_docs = load_document(file_path)
    if not loaded_docs:
        # load_document 内で既にログが出ているはず
        logger.warning(f"No documents were loaded from {os.path.basename(file_path)}. Processing stopped.")
        return []

    # 2. ドキュメントをチャンクに分割
    split_docs = split_documents(loaded_docs)
    if not split_docs:
        logger.warning(f"Failed to split documents from {os.path.basename(file_path)} into chunks. Processing stopped.")
        return []

    # 3. 各チャンクにソースファイル名のメタデータを付与
    source_filename = os.path.basename(file_path)
    for doc in split_docs:
        # doc.metadata が存在しない場合や辞書でない場合に初期化
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
             doc.metadata = {}
        # 既存の 'source' を上書きしないように注意（ローダーが設定する場合もある）
        # ここではファイル名を確実に設定するため上書きする
        doc.metadata["source"] = source_filename
        # page メタデータなどがローダーによって追加されているか確認
        # logger.debug(f"Chunk metadata: {doc.metadata}")

    logger.info(f"Finished processing {source_filename}. Generated {len(split_docs)} chunks.")
    return split_docs

# --- 直接実行時のテスト用コード (変更なし) ---
if __name__ == "__main__":
    # このファイルが直接実行された場合のテストコード
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    test_dir = "test_docs"
    os.makedirs(test_dir, exist_ok=True)
    test_file_pdf = os.path.join(test_dir, "example.pdf")
    test_file_txt = os.path.join(test_dir, "example.txt")
    test_file_docx = os.path.join(test_dir, "example.docx")

    # --- テスト用ファイル作成 ---
    # PDF (pypdfが必要。ここではダミーテキストで代替)
    # TXT
    try:
        with open(test_file_txt, "w", encoding="utf-8") as f:
            f.write("これはテスト用のテキストファイルです。\n")
            f.write("複数行のテスト。\n\n")
            f.write("これは第1段落です。句点で区切ります。改行も入れます。\n")
            f.write("これは第2段落です。長い文を書いて、チャンク分割がどのように行われるか確認します。" * 5)
    except Exception as e:
        print(f"Error creating test TXT file: {e}")

    # DOCX (python-docxが必要。ここではダミーテキストで代替)
    # 実際のテストでは、pypdf や python-docx / docx2txt を使ってサンプルファイルを作成・配置してください。
    print(f"Please place sample files for testing:")
    print(f"- PDF: {test_file_pdf}")
    print(f"- DOCX: {test_file_docx}")
    print(f"Test TXT file created at: {test_file_txt}")


    # --- PDFファイルのテスト ---
    if os.path.exists(test_file_pdf):
        print(f"\n--- Testing PDF Processing ({os.path.basename(test_file_pdf)}) ---")
        pdf_chunks = process_documents(test_file_pdf)
        if pdf_chunks:
            print(f"Number of chunks generated from PDF: {len(pdf_chunks)}")
            print("First chunk metadata:", pdf_chunks[0].metadata)
            print("First chunk content (first 100 chars):", pdf_chunks[0].page_content[:100] + "...")
        else:
            print("PDF processing failed or returned no chunks.")
    else:
        print(f"\nTest PDF file not found: {test_file_pdf}")

    # --- TXTファイルのテスト ---
    if os.path.exists(test_file_txt):
         print(f"\n--- Testing TXT Processing ({os.path.basename(test_file_txt)}) ---")
         txt_chunks = process_documents(test_file_txt)
         if txt_chunks:
             print(f"Number of chunks generated from TXT: {len(txt_chunks)}")
             print("First chunk metadata:", txt_chunks[0].metadata)
             print("First chunk content (first 100 chars):", txt_chunks[0].page_content[:100] + "...")
         else:
             print("TXT processing failed or returned no chunks.")
    else:
         print(f"\nTest TXT file not found: {test_file_txt}")

    # --- DOCXファイルのテスト ---
    if os.path.exists(test_file_docx):
        print(f"\n--- Testing DOCX Processing ({os.path.basename(test_file_docx)}) ---")
        docx_chunks = process_documents(test_file_docx)
        if docx_chunks:
            print(f"Number of chunks generated from DOCX: {len(docx_chunks)}")
            print("First chunk metadata:", docx_chunks[0].metadata)
            print("First chunk content (first 100 chars):", docx_chunks[0].page_content[:100] + "...")
        else:
            print("DOCX processing failed or returned no chunks.")
    else:
        print(f"\nTest DOCX file not found: {test_file_docx}")

    # テスト用ディレクトリの後片付け（オプション）
    # import shutil
    # shutil.rmtree(test_dir)
    # print(f"\nRemoved test directory: {test_dir}")