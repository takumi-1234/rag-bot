# src/rag/llm_gemini.py
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
import time
# typing と langchain.docstore.document をインポート
from typing import List, Dict, Any
from langchain_core.documents import Document # langchain-core からインポート推奨

# --- 環境変数の読み込み ---
load_dotenv()

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiChat:
    """
    Google Gemini API との連携を行うクラス。
    genai.configure() と genai.GenerativeModel() を使用します。
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        """
        Google Gemini API キーを設定し、指定されたモデルの GenerativeModel オブジェクトを初期化します。
        APIへの接続確認も行います。

        Args:
            api_key (str | None): Google Gemini API キー。Noneの場合は環境変数 "GEMINI_API_KEY" を使用。
            model_name (str | None): 使用するモデル名 (例: "gemini-1.5-flash-latest")。
                                     Noneの場合は環境変数 "GEMINI_MODEL_NAME" を使用。
                                     環境変数もない場合のデフォルトは "gemini-1.0-pro"。
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        default_model = "gemini-1.0-pro" # デフォルトモデル名
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", default_model)
        self.model: genai.GenerativeModel | None = None # モデルオブジェクトを保持

        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set.")
            raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")

        logger.info(f"Initializing GeminiChat with target model: {self.model_name}")

        try:
            # --- APIキーの設定 ---
            logger.info("Configuring Gemini API key...")
            genai.configure(api_key=self.api_key)

            # --- API疎通確認 (利用可能なモデルリストを取得) ---
            logger.info("Attempting to list available models to verify API connection...")
            available_models: List[Dict[str, Any]] = []
            for model_info in genai.list_models():
                 if 'generateContent' in model_info.supported_generation_methods:
                      available_models.append({
                           "name": model_info.name,
                           "description": model_info.description,
                           "version": model_info.version,
                      })

            if not available_models:
                 logger.error("No models supporting 'generateContent' found. Please check your API key and project settings.")
                 raise RuntimeError("Could not find any available models supporting 'generateContent'.")

            available_model_names = [m['name'] for m in available_models]
            logger.info(f"Successfully connected to Gemini API. Found {len(available_model_names)} models supporting generateContent.")
            # logger.debug(f"Available models supporting generateContent: {available_model_names}")

            # --- 使用予定モデルの存在確認 ---
            self.model_id_for_api = f"models/{self.model_name}" # 確認用
            if self.model_id_for_api not in available_model_names:
                logger.warning(f"Specified model '{self.model_id_for_api}' was not found in the list of models supporting 'generateContent'.")
                logger.warning(f"Available models supporting generateContent: {available_model_names}")
                logger.warning(f"Proceeding with initializing GenerativeModel for '{self.model_name}', but it might fail later during API call.")
            else:
                 logger.info(f"Target model '{self.model_id_for_api}' is available and supports 'generateContent'.")

            # --- GenerativeModel オブジェクトの初期化 ---
            logger.info(f"Initializing GenerativeModel for model: {self.model_name}...")
            # システム指示: RAGの応答に特化した指示
            system_instruction = """以下の「参考資料」に基づいて、「質問」に日本語で回答してください。
回答は「参考資料」の内容のみを根拠とし、推測や参考資料外の情報は含めないでください。
もし「参考資料」の中に回答に該当する情報が見つからない場合は、その旨を明確に述べ、「参考資料には関連する情報が見つかりませんでした。」と回答してください。
回答は自然な日本語で、簡潔かつ正確に記述してください。"""

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            logger.info(f"GenerativeModel for '{self.model_name}' initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to configure Gemini API or initialize GenerativeModel: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Gemini components: {e}") from e

    # ★★★ 型ヒントを修正 ★★★
    def _create_prompt_content(self, query: str, context_docs: List[Document]) -> List[str]:
        """
        LLMに渡すためのプロンプトコンテンツを作成します。
        システム指示はモデル初期化時に設定するため、ここではユーザーの質問とコンテキストのみを扱います。

        Args:
            query (str): ユーザーからの質問。
            context_docs (List[Document]): LangChainのDocumentオブジェクトのリスト。

        Returns:
            List[str]: generate_content に渡す contents のリスト (文字列要素のみ)。
        """
        # コンテキスト情報を整形
        context_text = "\n\n".join([
            f"--- 資料「{doc.metadata.get('source', '不明')}」より ---\n{doc.page_content}"
            for doc in context_docs if hasattr(doc, 'page_content') and hasattr(doc, 'metadata')
        ])

        # ユーザーの質問とコンテキストを組み合わせる
        # システム指示で役割や応答形式は定義済み
        full_prompt = f"""【参考資料】
{context_text if context_text else "（利用可能な参考資料はありません）"}

---
【質問】
{query}
"""
        # 単一のプロンプト文字列をリストに入れて返す
        return [full_prompt]

    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        ユーザーの質問と検索されたコンテキスト文書に基づき、Gemini API を呼び出して回答を生成します。
        self.model オブジェクトを使用します。
        """
        if not self.model:
             logger.error("Gemini model is not initialized.")
             return "エラー: Geminiモデルが初期化されていません。"
        if not query:
            logger.warning("Empty query received.")
            return "質問を入力してください。"

        # プロンプトコンテンツ作成
        contents = self._create_prompt_content(query, context_docs)
        prompt_length = sum(len(part) for part in contents) # 文字列のみなのでこれでOK
        logger.info(f"Sending content to Gemini (approx length: {prompt_length} chars, model: {self.model_name})")
        # logger.debug(f"Content (first 100 chars): {str(contents)[:100]}...")

        # API呼び出しパラメータ
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            # max_output_tokens=8192, # 必要に応じて設定
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # リトライ設定
        max_retries = 3
        initial_retry_delay = 2

        for attempt in range(max_retries):
            retry_delay = initial_retry_delay * (2 ** attempt)
            try:
                # --- API呼び出し (self.model を使用) ---
                logger.info(f"Calling Gemini API (Attempt {attempt + 1}/{max_retries})...")
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # --- レスポンス処理 ---
                generated_text = ""
                if hasattr(response, 'text') and response.text:
                     generated_text = response.text
                     logger.info("Successfully generated response text from Gemini.")
                elif hasattr(response, 'parts') and response.parts:
                     generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                     logger.info("Successfully generated response parts from Gemini.")
                elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                     block_message = f"回答生成がブロックされました。理由: {block_reason}"
                     safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                     logger.warning(f"{block_message}. Safety ratings: {safety_ratings_str}")
                     return block_message
                else:
                     logger.warning(f"Gemini response received, but no text or block reason found. Response object: {response}")
                     generated_text = "回答のテキスト部分が見つかりませんでした。"

                return generated_text.strip()

            # --- 例外処理 ---
            except Exception as e:
                logger.error(f"Error during Gemini API call (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                should_retry = False
                status_code = None
                if hasattr(e, 'grpc_status_code'): # gRPCエラー
                     status_code = e.grpc_status_code
                     if status_code in [429, 500, 503]:
                          should_retry = True
                elif isinstance(e, (genai.types.InternalServerError, genai.types.DeadlineExceededError, genai.types.ResourceExhaustedError)):
                     should_retry = True

                if should_retry and attempt < max_retries - 1:
                    logger.warning(f"Retrying after {retry_delay:.1f} seconds... (Status code: {status_code or 'N/A'})")
                    time.sleep(retry_delay)
                    continue
                else:
                    error_type_name = type(e).__name__
                    user_message = f"エラーが発生しました。回答を生成できませんでした。({error_type_name})"
                    if isinstance(e, genai.types.generation_types.BlockedPromptException):
                         user_message = f"回答生成がブロックされました: 安全性設定に抵触した可能性があります。"
                    elif isinstance(e, genai.types.generation_types.StopCandidateException):
                         user_message = "回答の生成が途中で停止しました。"
                         try:
                              if hasattr(e, 'response'):
                                   partial_text = e.response.text or "".join(part.text for part in e.response.parts)
                                   if partial_text:
                                        user_message += f"\n部分的な回答: {partial_text.strip()}"
                         except Exception: pass
                    elif isinstance(e, genai.types.NotFoundError):
                          user_message = f"エラー: 指定されたモデル '{self.model_name}' のAPI呼び出しに失敗しました。モデル名やAPI設定を確認してください。"
                    elif isinstance(e, genai.types.PermissionDeniedError):
                          user_message = f"エラー: APIキーが無効か、APIへのアクセス権限がありません。"

                    logger.error(f"Gemini API call failed after {attempt + 1} attempts or due to non-retriable error. Error: {e}")
                    return user_message

        logger.error("Reached end of generate_response function without returning a response or error.")
        return "予期せぬエラーが発生しました。回答を生成できませんでした。"

# --- 直接実行時のテスト用コード (オプション) ---
if __name__ == "__main__":
    logger.info("Running llm_gemini.py directly for testing...")
    try:
        gemini = GeminiChat() # 環境変数から設定を読み込む

        # テスト用のLangChain Documentオブジェクトを作成
        test_docs = [
            Document(page_content="これはテスト用の参考資料1です。AIは人工知能です。", metadata={"source": "test1.txt"}),
            Document(page_content="参考資料2：AIには様々な種類があります。", metadata={"source": "test2.pdf"}),
        ]

        test_query = "AIについて参考資料から教えてください。"
        print(f"\nTesting generate_response with query: '{test_query}'")
        response_text = llm.generate_response(query=query, context_docs=search_results)
        print("\n--- Gemini Response ---")
        print(response_text)
        print("-----------------------")

        # コンテキストなしの場合のテスト
        test_query_no_context = "日本の首都はどこですか？"
        print(f"\nTesting generate_response with query (no context): '{test_query_no_context}'")
        response_text_no_context = gemini.generate_response(test_query_no_context, [])
        print("\n--- Gemini Response (No Context) ---")
        print(response_text_no_context) # "参考資料には関連する情報が見つかりませんでした。" と返るはず
        print("------------------------------------")

    except Exception as main_e:
        print(f"\nError during direct execution test: {main_e}")
        logger.error(f"Error during direct execution test: {main_e}", exc_info=True)