# src/rag/llm_gemini.py
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain.docstore.document import Document

# .env ファイルから環境変数を読み込む
load_dotenv()

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiChat:
    """Google Gemini API との対話を管理するクラス (プロンプト再調整版)"""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        """
        Geminiクライアントを初期化し、API接続を確認します。
        (この部分は変更ありません)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
        self.target_model_name = f"models/{self.model_name}"

        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set.")
            raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")

        self.model: Optional[genai.GenerativeModel] = None

        try:
            logger.info(f"Initializing GeminiChat with target model: {self.model_name}")
            logger.info("Configuring Gemini API key...")
            genai.configure(api_key=self.api_key)

            logger.info("Attempting to list available models to verify API connection...")
            available_models = []
            model_found = False
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                        if m.name == self.target_model_name:
                            model_found = True
                logger.info(f"Successfully connected to Gemini API. Found {len(available_models)} models supporting generateContent.")

                if not model_found:
                    logger.warning(f"Target model '{self.target_model_name}' not found or does not support 'generateContent'. Available models: {available_models}")
                    raise ValueError(f"Target model '{self.model_name}' ('{self.target_model_name}') is not available or suitable.")

                logger.info(f"Target model '{self.target_model_name}' is available and supports 'generateContent'.")

            except Exception as e_list_models:
                 logger.error(f"Failed to list Gemini models. Check API key and network connection: {e_list_models}", exc_info=True)
                 raise RuntimeError("Failed to connect to Gemini API to verify models.") from e_list_models

            logger.info(f"Initializing GenerativeModel for model: {self.model_name} ({self.target_model_name})...")
            self.model = genai.GenerativeModel(self.target_model_name)
            logger.info(f"GenerativeModel for '{self.model_name}' initialized successfully.")

        except ValueError as ve:
            logger.error(f"Gemini configuration error: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"Failed to configure or initialize Gemini model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Gemini model: {e}") from e

    def _create_prompt_string(self, query: str, context_docs: Optional[List[Document]]) -> str:
        """
        最終的なプロンプト文字列を作成します (再調整版)。

        Args:
            query (str): ユーザーからの質問。
            context_docs (Optional[List[Document]]): 検索された関連ドキュメントのリスト。

        Returns:
            str: Gemini API に渡す最終的なプロンプト文字列。
        """
        # --- 再調整されたシステムプロンプト ---
        system_prompt = """あなたはAPU（立命館アジア太平洋大学）の入学案内に関する質問に答える専門アシスタントです。
提供された【参考資料】**のみ**に基づいて、以下の指示に従って日本語で回答してください。

**指示:**
1.  **正確性:** 資料に書かれている情報に**忠実に**回答してください。資料にない情報やあなたの推測を含めないでください。
2.  **網羅性:** 質問に関連すると思われる情報は、【参考資料】から可能な限り見つけ出し、回答に含めてください。資料を注意深く確認してください。
3.  **具体性:** 制度や手続きについて尋ねられた場合は、資料に記載されている範囲で、条件、対象、手順などを具体的に説明してください。
4.  **情報不足:** 資料を確認しても質問に答えられる情報が見つからない場合は、「提供された資料には、〇〇に関する情報は見当たりませんでした。」のように明確に回答してください。
5.  **比較・リスト:** 質問が比較を求めている場合は相違点を、リストアップを求めている場合は該当項目を、資料に基づいて回答してください。
6.  **出典:** 回答の根拠となった【参考資料】の出典（ファイル名と、可能であればページ番号）を、回答の最後に `Sources:` として示してください。（例: `Sources: sougou_handbook_E.pdf (p.5), sougou_application_Handbok_E_until37.pdf (p.10)`）
7.  **回答形式:** 自然で分かりやすい日本語で回答してください。箇条書きを使うと分かりやすい場合は適切に使用してください。"""

        context_header = "【参考資料】"
        query_header = "【質問】"
        # 回答指示ヘッダーをシンプルに
        answer_instruction_header = "【回答】"

        prompt_parts = [system_prompt + "\n\n"]

        # --- コンテキストドキュメントの処理 (メタデータ活用を意識) ---
        # (この部分は前回の提案1で修正したコードを維持)
        context_text = ""
        if isinstance(context_docs, list) and context_docs:
            valid_doc_contents = []
            logger.info(f"Processing {len(context_docs)} context documents for prompt...")
            for i, doc in enumerate(context_docs):
                if doc and hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                    source_info = "不明"
                    page_info = ""
                    # メタデータからファイル名とページ番号を取得
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                        source = doc.metadata.get('source')
                        page = doc.metadata.get('page') # PyPDFLoaderなどが付与する可能性
                        if source:
                            source_info = source
                        if page is not None: # ページ番号は0の場合もあるので is not None でチェック
                            # ページ番号はintで来ることが多いので+1して表示（人間が読みやすいように）
                            try:
                                page_info = f" (p.{int(page) + 1})"
                            except (ValueError, TypeError):
                                page_info = f" ({page})" # intに変換できない場合はそのまま表示

                    # 各資料を識別しやすいようにフォーマット（出典とページ番号を追加）
                    valid_doc_contents.append(
                        f"--- 資料 {i+1} (出典: {source_info}{page_info}) ---\n{doc.page_content}\n---"
                    )
                else:
                    # 無効なドキュメントや内容が空の場合のログ
                    page_content_info = type(doc.page_content).__name__ if (doc and hasattr(doc, 'page_content')) else 'N/A'
                    logger.warning(f"Skipping invalid document or document with empty/invalid page_content (type: {page_content_info}) at index {i}.")

            if valid_doc_contents:
                context_text = "\n\n".join(valid_doc_contents)
                logger.info(f"Generated context text from {len(valid_doc_contents)} valid documents.")
            else:
                 logger.info("No valid context documents found after filtering for prompt.")

        # --- プロンプトの組み立て ---
        if context_text:
            prompt_parts.append(context_header + "\n" + context_text + "\n\n")
        else:
            # コンテキストがない場合も明示
            prompt_parts.append("【参考資料】\n参考資料はありません。\n\n")

        prompt_parts.append(query_header + "\n" + query + "\n\n")
        prompt_parts.append(answer_instruction_header) # モデルに最終回答を促す

        final_prompt = "".join(prompt_parts)
        logger.debug(f"Final prompt string length: {len(final_prompt)} chars")
        return final_prompt

    def generate_response(self, query: str, context_docs: Optional[List[Document]]) -> str:
        """
        クエリとコンテキストドキュメントに基づいて、Gemini API を呼び出し、回答を生成します。
        (この部分は変更ありません)
        """
        if not query or not query.strip():
            logger.warning("Received empty query.")
            raise ValueError("質問内容を入力してください。")

        if not self.model:
            logger.error("Gemini model is not initialized. Cannot generate response.")
            raise RuntimeError("言語モデルが準備できていません。システム管理者に連絡してください。")

        try:
            prompt_string = self._create_prompt_string(query, context_docs)
        except Exception as e_prompt:
            logger.error(f"Error creating prompt string: {e_prompt}", exc_info=True)
            raise RuntimeError(f"回答生成のためのプロンプト作成中に問題が発生しました: {e_prompt}") from e_prompt

        try:
            logger.info(f"Sending request to Gemini API (model: {self.target_model_name})...")
            response = self.model.generate_content(
                prompt_string,
                # 必要に応じて generation_config や safety_settings を設定
            )

            if response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                if generated_text.strip():
                    logger.info("Successfully generated response from Gemini.")
                    # 回答の最後に自動でSourcesが付与されない場合、ここで付与する処理を追加することも検討可能
                    # (ただし、プロンプトで指示しているので基本的には不要なはず)
                    return generated_text.strip()
                else:
                    logger.warning("Gemini response parts were empty or contained no text.")
                    return "回答が得られませんでしたが、エラーはありませんでした。質問を変えてみてください。"

            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                safety_ratings_info = str(response.prompt_feedback.safety_ratings) if hasattr(response.prompt_feedback, 'safety_ratings') else "N/A"
                logger.warning(f"Gemini response blocked. Reason: {block_reason}, SafetyRatings: {safety_ratings_info}")
                raise RuntimeError(f"回答を生成できませんでした。コンテンツが安全でないと判断された可能性があります (理由: {block_reason})。")

            elif not response.candidates:
                 prompt_feedback_info = str(response.prompt_feedback) if hasattr(response, 'prompt_feedback') else 'N/A'
                 finish_reason_info = "N/A"
                 if hasattr(response, 'candidates') and response.candidates:
                      candidate = response.candidates[0]
                      if hasattr(candidate, 'finish_reason'):
                           finish_reason_info = candidate.finish_reason.name
                 logger.warning(f"Gemini response has no candidates. Finish Reason: {finish_reason_info}, Prompt Feedback: {prompt_feedback_info}")
                 raise RuntimeError(f"回答を生成できませんでした。モデルが有効な応答を返しませんでした (Finish Reason: {finish_reason_info})。")

            else:
                 response_summary = str(response)[:200]
                 logger.warning(f"Gemini response format unexpected. Response summary: {response_summary}")
                 raise RuntimeError("回答を取得できませんでした (予期せぬレスポンス形式)。")

        # API呼び出し中の google.api_core.exceptions などはそのまま上位に伝播させる
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            raise e # 元の例外を送出