from typing import List
from openai import OpenAI
from core.config import LLAMA_SERVER_URL
from core.logger import get_logger
from core.llm_runtime import detect_server_model
from core.prompt_templates import build_chat_prompt_from_messages, get_stop_sequences_for_model

logger = get_logger(__name__)

class ContextDistiller:
    """Compresses noisy retrieved memories into a dense summary before injection."""
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.model_name = detect_server_model(self.client)
        
    def distill(self, query: str, context_items: List[str]) -> str:
        if not context_items:
            return ""
            
        context_block = "\n---\n".join(context_items)
        
        system_prompt = (
            "You are an internal memory distillation module for an AI. "
            "Your job is to read the raw retrieved memories/documents and the user's current query, "
            "and synthesize ONLY the facts from the memories that are directly relevant to answering the query. "
            "Condense them into a dense, concise set of bullet points. "
            "DO NOT answer the user's query. DO NOT hallucinate. DO NOT explain what you are doing. "
            "ONLY output the distilled facts. If none of the memories are relevant to the query at all, output 'NO_RELEVANT_CONTEXT'."
        )
        
        user_prompt = f"User Query: {query}\n\nRaw Memories:\n{context_block}"
        
        try:
            prompt = build_chat_prompt_from_messages(
                self.model_name,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=400,
                stop=get_stop_sequences_for_model(self.model_name),
            )
            
            distilled = response.choices[0].text.strip()
            if "NO_RELEVANT_CONTEXT" in distilled.upper():
                logger.info("[DISTILL] Memories were determined to be irrelevant to the query.")
                return ""
                
            logger.info(f"[DISTILL] Compressed {len(context_items)} items into {len(distilled)} characters.")
            return distilled
            
        except Exception as e:
            logger.error(f"[DISTILL] Distillation failed: {e}")
            return "\n".join(context_items) # fallback to raw
