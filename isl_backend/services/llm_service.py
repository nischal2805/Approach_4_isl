"""
ISL Backend - LLM Service for Grammar Correction

Uses Ollama to convert ISL gloss (word sequence) into grammatically correct English.
"""
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"  # Fast, small model. Change to llama3.2:3b for better quality


class LLMService:
    """Service for grammar correction using Ollama."""
    
    def __init__(self):
        self._available = False
        self._model = MODEL_NAME
        
    async def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    # Check if our model or a variant is available
                    self._available = any(self._model.split(":")[0] in name for name in model_names)
                    if self._available:
                        logger.info(f"Ollama available with model: {self._model}")
                    else:
                        logger.warning(f"Ollama running but {self._model} not found. Available: {model_names}")
                    return self._available
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
        return False
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    async def correct_grammar(self, gloss: str) -> str:
        """
        Convert ISL gloss to grammatically correct English.
        
        ISL gloss is typically in SOV order with minimal grammar markers.
        Example: "I SCHOOL GO" -> "I am going to school"
        """
        if not gloss or not gloss.strip():
            return ""
        
        # Filter out "no_sign" from the gloss
        words = [w for w in gloss.split() if w.lower() != "no_sign"]
        if not words:
            return "No signs detected"
        
        clean_gloss = " ".join(words)
        
        # If Ollama not available, return the gloss as-is
        if not self._available:
            return clean_gloss.lower().capitalize()
        
        prompt = f"""Convert this Indian Sign Language gloss into a natural, grammatically correct English sentence.

ISL Gloss: {clean_gloss}

Rules:
- ISL uses SOV (Subject-Object-Verb) order, convert to English SVO
- Add appropriate articles (a, an, the), prepositions, and verb tenses
- Keep it simple and natural
- Only output the corrected sentence, nothing else

English:"""

        try:
            logger.info(f"Calling Ollama with gloss: {clean_gloss}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 50
                        }
                    }
                )
                
                logger.info(f"Ollama response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Ollama response data: {data}")
                    result = data.get("response", "").strip()
                    # Clean up the response
                    result = result.split("\n")[0].strip()  # Take first line only
                    if result:
                        logger.info(f"LLM result: {result}")
                        return result
                    else:
                        logger.warning("Ollama returned empty response")
                else:
                    logger.warning(f"Ollama returned status {response.status_code}: {response.text}")
                        
        except Exception as e:
            import traceback
            logger.error(f"LLM error: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
        
        # Fallback: return clean gloss with basic formatting
        return clean_gloss.lower().capitalize()


# Singleton
_llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
