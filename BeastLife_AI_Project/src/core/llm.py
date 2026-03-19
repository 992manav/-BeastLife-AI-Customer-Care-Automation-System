import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
import re
from src.core.config import get_settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


def _safe_parse_json(response_text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse model output into JSON even when wrapped in markdown/code fences."""
    try:
        return json.loads(response_text)
    except Exception:
        pass

    if not response_text:
        return fallback

    # Handle fenced code blocks or extra commentary around JSON.
    cleaned = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # Fallback to first JSON object found in response.
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    return fallback

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text response from prompt."""
        pass
    
    @abstractmethod
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text into one of the given categories."""
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        pass

class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
            settings = get_settings()
            self.genai.configure(api_key=settings.gemini_api_key)
            self.model_name = settings.gemini_model
            self.model = self.genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise
    
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text using Gemini."""
        prompt = f"""You are a customer-care intent classifier for BeastLife.

Choose exactly one category from this list:
{', '.join(categories)}

Input text:
{text}

Rules:
- Return only valid JSON.
- Do not include markdown, backticks, or explanation.
- confidence must be a float from 0.0 to 1.0.
- If uncertain, choose the closest category and reduce confidence.

Required JSON schema:
{{
    "category": "one category from the provided list",
    "confidence": 0.0,
    "intents": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.2)
            response_dict = _safe_parse_json(
                response_text,
                {"category": "unknown", "confidence": 0.0, "intents": []}
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error classifying text with Gemini: {e}")
            return {"category": "unknown", "confidence": 0.0, "intents": []}
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using Gemini."""
        prompt = f"""Extract customer-support entities from the text below.

Text:
{text}

Rules:
- Return only valid JSON.
- If a field is not present, return an empty list for that field.
- Keep original values without rewriting.

Required JSON schema:
{{
    "order_id": [],
    "payment_id": [],
    "product_name": [],
    "amount": [],
    "customer_issue": [],
    "phone": [],
    "email": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = _safe_parse_json(
                response_text,
                {
                    "order_id": [],
                    "payment_id": [],
                    "product_name": [],
                    "amount": [],
                    "customer_issue": [],
                    "phone": [],
                    "email": [],
                }
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error extracting entities with Gemini: {e}")
            return {
                "order_id": [],
                "payment_id": [],
                "product_name": [],
                "amount": [],
                "customer_issue": [],
                "phone": [],
                "email": [],
            }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Gemini."""
        prompt = f"""Analyze customer sentiment and urgency from this text.

Text:
{text}

Rules:
- Return only valid JSON.
- sentiment must be one of: positive, negative, neutral, critical.
- confidence must be 0.0 to 1.0.
- score must be -1.0 to 1.0.
- urgency must be one of: low, medium, high, critical.

Required JSON schema:
{{
    "sentiment": "neutral",
    "confidence": 0.0,
    "score": 0.0,
    "urgency": "medium"
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = _safe_parse_json(
                response_text,
                {
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "urgency": "medium",
                }
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error analyzing sentiment with Gemini: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "urgency": "medium",
            }

class GroqProvider(LLMProvider):
    """Groq API provider."""
    
    def __init__(self):
        try:
            from groq import Groq
            settings = get_settings()
            self.client = Groq(api_key=settings.groq_api_key)
            self.model_name = settings.groq_model
            logger.info(f"Initialized Groq provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq provider: {e}")
            raise
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Groq."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Groq: {e}")
            raise
    
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text using Groq."""
        prompt = f"""You are a customer-care intent classifier for BeastLife.

Choose exactly one category from this list:
{', '.join(categories)}

Input text:
{text}

Rules:
- Return only valid JSON.
- Do not include markdown, backticks, or explanation.
- confidence must be a float from 0.0 to 1.0.
- If uncertain, choose the closest category and reduce confidence.

Required JSON schema:
{{
    "category": "one category from the provided list",
    "confidence": 0.0,
    "intents": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.2)
            response_dict = _safe_parse_json(
                response_text,
                {"category": "unknown", "confidence": 0.0, "intents": []}
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error classifying text with Groq: {e}")
            return {"category": "unknown", "confidence": 0.0, "intents": []}
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using Groq."""
        prompt = f"""Extract customer-support entities from the text below.

Text:
{text}

Rules:
- Return only valid JSON.
- If a field is not present, return an empty list for that field.
- Keep original values without rewriting.

Required JSON schema:
{{
    "order_id": [],
    "payment_id": [],
    "product_name": [],
    "amount": [],
    "customer_issue": [],
    "phone": [],
    "email": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = _safe_parse_json(
                response_text,
                {
                    "order_id": [],
                    "payment_id": [],
                    "product_name": [],
                    "amount": [],
                    "customer_issue": [],
                    "phone": [],
                    "email": [],
                }
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error extracting entities with Groq: {e}")
            return {
                "order_id": [],
                "payment_id": [],
                "product_name": [],
                "amount": [],
                "customer_issue": [],
                "phone": [],
                "email": [],
            }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Groq."""
        prompt = f"""Analyze customer sentiment and urgency from this text.

Text:
{text}

Rules:
- Return only valid JSON.
- sentiment must be one of: positive, negative, neutral, critical.
- confidence must be 0.0 to 1.0.
- score must be -1.0 to 1.0.
- urgency must be one of: low, medium, high, critical.

Required JSON schema:
{{
    "sentiment": "neutral",
    "confidence": 0.0,
    "score": 0.0,
    "urgency": "medium"
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = _safe_parse_json(
                response_text,
                {
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "urgency": "medium",
                }
            )
            return response_dict
        except Exception as e:
            logger.error(f"Error analyzing sentiment with Groq: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "urgency": "medium",
            }

def get_llm_provider() -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    settings = get_settings()
    
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    elif settings.llm_provider == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
