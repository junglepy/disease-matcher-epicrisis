"""
LLM-based query generator для генерации множественных английских запросов из русских симптомов
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .config import Settings

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Генератор поисковых запросов с помощью LLM"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.3  # Немного креативности для вариативности
        
        # Загружаем промпт
        prompt_path = Path(__file__).parent.parent / "prompts" / "query_generation.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Минимальный промпт для генерации запросов"""
        return """Generate 3-5 English medical term variations for a Russian symptom.
Include:
- Direct translation
- Medical synonyms
- Related clinical terms
- Alternative descriptions

You must return a JSON object in this format: {"queries": ["term1", "term2", "term3", "term4", "term5"]}

Example:
Russian: мозжечковая атаксия
JSON response: {"queries": ["cerebellar ataxia", "ataxia", "gait instability", "coordination disorder", "cerebellar dysfunction"]}"""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate_queries(self, symptom: str) -> List[str]:
        """
        Генерировать множественные английские запросы для русского симптома
        
        Args:
            symptom: Симптом на русском языке
            
        Returns:
            Список английских вариантов для поиска
        """
        try:
            # Вызов OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Russian symptom: {symptom}"}
                ],
                temperature=self.temperature,
                max_tokens=200,  # Не нужно много токенов для списка терминов
                response_format={"type": "json_object"}  # Принудительный JSON
            )
            
            # Парсим ответ
            content = response.choices[0].message.content
            logger.debug(f"LLM response for '{symptom}': {content}")
            
            data = json.loads(content)
            queries = data.get("queries", [])
            
            # Валидация и очистка
            queries = [q.strip() for q in queries if q and q.strip()]
            
            # Если не получили запросов, возвращаем оригинал
            if not queries:
                logger.warning(f"No queries generated for '{symptom}', returning original")
                return [symptom]
                
            logger.info(f"Generated {len(queries)} queries for '{symptom}': {queries}")
            return queries[:5]  # Максимум 5 запросов
            
        except Exception as e:
            logger.error(f"Error generating queries for '{symptom}': {e}")
            # Fallback на простой перевод из словаря
            return self._fallback_translation(symptom)
            
    def _fallback_translation(self, symptom: str) -> List[str]:
        """Fallback перевод через словарь если LLM недоступен"""
        # Импортируем переводчик для fallback
        try:
            from .symptom_translator import SymptomTranslator
            translator = SymptomTranslator()
            translated = translator.translate_symptom(symptom)
            if translated != symptom:
                return [translated]
        except Exception:
            pass
            
        # Возвращаем оригинал если ничего не помогло
        return [symptom]
        
    async def generate_batch_queries(self, symptoms: List[str]) -> Dict[str, List[str]]:
        """
        Генерировать запросы для списка симптомов
        
        Args:
            symptoms: Список симптомов на русском языке
            
        Returns:
            Словарь {симптом: [запросы]}
        """
        results = {}
        
        # TODO: Можно оптимизировать через батчинг в будущем
        for symptom in symptoms:
            queries = await self.generate_queries(symptom)
            results[symptom] = queries
            
        return results