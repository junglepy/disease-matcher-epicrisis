import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import HPOCandidate
from .config import Settings

logger = logging.getLogger(__name__)


class HPOSelector:
    """LLM-based селектор для выбора наиболее подходящего HPO кода"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.openai_model
        self.temperature = 0.0  # Для консистентности
        
        # Проверяем наличие API ключа
        self.has_openai = settings.openai_api_key and settings.openai_api_key != "your-api-key-here"
        if self.has_openai:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured, using fallback HPO selector")
        
        # Загружаем промпт
        prompt_path = Path(__file__).parent.parent / "prompts" / "hpo_selection.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Минимальный промпт для выбора HPO"""
        return """Select best HPO term for symptom.
Consider exact match, clinical relevance, specificity.
Return only HPO ID, nothing else."""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def select_best_hpo(self, symptom: str, candidates: List[HPOCandidate]) -> Optional[str]:
        """
        Выбрать лучший HPO код для симптома
        
        Args:
            symptom: Симптом на русском языке
            candidates: Список кандидатов HPO
            
        Returns:
            HPO ID лучшего кандидата или None
        """
        if not candidates:
            logger.warning(f"No HPO candidates for symptom: {symptom}")
            return None
            
        # Если только один кандидат - возвращаем его
        if len(candidates) == 1:
            return candidates[0].hpo_id
        
        # Если нет OpenAI API key, используем простой fallback
        if not self.has_openai:
            # Возвращаем кандидата с наивысшим score
            best_candidate = max(candidates, key=lambda c: c.score)
            logger.debug(f"Fallback selector: {best_candidate.hpo_id} for '{symptom}' (score: {best_candidate.score})")
            return best_candidate.hpo_id
            
        try:
            # Формируем список кандидатов для промпта
            candidates_text = self._format_candidates(candidates)
            
            user_prompt = f"""Symptom: {symptom}

HPO candidates:
{candidates_text}

Select the most appropriate HPO ID."""
            
            # Вызов OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=50  # Нужен только ID
            )
            
            # Парсим ответ
            content = response.choices[0].message.content.strip()
            
            # Извлекаем HPO ID из ответа
            hpo_id = self._extract_hpo_id(content, candidates)
            
            if hpo_id:
                logger.debug(f"Selected {hpo_id} for symptom: {symptom}")
            else:
                logger.warning(f"Failed to extract valid HPO ID from: {content}")
                
            return hpo_id
            
        except Exception as e:
            logger.error(f"Error selecting HPO for symptom '{symptom}': {e}")
            # Fallback - возвращаем кандидата с наивысшим score
            return candidates[0].hpo_id if candidates else None
            
    def _format_candidates(self, candidates: List[HPOCandidate]) -> str:
        """Форматирование кандидатов для промпта"""
        lines = []
        for i, candidate in enumerate(candidates[:5], 1):  # Максимум 5 кандидатов
            # Включаем основное название и первые 2 синонима
            synonyms = candidate.synonyms[:2] if candidate.synonyms else []
            synonyms_str = f" (also: {', '.join(synonyms)})" if synonyms else ""
            
            lines.append(f"{i}. {candidate.hpo_id}: {candidate.name}{synonyms_str}")
            
        return "\n".join(lines)
        
    def _extract_hpo_id(self, response: str, candidates: List[HPOCandidate]) -> Optional[str]:
        """Извлечь HPO ID из ответа LLM"""
        # Ищем паттерн HP:XXXXXXX
        import re
        match = re.search(r'HP:\d{7}', response)
        
        if match:
            hpo_id = match.group(0)
            # Проверяем, что это один из кандидатов
            candidate_ids = [c.hpo_id for c in candidates]
            if hpo_id in candidate_ids:
                return hpo_id
                
        # Если не нашли валидный ID, пробуем найти номер кандидата
        match = re.search(r'^\s*(\d+)', response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx].hpo_id
                
        return None
        
    async def select_batch(self, symptom_candidates: Dict[str, List[HPOCandidate]]) -> Dict[str, Optional[str]]:
        """
        Выбрать HPO коды для множества симптомов параллельно
        
        Args:
            symptom_candidates: Словарь {симптом: [кандидаты]}
            
        Returns:
            Словарь {симптом: выбранный_hpo_id}
        """
        tasks = []
        symptoms = []
        
        for symptom, candidates in symptom_candidates.items():
            if candidates:  # Только если есть кандидаты
                tasks.append(self.select_best_hpo(symptom, candidates))
                symptoms.append(symptom)
                
        # Параллельное выполнение
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Формируем результат
            selected = {}
            for symptom, result in zip(symptoms, results):
                if isinstance(result, Exception):
                    logger.error(f"Error selecting HPO for {symptom}: {result}")
                    # Fallback на первый кандидат
                    candidates = symptom_candidates[symptom]
                    selected[symptom] = candidates[0].hpo_id if candidates else None
                else:
                    selected[symptom] = result
                    
            return selected
        else:
            return {}