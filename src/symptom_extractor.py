import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import ExtractedSymptoms
from .config import Settings

logger = logging.getLogger(__name__)


class SymptomExtractor:
    """Извлечение симптомов из эпикриза с помощью LLM"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.0  # Для детерминированности
        
        # Загружаем промпт
        prompt_path = Path(__file__).parent.parent / "prompts" / "symptom_extraction.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Минимальный промпт для извлечения симптомов"""
        return """Extract symptoms from Russian medical text.
Separate: symptoms, existing codes (OMIM/HPO/ICD), mentioned diagnoses.
Return JSON: {"symptoms": [...], "codes": {"omim": [...], "hpo": [...], "icd": [...]}, "diagnoses_mentioned": [...]}
Only symptoms, no diagnoses in symptoms list."""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def extract_symptoms(self, epicrisis_text: str) -> ExtractedSymptoms:
        """
        Извлечь симптомы из текста эпикриза
        
        Args:
            epicrisis_text: Текст эпикриза на русском языке
            
        Returns:
            ExtractedSymptoms с симптомами, кодами и упомянутыми диагнозами
        """
        try:
            # Вызов OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Текст эпикриза:\n{epicrisis_text}"}
                ],
                temperature=self.temperature,
                max_tokens=self.settings.openai_max_tokens,
                response_format={"type": "json_object"}  # Принудительный JSON
            )
            
            # Парсим ответ
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content}")
            
            data = json.loads(content)
            
            # Создаем объект результата
            extracted = ExtractedSymptoms(
                symptoms=data.get("symptoms", []),
                codes=data.get("codes", {}),
                diagnoses_mentioned=data.get("diagnoses_mentioned", []),
                family_history=data.get("family_history", []),
                pregnancy_birth=data.get("pregnancy_birth", []),
                age_onset=data.get("age_onset", {})
            )
            
            # Валидация и очистка
            extracted = self._validate_and_clean(extracted)
            
            logger.info(f"Extracted {len(extracted.symptoms)} symptoms from epicrisis")
            return extracted
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Возвращаем пустой результат
            return ExtractedSymptoms(symptoms=[], codes={}, diagnoses_mentioned=[])
            
        except Exception as e:
            logger.error(f"Error extracting symptoms: {e}")
            raise
            
    def _validate_and_clean(self, extracted: ExtractedSymptoms) -> ExtractedSymptoms:
        """Валидация и очистка извлеченных данных"""
        # Убираем дубликаты из симптомов
        extracted.symptoms = list(dict.fromkeys(extracted.symptoms))
        
        # Фильтруем пустые симптомы
        extracted.symptoms = [s.strip() for s in extracted.symptoms if s.strip()]
        
        # Валидация кодов
        valid_codes = {}
        
        # OMIM коды должны быть числами
        if "omim" in extracted.codes:
            valid_omim = []
            for code in extracted.codes["omim"]:
                # Убираем префикс OMIM: если есть
                code = code.replace("OMIM:", "").strip()
                if code.isdigit() and len(code) == 6:
                    valid_omim.append(code)
            if valid_omim:
                valid_codes["omim"] = valid_omim
                
        # HPO коды должны начинаться с HP:
        if "hpo" in extracted.codes:
            valid_hpo = []
            for code in extracted.codes["hpo"]:
                if not code.startswith("HP:"):
                    code = f"HP:{code}"
                # Проверяем формат HP:XXXXXXX
                if code.startswith("HP:") and len(code) == 10:
                    valid_hpo.append(code)
            if valid_hpo:
                valid_codes["hpo"] = valid_hpo
                
        # ICD коды - пока просто сохраняем как есть
        if "icd" in extracted.codes and extracted.codes["icd"]:
            valid_codes["icd"] = extracted.codes["icd"]
            
        extracted.codes = valid_codes
        
        # Убираем дубликаты из диагнозов
        extracted.diagnoses_mentioned = list(dict.fromkeys(extracted.diagnoses_mentioned))
        
        return extracted