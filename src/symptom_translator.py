"""Перевод медицинских симптомов с русского на английский"""

import logging
from typing import List, Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SymptomTranslator:
    """Переводчик симптомов с русского на английский"""
    
    def __init__(self):
        # Словарь медицинских терминов
        self.medical_dict = self._load_medical_dictionary()
        
    def _load_medical_dictionary(self) -> Dict[str, str]:
        """Загрузить словарь медицинских терминов"""
        # Базовый словарь распространенных симптомов
        base_dict = {
            # Неврологические симптомы
            "головная боль": "headache",
            "головокружение": "dizziness",
            "судороги": "seizures",
            "эпилепсия": "epilepsy",
            "параличи": "paralysis",
            "парез": "paresis",
            "тремор": "tremor",
            "дрожание": "tremor",
            "нистагм": "nystagmus",
            "атаксия": "ataxia",
            "дизартрия": "dysarthria",
            "афазия": "aphasia",
            
            # Когнитивные нарушения
            "умственная отсталость": "intellectual disability",
            "задержка развития": "developmental delay",
            "задержка психического развития": "mental retardation",
            "деменция": "dementia",
            "нарушение памяти": "memory impairment",
            "когнитивные нарушения": "cognitive impairment",
            
            # Общие симптомы
            "лихорадка": "fever",
            "температура": "fever",
            "слабость": "weakness",
            "утомляемость": "fatigue",
            "боль": "pain",
            "отек": "edema",
            "отеки": "edema",
            "одышка": "dyspnea",
            "кашель": "cough",
            "тошнота": "nausea",
            "рвота": "vomiting",
            
            # Симптомы со стороны глаз
            "слепота": "blindness",
            "нарушение зрения": "visual impairment",
            "косоглазие": "strabismus",
            "катаракта": "cataract",
            "глаукома": "glaucoma",
            
            # Симптомы со стороны слуха
            "глухота": "deafness",
            "тугоухость": "hearing loss",
            "нарушение слуха": "hearing impairment",
            
            # Кожные проявления
            "сыпь": "rash",
            "зуд": "pruritus",
            "желтуха": "jaundice",
            "цианоз": "cyanosis",
            "бледность": "pallor",
            
            # Костно-мышечные симптомы
            "контрактуры": "contractures",
            "деформация": "deformity",
            "сколиоз": "scoliosis",
            "кифоз": "kyphosis",
            "гипотония": "hypotonia",
            "гипертония": "hypertonia",
            "мышечная слабость": "muscle weakness",
            "мышечная атрофия": "muscle atrophy",
            
            # Сердечно-сосудистые симптомы
            "аритмия": "arrhythmia",
            "тахикардия": "tachycardia",
            "брадикардия": "bradycardia",
            "гипертензия": "hypertension",
            "гипотензия": "hypotension",
            
            # Желудочно-кишечные симптомы
            "диарея": "diarrhea",
            "запор": "constipation",
            "дисфагия": "dysphagia",
            "гепатомегалия": "hepatomegaly",
            "спленомегалия": "splenomegaly",
            
            # Эндокринные симптомы
            "низкий рост": "short stature",
            "высокий рост": "tall stature",
            "ожирение": "obesity",
            "истощение": "cachexia",
            
            # Лицевые аномалии
            "микроцефалия": "microcephaly",
            "макроцефалия": "macrocephaly",
            "гидроцефалия": "hydrocephalus",
            "дисморфизм": "dysmorphism",
            "лицевые аномалии": "facial anomalies",
            "эпикантус": "epicanthus",
            "гипертелоризм": "hypertelorism",
            "гипотелоризм": "hypotelorism",
            
            # Конечности
            "полидактилия": "polydactyly",
            "синдактилия": "syndactyly",
            "брахидактилия": "brachydactyly",
            "арахнодактилия": "arachnodactyly",
            
            # Другие
            "аутизм": "autism",
            "гиперактивность": "hyperactivity",
            "агрессивность": "aggressiveness",
            "тревожность": "anxiety",
            "депрессия": "depression",
            "бессонница": "insomnia",
            "апноэ": "apnea",
            "дыхательная недостаточность": "respiratory failure",
            "почечная недостаточность": "renal failure",
            "печеночная недостаточность": "liver failure"
        }
        
        # Попробуем загрузить расширенный словарь из файла
        dict_path = Path(__file__).parent.parent / "data" / "medical_dictionary_ru_en.json"
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    extended_dict = json.load(f)
                    base_dict.update(extended_dict)
                    logger.info(f"Loaded extended medical dictionary with {len(extended_dict)} terms")
            except Exception as e:
                logger.warning(f"Failed to load extended dictionary: {e}")
        
        return base_dict
    
    def translate_symptom(self, symptom: str) -> str:
        """
        Перевести один симптом с русского на английский
        
        Args:
            symptom: Симптом на русском языке
            
        Returns:
            Переведенный симптом или оригинал, если перевод не найден
        """
        # Приводим к нижнему регистру для поиска
        symptom_lower = symptom.lower().strip()
        
        # Ищем в словаре
        if symptom_lower in self.medical_dict:
            return self.medical_dict[symptom_lower]
        
        # Пробуем найти частичное совпадение
        for ru_term, en_term in self.medical_dict.items():
            if ru_term in symptom_lower or symptom_lower in ru_term:
                logger.debug(f"Partial match: '{symptom}' -> '{en_term}'")
                return en_term
        
        # Если не нашли перевод, возвращаем как есть
        logger.debug(f"No translation found for: '{symptom}'")
        return symptom
    
    def translate_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Перевести список симптомов
        
        Args:
            symptoms: Список симптомов на русском языке
            
        Returns:
            Список переведенных симптомов
        """
        translated = []
        
        for symptom in symptoms:
            translated_symptom = self.translate_symptom(symptom)
            translated.append(translated_symptom)
            
            if translated_symptom != symptom:
                logger.debug(f"Translated: '{symptom}' -> '{translated_symptom}'")
        
        return translated
    
    def is_russian(self, text: str) -> bool:
        """Проверить, содержит ли текст русские символы"""
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        return any(char.lower() in russian_chars for char in text)