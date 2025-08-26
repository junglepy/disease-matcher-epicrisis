from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# Request models
class EpicrisisQuery(BaseModel):
    """Запрос для анализа эпикриза"""
    epicrisis_text: str = Field(..., description="Текст эпикриза с симптомами и анамнезом")
    language: str = Field("ru", description="Язык эпикриза (ru/en)")
    top_k: int = Field(10, ge=1, le=50, description="Количество результатов")


# Response models (unified format - совместимый с существующим API)
class UnifiedDiseaseMatch(BaseModel):
    """Единый формат результата"""
    rank: int = Field(..., description="Позиция в результатах")
    omim_id: Optional[str] = Field(None, description="OMIM идентификатор")
    mondo_id: Optional[str] = Field(None, description="MONDO идентификатор")
    name: str = Field(..., description="Название заболевания")
    score: float = Field(..., ge=0.0, le=1.0, description="Оценка уверенности")
    genes: List[str] = Field(default_factory=list, description="Связанные гены")


class UnifiedMetadata(BaseModel):
    """Метаданные ответа"""
    processing_time_ms: int = Field(..., description="Время обработки в миллисекундах")
    architecture: str = Field("epicrisis_lirical", description="Архитектура системы")
    model: str = Field(..., description="Используемая модель")
    total_results: int = Field(..., description="Количество результатов")


class ClinicalNotes(BaseModel):
    """Клинические заметки для врача"""
    summary: str = Field(..., description="Краткое резюме анализа")
    key_findings: List[str] = Field(..., description="Ключевые находки")
    recommendations: Optional[str] = Field(None, description="Рекомендации по дальнейшей диагностике")


class HPOTerm(BaseModel):
    """HPO термин с кодом и названием"""
    code: str = Field(..., description="HPO код (например, HP:0001234)")
    name: str = Field(..., description="Название термина")


class UnifiedExtended(BaseModel):
    """Расширенная информация"""
    extracted_symptoms: List[str] = Field(..., description="Извлеченные симптомы")
    hpo_codes: List[str] = Field(..., description="Найденные HPO коды")
    hpo_terms: Optional[List[HPOTerm]] = Field(None, description="HPO коды с названиями")
    lirical_scores: Optional[Dict[str, float]] = Field(None, description="Оценки LIRICAL")
    extracted_codes: Optional[Dict[str, List[str]]] = Field(None, description="Извлеченные коды (OMIM, HPO, ICD)")
    clinical_notes: Optional[ClinicalNotes] = Field(None, description="Клинические заметки для врача")


class UnifiedMatchResponse(BaseModel):
    """Унифицированный формат ответа"""
    results: List[UnifiedDiseaseMatch] = Field(..., description="Результаты поиска")
    metadata: UnifiedMetadata = Field(..., description="Метаданные")
    extended: Optional[UnifiedExtended] = Field(None, description="Расширенная информация")
    error: Optional[Dict[str, str]] = Field(None, description="Информация об ошибке")


# Internal models
class ExtractedSymptoms(BaseModel):
    """Результат извлечения симптомов из эпикриза"""
    symptoms: List[str] = Field(..., description="Список симптомов на русском")
    codes: Dict[str, List[str]] = Field(default_factory=dict, description="Извлеченные коды")
    diagnoses_mentioned: List[str] = Field(default_factory=list, description="Упомянутые диагнозы")
    family_history: List[str] = Field(default_factory=list, description="Семейная история")
    pregnancy_birth: List[str] = Field(default_factory=list, description="Данные о беременности и родах")
    age_onset: Dict[str, str] = Field(default_factory=dict, description="Возраст начала симптомов")


class HPOCandidate(BaseModel):
    """Кандидат HPO термина"""
    hpo_id: str = Field(..., description="HPO идентификатор")
    name: str = Field(..., description="Название термина")
    synonyms: List[str] = Field(default_factory=list, description="Синонимы")
    score: float = Field(..., description="BM25 score")


class LiricalResult(BaseModel):
    """Результат анализа LIRICAL"""
    disease_id: str
    disease_name: str
    post_test_probability: float
    maximum_likelihood_ratio: float
    combined_likelihood_ratio: float


# Models for LLM reranking
class EnrichedDiseaseCandidate(BaseModel):
    """Обогащенный кандидат заболевания для reranking"""
    disease_id: str = Field(..., description="ID заболевания (OMIM:xxxxx или MONDO:xxxxx)")
    disease_name: str = Field(..., description="Название заболевания")
    lirical_score: float = Field(..., description="Оценка от LIRICAL")
    hpo_terms_with_frequencies: Dict[str, float] = Field(..., description="HPO коды с частотами (HPO:xxxxx -> 0.95)")
    matched_hpo_codes: List[str] = Field(..., description="HPO коды пациента, найденные у заболевания")
    missing_key_symptoms: List[str] = Field(..., description="Ключевые симптомы заболевания, отсутствующие у пациента")
    genes: List[str] = Field(default_factory=list, description="Ассоциированные гены")


class EpicrisisRerankingResult(BaseModel):
    """Результат переранжирования для эпикризов"""
    ranked_candidates: List[LiricalResult] = Field(..., description="Переранжированные кандидаты")
    reasoning: str = Field(..., description="Объяснение переранжирования")
    symptom_matches: Dict[str, List[str]] = Field(..., description="Маппинг симптом -> найденные HPO коды")
    confidence_adjustments: Dict[str, float] = Field(..., description="Изменения confidence для каждого заболевания")


# Health check
class HealthResponse(BaseModel):
    """Статус сервиса"""
    status: str = "healthy"
    version: str = "1.0.0"
    components: Dict[str, bool] = Field(default_factory=dict)
    lirical_status: Optional[str] = None