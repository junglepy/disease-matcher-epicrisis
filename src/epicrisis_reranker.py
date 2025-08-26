import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import (
    LiricalResult, EpicrisisRerankingResult, EnrichedDiseaseCandidate
)
from .config import Settings
from .hpo_loader import HPOLoader

logger = logging.getLogger(__name__)


class EpicrisisReranker:
    """LLM-based переранжирование результатов LIRICAL для эпикризов"""
    
    def __init__(self, settings: Settings, hpo_loader: HPOLoader):
        self.settings = settings
        self.hpo_loader = hpo_loader
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.0  # Детерминированность важна
        self.max_tokens = 4000  # Достаточно для детального анализа
        
        # Загружаем промпт
        prompt_path = Path(__file__).parent.parent / "prompts" / "epicrisis_reranking.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_system_prompt()
            
    def _get_default_system_prompt(self) -> str:
        """Промпт по умолчанию для переранжирования эпикризов"""
        return """You are a medical genetics expert analyzing epicrisis (discharge summary) in Russian.

Your task: Re-rank disease candidates based on clinical presentation match.

CRITICAL INSTRUCTIONS:
1. Compare patient symptoms with typical disease manifestations
2. Consider symptom frequencies (>80% = very characteristic, <20% = rare)
3. Pay attention to MISSING key symptoms that should be present
4. Consider family history and inheritance patterns
5. Full epicrisis text provides important context - use it

For family cases:
- Multiple affected siblings suggest genetic condition
- Gender distribution hints at inheritance pattern (X-linked if only males affected)
- Consanguinity increases recessive disease probability
- Variable expressivity common in genetic disorders

Output JSON:
{
  "best_match_id": "OMIM:xxxxx",
  "reasoning": "Detailed explanation in English",
  "ranked_candidates": [
    {
      "disease_id": "OMIM:xxxxx", 
      "adjusted_score": 0.95,
      "key_matches": ["symptom1", "symptom2"],
      "key_misses": ["expected_symptom"],
      "confidence_rationale": "Why this ranking"
    }
  ]
}"""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def rerank_lirical_results(
        self,
        lirical_results: List[LiricalResult],
        extracted_symptoms: List[str],
        hpo_codes: List[str],
        original_text: str,
        symptom_to_hpo_mapping: Dict[str, str]
    ) -> EpicrisisRerankingResult:
        """
        Переранжировать результаты LIRICAL на основе полного контекста эпикриза
        
        Args:
            lirical_results: Топ результаты от LIRICAL
            extracted_symptoms: Извлеченные симптомы на русском
            hpo_codes: Найденные HPO коды
            original_text: Полный текст эпикриза
            symptom_to_hpo_mapping: Маппинг симптом -> "HPO:XXX (название)"
            
        Returns:
            EpicrisisRerankingResult с переранжированными кандидатами
        """
        if not lirical_results:
            return EpicrisisRerankingResult(
                ranked_candidates=[],
                reasoning="No candidates to rerank",
                symptom_matches={},
                confidence_adjustments={}
            )
            
        try:
            # Обогащаем кандидатов информацией о HPO профилях
            enriched_candidates = self._enrich_candidates(lirical_results, hpo_codes)
            
            # Формируем промпт для LLM
            user_prompt = self._format_user_prompt(
                enriched_candidates,
                extracted_symptoms,
                hpo_codes,
                original_text,
                symptom_to_hpo_mapping
            )
            
            # Запрос к OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Парсим ответ
            content = response.choices[0].message.content
            logger.debug(f"LLM reranking response: {content[:500]}...")
            
            result = self._parse_llm_response(content, lirical_results)
            return result
            
        except Exception as e:
            logger.error(f"Error in epicrisis reranking: {e}")
            # Возвращаем исходный порядок при ошибке
            return EpicrisisRerankingResult(
                ranked_candidates=lirical_results,
                reasoning=f"Reranking failed: {str(e)}",
                symptom_matches={},
                confidence_adjustments={}
            )
            
    def _enrich_candidates(
        self, 
        lirical_results: List[LiricalResult], 
        patient_hpo_codes: List[str]
    ) -> List[EnrichedDiseaseCandidate]:
        """Обогатить кандидатов информацией о HPO профилях"""
        enriched = []
        
        for result in lirical_results:
            # Получаем HPO профиль заболевания с частотами
            disease_hpo_profile = self.hpo_loader.get_disease_hpo_with_frequencies(
                result.disease_id
            )
            
            # Определяем какие HPO коды пациента совпадают
            matched_hpo = [
                hpo for hpo in patient_hpo_codes 
                if hpo in disease_hpo_profile
            ]
            
            # Определяем ключевые отсутствующие симптомы (частота > 70%)
            missing_key_symptoms = []
            for hpo_id, freq in disease_hpo_profile.items():
                if freq > 0.7 and hpo_id not in patient_hpo_codes:
                    if hpo_id in self.hpo_loader.hpo_terms:
                        term = self.hpo_loader.hpo_terms[hpo_id]
                        missing_key_symptoms.append(f"{hpo_id} ({term.name})")
            
            # Получаем гены заболевания
            genes = self._get_disease_genes(result.disease_id)
            
            enriched_candidate = EnrichedDiseaseCandidate(
                disease_id=result.disease_id,
                disease_name=result.disease_name,
                lirical_score=result.post_test_probability,
                hpo_terms_with_frequencies=disease_hpo_profile,
                matched_hpo_codes=matched_hpo,
                missing_key_symptoms=missing_key_symptoms[:5],  # Топ-5
                genes=genes
            )
            enriched.append(enriched_candidate)
            
        return enriched
        
    def _format_user_prompt(
        self,
        enriched_candidates: List[EnrichedDiseaseCandidate],
        extracted_symptoms: List[str],
        hpo_codes: List[str],
        original_text: str,
        symptom_to_hpo_mapping: Dict[str, str]
    ) -> str:
        """Форматирование промпта для LLM"""
        
        prompt_parts = [
            "Задача: Переранжировать кандидатов заболеваний на основе соответствия клинической картине.",
            "",
            "ПОЛНЫЙ ТЕКСТ ЭПИКРИЗА:",
            original_text,
            "",
            "ИЗВЛЕЧЕННЫЕ СИМПТОМЫ И ИХ HPO КОДЫ:"
        ]
        
        # Добавляем симптомы с их HPO кодами
        for i, symptom in enumerate(extracted_symptoms, 1):
            hpo_info = symptom_to_hpo_mapping.get(symptom, "HPO код не найден")
            prompt_parts.append(f"{i}. \"{symptom}\" → {hpo_info}")
            
        prompt_parts.extend([
            "",
            f"ВСЕГО НАЙДЕНО HPO КОДОВ: {len(hpo_codes)}",
            "",
            "КАНДИДАТЫ ЗАБОЛЕВАНИЙ:"
        ])
        
        # Форматируем кандидатов
        for i, candidate in enumerate(enriched_candidates[:10], 1):
            prompt_parts.extend([
                "",
                f"Кандидат #{i}: {candidate.disease_name} ({candidate.disease_id})",
                f"LIRICAL вероятность: {candidate.lirical_score:.1%}",
                "HPO профиль заболевания:"
            ])
            
            # Показываем топ-10 HPO симптомов заболевания
            sorted_hpo = sorted(
                candidate.hpo_terms_with_frequencies.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for hpo_id, freq in sorted_hpo:
                status = "✓ НАЙДЕН" if hpo_id in hpo_codes else "✗ НЕ НАЙДЕН"
                hpo_name = ""
                if hpo_id in self.hpo_loader.hpo_terms:
                    hpo_name = self.hpo_loader.hpo_terms[hpo_id].name
                prompt_parts.append(
                    f"- {hpo_id} ({hpo_name}) - у {freq:.0%} пациентов {status}"
                )
                
            prompt_parts.extend([
                f"Совпадения: {len(candidate.matched_hpo_codes)}/{len(hpo_codes)} симптомов пациента найдены",
                f"Гены: {', '.join(candidate.genes[:5]) if candidate.genes else 'Не указаны'}"
            ])
            
            if candidate.missing_key_symptoms:
                prompt_parts.append(
                    f"Отсутствуют ключевые симптомы: {', '.join(candidate.missing_key_symptoms[:3])}"
                )
                
        prompt_parts.extend([
            "",
            "Проанализируй соответствие каждого кандидата клинической картине и переранжируй их."
        ])
        
        return "\n".join(prompt_parts)
        
    def _parse_llm_response(
        self, 
        json_str: str, 
        original_results: List[LiricalResult]
    ) -> EpicrisisRerankingResult:
        """Парсинг ответа LLM и создание результата"""
        try:
            data = json.loads(json_str)
            
            # Создаем словарь для быстрого доступа к оригинальным результатам
            result_map = {r.disease_id: r for r in original_results}
            
            # Парсим переранжированных кандидатов
            ranked_candidates = []
            confidence_adjustments = {}
            
            for ranked in data.get("ranked_candidates", []):
                disease_id = ranked["disease_id"]
                if disease_id in result_map:
                    # Копируем оригинальный результат с новым score
                    original = result_map[disease_id]
                    adjusted = LiricalResult(
                        disease_id=original.disease_id,
                        disease_name=original.disease_name,
                        post_test_probability=ranked.get("adjusted_score", original.post_test_probability),
                        maximum_likelihood_ratio=original.maximum_likelihood_ratio,
                        combined_likelihood_ratio=original.combined_likelihood_ratio
                    )
                    ranked_candidates.append(adjusted)
                    
                    # Сохраняем adjustment
                    confidence_adjustments[disease_id] = (
                        ranked["adjusted_score"] - original.post_test_probability
                    )
                    
            # Если не все кандидаты были переранжированы, добавляем остальные
            ranked_ids = {c.disease_id for c in ranked_candidates}
            for original in original_results:
                if original.disease_id not in ranked_ids:
                    ranked_candidates.append(original)
                    
            # Создаем маппинг симптомов (упрощенно)
            symptom_matches = {}
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return EpicrisisRerankingResult(
                ranked_candidates=ranked_candidates,
                reasoning=reasoning,
                symptom_matches=symptom_matches,
                confidence_adjustments=confidence_adjustments
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Возвращаем оригинальный порядок
            return EpicrisisRerankingResult(
                ranked_candidates=original_results,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                symptom_matches={},
                confidence_adjustments={}
            )
            
    def _get_disease_genes(self, disease_id: str) -> List[str]:
        """Получить гены для заболевания"""
        return self.hpo_loader.get_genes_for_disease(disease_id)