"""
Disease Reranker - uses LLM to rerank top disease candidates based on epicrisis
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from openai import AsyncOpenAI

from .models import LiricalResult
from .config import Settings
from .hpo_loader import HPOLoader

logger = logging.getLogger(__name__)


class DiseaseReranker:
    """Reranks disease candidates using LLM with full context"""
    
    def __init__(self, settings: Settings, hpo_loader: HPOLoader):
        self.settings = settings
        self.hpo_loader = hpo_loader
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        # Use dedicated model for final reranking if specified
        self.model = getattr(settings, 'final_disease_rerank_model', settings.openai_model)
        self.temperature = 0.0  # Deterministic for reranking
        
    async def rerank_diseases(
        self,
        epicrisis_text: str,
        disease_candidates: List[LiricalResult],
        extracted_hpo_codes: List[str],
        top_k: int = 5,
        generate_notes: bool = True
    ) -> Dict[str, Any]:
        """
        Rerank disease candidates using LLM
        
        Args:
            epicrisis_text: Original epicrisis text
            disease_candidates: Top 20 disease candidates from LIRICAL
            extracted_hpo_codes: HPO codes extracted from epicrisis
            top_k: Number of top diseases to return
            
        Returns:
            Reranked list of top K diseases
        """
        if len(disease_candidates) <= top_k:
            return disease_candidates
            
        logger.info(f"Reranking {len(disease_candidates)} disease candidates to select top {top_k}")
        
        # Prepare disease context with symptoms and frequencies
        disease_context = []
        for disease in disease_candidates[:20]:  # Take top 20
            disease_info = {
                "disease_id": disease.disease_id,
                "disease_name": disease.disease_name,
                "lirical_score": disease.post_test_probability,
                "associated_genes": self._get_disease_genes(disease.disease_id),
                "symptoms": []
            }
            
            # Get HPO codes and frequencies for this disease
            hpo_with_freq = self.hpo_loader.get_disease_hpo_with_frequencies(disease.disease_id)
            
            # Sort by frequency and take most common symptoms
            sorted_symptoms = sorted(
                hpo_with_freq.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:15]  # Top 15 symptoms
            
            for hpo_code, frequency in sorted_symptoms:
                if hpo_code in self.hpo_loader.hpo_terms:
                    term = self.hpo_loader.hpo_terms[hpo_code]
                    disease_info["symptoms"].append({
                        "hpo_code": hpo_code,
                        "name": term.name,
                        "frequency": f"{frequency*100:.0f}%",
                        "present_in_patient": hpo_code in extracted_hpo_codes
                    })
                    
            disease_context.append(disease_info)
            
        # Create reranking prompt with optional clinical notes
        if generate_notes:
            prompt = """You are a medical expert reranking disease candidates based on patient epicrisis.

Given:
1. Patient epicrisis (clinical description)
2. HPO codes extracted from the epicrisis
3. Top 20 disease candidates with their symptoms and frequencies

Your task: Select the TOP 5 most likely diseases and generate clinical notes for the physician.

For clinical notes, be SPECIFIC and ACTIONABLE:
- Name the most likely diseases (top 1-2)
- List specific genes to test
- Suggest concrete diagnostic steps
- Mention differential diagnoses to consider

Return JSON:
{
  "reranked_diseases": [
    {
      "disease_id": "OMIM:123456",
      "disease_name": "Disease Name",
      "confidence": 0.95,
      "reasoning": "Strong match: patient has 4/5 core symptoms including..."
    }
  ],
  "analysis": "Brief analysis of why these 5 were selected",
  "clinical_notes": {
    "summary": "Наиболее вероятный диагноз - [конкретное заболевание] (OMIM:XXXXX) на основании сочетания [ключевые симптомы]. Альтернативные диагнозы включают...",
    "key_findings": [
      "Характерная триада симптомов: [конкретные симптомы]",
      "Возраст манифестации соответствует типичному для [заболевание]",
      "Наличие [конкретный симптом] высокоспецифично для данной патологии"
    ],
    "recommendations": "1. Генетическое тестирование: секвенирование гена [конкретный ген] (наиболее частая причина)\n2. Дополнительные исследования: [конкретные исследования]\n3. Консультации: [конкретные специалисты]\n4. При отрицательном результате рассмотреть тестирование генов [альтернативные гены]"
  }
}"""
        else:
            prompt = """You are a medical expert reranking disease candidates based on patient epicrisis.

Given:
1. Patient epicrisis (clinical description)
2. HPO codes extracted from the epicrisis
3. Top 20 disease candidates with their symptoms and frequencies

Your task: Select the TOP 5 most likely diseases based on:
- How well the patient's symptoms match the disease profile
- Frequency of symptoms in each disease
- Clinical coherence with the epicrisis

Return JSON with exactly 5 diseases in order of likelihood:
{
  "reranked_diseases": [
    {
      "disease_id": "OMIM:123456",
      "disease_name": "Disease Name",
      "confidence": 0.95,
      "reasoning": "Strong match: patient has 4/5 core symptoms including..."
    }
  ],
  "analysis": "Brief analysis of why these 5 were selected"
}"""
        
        # Prepare request
        request = {
            "epicrisis": epicrisis_text,
            "extracted_hpo_codes": [
                {
                    "code": code,
                    "name": self.hpo_loader.hpo_terms[code].name if code in self.hpo_loader.hpo_terms else code
                }
                for code in extracted_hpo_codes
            ],
            "disease_candidates": disease_context
        }
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)}
                ],
                temperature=self.temperature,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            logger.debug(f"Reranking analysis: {data.get('analysis', '')}")
            
            # Create mapping of disease_id to original disease object
            disease_map = {d.disease_id: d for d in disease_candidates}
            
            # Build reranked list
            reranked = []
            for item in data.get("reranked_diseases", [])[:top_k]:
                disease_id = item.get("disease_id")
                if disease_id in disease_map:
                    disease = disease_map[disease_id]
                    # Update confidence based on LLM reranking
                    disease.post_test_probability = item.get("confidence", disease.post_test_probability)
                    reranked.append(disease)
                    logger.info(
                        f"Reranked: {disease_id} ({item.get('disease_name', 'Unknown')}) "
                        f"confidence={item.get('confidence', 0):.2f} - {item.get('reasoning', '')[:100]}"
                    )
                    
            # If we didn't get enough results, fill with original order
            if len(reranked) < top_k:
                for disease in disease_candidates:
                    if disease not in reranked:
                        reranked.append(disease)
                        if len(reranked) >= top_k:
                            break
                            
            logger.info(f"Reranking complete: selected {len(reranked)} diseases")
            
            # Prepare result
            result = {
                'diseases': reranked,
                'clinical_notes': data.get('clinical_notes') if generate_notes else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in disease reranking: {e}")
            # Fallback: return original top K
            return {
                'diseases': disease_candidates[:top_k],
                'clinical_notes': None
            }
    
    def _get_disease_genes(self, disease_id: str) -> List[str]:
        """Получить гены для заболевания"""
        return self.hpo_loader.get_genes_for_disease(disease_id)