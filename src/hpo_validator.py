"""
HPO Validator - validates HPO candidates found by BM25 search using LLM
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import HPOCandidate
from .config import Settings

logger = logging.getLogger(__name__)


class HPOValidator:
    """Validates HPO codes by checking if they match the original symptoms"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.0  # Zero temperature for consistent validation
        
        # Load validation prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "hpo_validation.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Default validation prompt"""
        return """You are a medical expert validating HPO (Human Phenotype Ontology) codes.

Your task is to determine if each HPO code accurately represents the symptom from the original clinical text.

For each HPO candidate, you must:
1. Check if the HPO term definition matches the symptom
2. Consider medical synonyms and related concepts
3. Be strict - only approve exact or very close matches
4. Consider the context from the epicrisis

Return a JSON object with this structure:
{
  "validated_hpo": [
    {
      "hpo_id": "HP:0001250",
      "is_valid": true,
      "confidence": 0.95,
      "reasoning": "HPO term 'Seizure' directly matches symptom 'судороги'"
    }
  ]
}

Important:
- is_valid: true only if the HPO accurately represents the symptom
- confidence: 0.0-1.0 score of how well it matches
- reasoning: brief explanation in English"""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def validate_hpo_candidates(
        self, 
        symptom: str,
        candidates: List[HPOCandidate],
        epicrisis_context: str,
        threshold: float = 0.7
    ) -> List[Tuple[HPOCandidate, float]]:
        """
        Validate HPO candidates for a symptom
        
        Args:
            symptom: Original symptom text
            candidates: List of HPO candidates from BM25 search
            epicrisis_context: Original epicrisis text for context
            threshold: Minimum confidence threshold
            
        Returns:
            List of (HPOCandidate, confidence) tuples for valid matches
        """
        if not candidates:
            return []
            
        try:
            # Prepare candidates data for validation
            candidates_data = []
            for candidate in candidates[:10]:  # Limit to top 10
                candidates_data.append({
                    "hpo_id": candidate.hpo_id,
                    "hpo_name": candidate.name,
                    "synonyms": candidate.synonyms[:3] if candidate.synonyms else []
                })
                
            # Create validation request
            validation_request = {
                "symptom": symptom,
                "candidates": candidates_data,
                "epicrisis_excerpt": epicrisis_context[:500]  # Limit context size
            }
            
            # Call LLM for validation
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(validation_request, ensure_ascii=False)}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.debug(f"Validation response: {content}")
            
            data = json.loads(content)
            validated = data.get("validated_hpo", [])
            
            # Match validated results with original candidates
            valid_candidates = []
            for item in validated:
                if item.get("is_valid", False) and item.get("confidence", 0) >= threshold:
                    # Find the original candidate
                    hpo_id = item["hpo_id"]
                    for candidate in candidates:
                        if candidate.hpo_id == hpo_id:
                            valid_candidates.append((candidate, item["confidence"]))
                            logger.info(
                                f"Validated {hpo_id} for '{symptom}' "
                                f"with confidence {item['confidence']}: {item.get('reasoning', '')}"
                            )
                            break
                            
            return valid_candidates
            
        except Exception as e:
            logger.error(f"Error validating HPO candidates: {e}")
            # Return top candidates with reduced confidence as fallback
            return [(c, 0.5) for c in candidates[:3]]
            
    async def validate_batch(
        self,
        symptom_candidates: Dict[str, List[HPOCandidate]],
        epicrisis_context: str,
        threshold: float = 0.7
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Validate HPO candidates for multiple symptoms
        
        Args:
            symptom_candidates: Dict mapping symptoms to HPO candidates
            epicrisis_context: Original epicrisis text
            threshold: Minimum confidence threshold
            
        Returns:
            Dict mapping symptoms to list of (hpo_id, confidence) tuples
        """
        results = {}
        
        for symptom, candidates in symptom_candidates.items():
            if not candidates:
                results[symptom] = []
                continue
                
            validated = await self.validate_hpo_candidates(
                symptom, candidates, epicrisis_context, threshold
            )
            
            # Convert to (hpo_id, confidence) tuples
            results[symptom] = [(c.hpo_id, conf) for c, conf in validated]
            
        return results