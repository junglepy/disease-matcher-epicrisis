"""
Enhanced symptom extractor that extracts symptoms with English synonyms in a single LLM call
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import ExtractedSymptoms
from .config import Settings

logger = logging.getLogger(__name__)


class EnhancedSymptomExtractor:
    """Extract symptoms with English synonyms in a single LLM call"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.1  # Low temperature for consistency
        
        # Load prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "symptom_extraction_with_synonyms.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Default prompt for symptom extraction with synonyms"""
        return """You are a medical expert tasked with extracting symptoms from Russian clinical texts and providing English medical term variations for each symptom.

For each symptom found in the text:
1. Extract the symptom in Russian as it appears
2. Provide 3-5 English medical term variations including:
   - Direct medical translation
   - Medical synonyms and related terms
   - Alternative clinical descriptions
   - Common medical abbreviations if applicable

Also extract:
- Any mentioned medical codes (OMIM, HPO, ICD)
- Mentioned diagnoses (not symptoms)
- Family history
- Pregnancy/birth complications
- Age of onset information

Return a JSON object with this exact structure:
{
  "symptoms_with_synonyms": {
    "судороги": ["seizures", "convulsions", "epileptic seizures", "fits", "spasms"],
    "задержка развития": ["developmental delay", "delayed development", "growth retardation", "developmental retardation", "delayed milestones"]
  },
  "codes": {
    "omim": ["123456"],
    "hpo": ["HP:0001250"],
    "icd": ["G40.9"]
  },
  "diagnoses_mentioned": ["эпилепсия", "ДЦП"],
  "family_history": ["мать - эпилепсия"],
  "pregnancy_birth": ["преждевременные роды"],
  "age_onset": {"first_symptoms": "3 месяца", "diagnosis": "1 год"}
}

Important:
- Extract ONLY symptoms, not diagnoses in symptoms_with_synonyms
- Provide diverse English variations that might be used in medical databases
- Include both formal medical terms and common clinical descriptions
- Ensure all English terms are medically accurate"""
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def extract_symptoms_with_synonyms(self, epicrisis_text: str) -> Dict[str, List[str]]:
        """
        Extract symptoms with English synonyms from epicrisis text
        
        Args:
            epicrisis_text: Russian clinical text
            
        Returns:
            Dictionary mapping Russian symptoms to lists of English synonyms
        """
        try:
            # Call OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Извлеките симптомы из эпикриза:\n\n{epicrisis_text}"}
                ],
                temperature=self.temperature,
                max_tokens=self.settings.openai_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content}")
            
            data = json.loads(content)
            
            # Extract symptoms with synonyms
            symptoms_with_synonyms = data.get("symptoms_with_synonyms", {})
            
            # Validate and clean
            cleaned_symptoms = {}
            for symptom, synonyms in symptoms_with_synonyms.items():
                # Clean symptom
                symptom = symptom.strip()
                if not symptom:
                    continue
                    
                # Clean and validate synonyms
                cleaned_synonyms = []
                for synonym in synonyms:
                    if isinstance(synonym, str) and synonym.strip():
                        cleaned_synonyms.append(synonym.strip())
                        
                # Only include if we have valid synonyms
                if cleaned_synonyms:
                    cleaned_symptoms[symptom] = cleaned_synonyms[:5]  # Max 5 synonyms
                    
            logger.info(f"Extracted {len(cleaned_symptoms)} symptoms with synonyms")
            
            # Create ExtractedSymptoms object for compatibility
            self.last_extraction = ExtractedSymptoms(
                symptoms=list(cleaned_symptoms.keys()),
                codes=data.get("codes", {}),
                diagnoses_mentioned=data.get("diagnoses_mentioned", []),
                family_history=data.get("family_history", []),
                pregnancy_birth=data.get("pregnancy_birth", []),
                age_onset=data.get("age_onset", {})
            )
            
            return cleaned_symptoms
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting symptoms with synonyms: {e}")
            raise
            
    async def get_full_extraction(self) -> ExtractedSymptoms:
        """Get the full extraction result from the last call"""
        if hasattr(self, 'last_extraction'):
            return self.last_extraction
        return ExtractedSymptoms(symptoms=[], codes={}, diagnoses_mentioned=[])