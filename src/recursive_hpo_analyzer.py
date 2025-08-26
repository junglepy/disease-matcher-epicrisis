"""
Recursive HPO Analyzer - performs recursive analysis with LIRICAL feedback
"""

import json
import logging
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import requests

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from .models import LiricalResult
from .config import Settings
from .hpo_loader import HPOLoader

logger = logging.getLogger(__name__)


class RecursiveHPOAnalyzer:
    """Performs recursive HPO analysis using LIRICAL predictions"""
    
    def __init__(self, settings: Settings, hpo_loader: HPOLoader, lirical_wrapper=None):
        self.settings = settings
        self.hpo_loader = hpo_loader
        self.lirical_wrapper = lirical_wrapper
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = 0.1
        
        # LIRICAL service URL (used only if lirical_wrapper not provided)
        self.lirical_url = getattr(settings, 'lirical_service_url', 'http://localhost:8083')
        
        # Recursive analysis parameters
        self.max_iterations = getattr(settings, 'recursive_iterations', 2)
        self.top_diseases = getattr(settings, 'recursive_top_diseases', 20)
        
        # Load prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "recursive_hpo_selection.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = self._get_default_prompt()
            
    def _get_default_prompt(self) -> str:
        """Default prompt for recursive HPO selection"""
        return """You are a medical expert selecting relevant HPO codes from an expanded list.

Given:
1. Original epicrisis text
2. HPO codes already found
3. Possible diseases from LIRICAL
4. Extended list of HPO codes from these diseases

Select ONLY HPO codes that are clearly present in the epicrisis.

Return JSON:
{
  "selected_hpo": [
    {
      "hpo_id": "HP:0001250",
      "hpo_name": "Seizure",
      "confidence": 0.95,
      "evidence": "судороги mentioned in epicrisis"
    }
  ],
  "rejected_count": 45,
  "reasoning": "Selected 3 additional HPO codes clearly present in text"
}"""
        
    async def perform_recursive_analysis(
        self,
        initial_hpo: List[str],
        epicrisis_text: str,
        iteration: int = 0
    ) -> Tuple[List[str], List[LiricalResult]]:
        """
        Perform recursive HPO analysis
        
        Args:
            initial_hpo: Initial HPO codes found
            epicrisis_text: Original epicrisis text
            iteration: Current iteration number
            
        Returns:
            Tuple of (final HPO codes, final LIRICAL results)
        """
        logger.info(f"Starting recursive analysis iteration {iteration + 1}/{self.max_iterations}")
        
        # Track all HPO codes found across iterations
        all_hpo_codes = set(initial_hpo)
        current_hpo = initial_hpo.copy()
        previous_lirical_hpo = initial_hpo.copy()  # Cache for HPO from previous LIRICAL iteration
        
        for i in range(self.max_iterations):
            logger.info(f"Recursive iteration {i + 1}: Starting with {len(current_hpo)} HPO codes")
            
            # Step 1: Get disease predictions from LIRICAL
            lirical_results = await self._call_lirical(current_hpo)
            if not lirical_results:
                logger.warning("No LIRICAL results, stopping recursion")
                break
                
            # Cache the HPO codes used in this LIRICAL call
            previous_lirical_hpo = current_hpo.copy()
                
            # Step 2: Collect HPO codes from top diseases
            expanded_hpo = self._collect_hpo_from_diseases(lirical_results[:self.top_diseases])
            logger.info(f"Collected {len(expanded_hpo)} HPO codes from top {self.top_diseases} diseases")
            
            # Filter out already found HPO codes
            new_candidates = expanded_hpo - all_hpo_codes
            logger.info(f"Found {len(new_candidates)} new HPO candidates")
            
            if not new_candidates:
                logger.info("No new HPO candidates, stopping recursion")
                break
                
            # Step 3: Use LLM to select relevant HPO from expanded list (in batches)
            batch_selected_hpo = await self._select_relevant_hpo(
                epicrisis_text,
                list(all_hpo_codes),
                list(new_candidates),
                lirical_results[:10]  # Top 10 diseases for context
            )
            
            logger.info(f"Batch processing selected {len(batch_selected_hpo)} HPO codes")
            
            if not batch_selected_hpo:
                logger.info("No new HPO codes selected from batches, stopping recursion")
                break
                
            # Step 4: Consolidate selected HPO with previous LIRICAL HPO
            consolidated_hpo = await self._consolidate_selected_hpo(
                batch_selected_hpo,
                previous_lirical_hpo,
                epicrisis_text
            )
            
            # Update HPO lists with consolidated results
            new_selected = set(consolidated_hpo) - all_hpo_codes
            all_hpo_codes.update(new_selected)
            current_hpo = list(all_hpo_codes)
            
            logger.info(f"After consolidation: {len(new_selected)} new HPO codes added, total: {len(current_hpo)}")
            
            # Optional: Early stopping if we're not finding many new codes
            if len(new_selected) < 2:
                logger.info(f"Only {len(new_selected)} new codes found after consolidation, stopping early")
                break
                
        # Final LIRICAL analysis with all HPO codes
        logger.info(f"Final analysis with {len(current_hpo)} total HPO codes")
        final_results = await self._call_lirical(current_hpo)
        
        return current_hpo, final_results
        
    async def _call_lirical(self, hpo_codes: List[str]) -> List[LiricalResult]:
        """Call LIRICAL service"""
        # Use lirical_wrapper if available (it has disease name correction)
        if self.lirical_wrapper:
            logger.debug("Using lirical_wrapper for LIRICAL analysis")
            return self.lirical_wrapper.analyze_phenotypes(hpo_codes, top_k=30)
            
        # Fallback to direct HTTP call
        logger.debug("Using direct HTTP call to LIRICAL")
        try:
            url = f"{self.lirical_url}/api/v1/analyze"
            payload = {
                "hpo_codes": hpo_codes,
                "max_results": 30  # Get more results for recursive analysis
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"LIRICAL error: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                result = LiricalResult(
                    disease_id=item.get("disease_id", ""),
                    disease_name=item.get("disease_name", ""),
                    post_test_probability=item.get("post_test_probability", 0.0),
                    maximum_likelihood_ratio=item.get("composite_lr", 0.0),
                    combined_likelihood_ratio=item.get("composite_lr", 0.0)
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error calling LIRICAL: {e}")
            return []
            
    def _collect_hpo_from_diseases(self, diseases: List[LiricalResult]) -> Set[str]:
        """Collect all HPO codes associated with the given diseases"""
        all_hpo = set()
        
        # Get frequency threshold from settings
        frequency_threshold = getattr(self.settings, 'hpo_frequency_threshold', 0.1)
        max_hpo_per_disease = getattr(self.settings, 'max_hpo_per_disease', 10)
        
        for disease in diseases:
            # Extract disease ID (handle both OMIM:123456 and 123456 formats)
            disease_id = disease.disease_id
            
            # Get HPO codes with frequencies
            hpo_with_freq = self.hpo_loader.get_disease_hpo_with_frequencies(disease_id)
            
            # Also try without prefix if no results
            if not hpo_with_freq and ":" in disease_id:
                bare_id = disease_id.split(":")[-1]
                hpo_with_freq = self.hpo_loader.get_disease_hpo_with_frequencies(bare_id)
                    
            # If still no local data, try LIRICAL API
            if not hpo_with_freq:
                logger.debug(f"No local HPO data for {disease_id}, trying LIRICAL API")
                hpo_from_api = self._get_hpo_from_lirical_api(disease_id)
                if hpo_from_api:
                    # API doesn't provide frequencies, use default 0.5
                    hpo_with_freq = {hpo: 0.5 for hpo in hpo_from_api}
            
            if hpo_with_freq:
                # Filter by frequency threshold
                filtered_hpo = {
                    hpo: freq for hpo, freq in hpo_with_freq.items() 
                    if freq >= frequency_threshold
                }
                
                # Sort by frequency and take top N
                sorted_hpo = sorted(
                    filtered_hpo.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:max_hpo_per_disease]
                
                # Add to results
                disease_hpo = [hpo for hpo, freq in sorted_hpo]
                all_hpo.update(disease_hpo)
                
                logger.info(
                    f"Disease {disease_id}: {len(hpo_with_freq)} total HPO -> "
                    f"{len(filtered_hpo)} after frequency filter -> "
                    f"{len(disease_hpo)} after top-{max_hpo_per_disease} limit"
                )
                    
        logger.info(f"Total collected HPO codes from {len(diseases)} diseases: {len(all_hpo)}")
        return all_hpo
        
    def _get_hpo_from_lirical_api(self, disease_id: str) -> Set[str]:
        """Get HPO codes for a disease from LIRICAL API"""
        try:
            url = f"{self.lirical_url}/api/v1/disease/{disease_id}/hpo"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return set(data.get("hpo_codes", []))
            else:
                logger.debug(f"LIRICAL API returned {response.status_code} for disease {disease_id}")
                return set()
                
        except Exception as e:
            logger.debug(f"Error getting HPO from LIRICAL API: {e}")
            return set()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _select_relevant_hpo(
        self,
        epicrisis_text: str,
        existing_hpo: List[str],
        candidate_hpo: List[str],
        disease_context: List[LiricalResult]
    ) -> List[str]:
        """Use LLM to select relevant HPO codes from expanded list"""
        
        # Process in batches to handle large candidate lists
        batch_size = getattr(self.settings, 'batch_size', 80)  # Process 80 HPO codes at a time
        all_selected = []
        
        # Prepare disease context once
        disease_list = [
            {
                "id": d.disease_id,
                "name": d.disease_name,
                "probability": round(d.post_test_probability, 3)
            }
            for d in disease_context[:10]
        ]
        
        # Process candidates in batches
        for i in range(0, len(candidate_hpo), batch_size):
            batch = candidate_hpo[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} candidates")
            
            # Prepare HPO candidates with names for this batch
            candidates_with_names = []
            for hpo_id in batch:
                if hpo_id in self.hpo_loader.hpo_terms:
                    term = self.hpo_loader.hpo_terms[hpo_id]
                    candidates_with_names.append({
                        "hpo_id": hpo_id,
                        "name": term.name,
                        "synonyms": term.synonyms[:2] if term.synonyms else []
                    })
                    
            if not candidates_with_names:
                continue
                
            # Create request for this batch
            selection_request = {
                "epicrisis": epicrisis_text[:1500],  # Limit text size
                "existing_hpo": existing_hpo,
                "candidate_hpo": candidates_with_names,
                "disease_context": disease_list,
                "batch_info": f"Batch {i//batch_size + 1} of {(len(candidate_hpo) + batch_size - 1)//batch_size}"
            }
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": json.dumps(selection_request, ensure_ascii=False)}
                    ],
                    temperature=self.temperature,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Extract selected HPO IDs from this batch
                batch_selected = 0
                for item in data.get("selected_hpo", []):
                    if item.get("confidence", 0) >= 0.7:
                        all_selected.append(item["hpo_id"])
                        batch_selected += 1
                        logger.info(
                            f"Selected {item['hpo_id']} ({item.get('hpo_name', 'Unknown')}) "
                            f"with confidence {item.get('confidence', 0)}"
                        )
                        
                logger.info(f"Batch {i//batch_size + 1}: selected {batch_selected} HPO codes")
                        
            except Exception as e:
                logger.error(f"Error in HPO selection for batch {i//batch_size + 1}: {e}")
                continue
                
        logger.info(f"Total selected HPO codes from all batches: {len(all_selected)}")
        return all_selected
    
    async def _consolidate_selected_hpo(
        self,
        batch_selected_hpo: List[str],
        previous_lirical_hpo: List[str],
        epicrisis_text: str
    ) -> List[str]:
        """Consolidate HPO codes from batches and previous LIRICAL iteration"""
        
        # Combine all HPO codes (remove duplicates)
        all_candidates = list(set(batch_selected_hpo + previous_lirical_hpo))
        
        # Always consolidate - even with few candidates we need to select the most relevant
            
        logger.info(f"Consolidating {len(all_candidates)} HPO codes " +
                   f"({len(batch_selected_hpo)} from batches + {len(previous_lirical_hpo)} from previous LIRICAL)")
        
        # Prepare HPO candidates with names
        candidates_with_names = []
        for hpo_id in all_candidates:
            if hpo_id in self.hpo_loader.hpo_terms:
                term = self.hpo_loader.hpo_terms[hpo_id]
                candidates_with_names.append({
                    "hpo_id": hpo_id,
                    "name": term.name,
                    "source": "batch" if hpo_id in batch_selected_hpo else "previous_lirical"
                })
                
        # Create consolidation request
        consolidation_prompt = """You are consolidating HPO codes from recursive analysis.
        
Given HPO codes from:
1. Batch processing of new candidates (marked as 'batch')
2. Previous LIRICAL iteration (marked as 'previous_lirical')

Select the most relevant HPO codes that are clearly present in the epicrisis.
Prioritize codes that appear in both sources or have strong evidence in the text.

Return JSON with selected HPO codes and reasoning."""

        request = {
            "epicrisis": epicrisis_text[:1500],
            "candidates": candidates_with_names,
            "instruction": "Select up to 20 most relevant HPO codes that are clearly supported by the epicrisis text"
        }
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": consolidation_prompt},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)}
                ],
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Log response for debugging
            logger.debug(f"Consolidation LLM response: {json.dumps(data, ensure_ascii=False)[:500]}")
            
            # Extract consolidated HPO IDs
            consolidated = []
            # Model returns either selected_hpo or selected_hpo_codes
            selected_items = data.get("selected_hpo", data.get("selected_hpo_codes", []))
            
            for item in selected_items:
                hpo_id = item.get("hpo_id")
                if hpo_id:
                    consolidated.append(hpo_id)
                    logger.debug(f"Selected HPO {hpo_id}: {item.get('name', '')} - {item.get('reasoning', '')[:100]}")
                    
            logger.info(f"Consolidation selected {len(consolidated)} HPO codes from {len(all_candidates)} candidates")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error in HPO consolidation: {e}")
            # Fallback: return all candidates
            return all_candidates