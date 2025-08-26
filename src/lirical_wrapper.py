"""
LIRICAL service wrapper for HTTP calls
"""

import json
import logging
import requests
from typing import List, Optional
from .models import LiricalResult
from .config import Settings

logger = logging.getLogger(__name__)


class LiricalWrapper:
    """Wrapper for LIRICAL service HTTP calls"""
    
    def __init__(self, settings: Settings = None, hpo_loader=None):
        """
        Initialize LIRICAL wrapper
        
        Args:
            settings: Application settings
            hpo_loader: HPO loader for disease name lookup
        """
        if settings is None:
            settings = Settings()
        
        self.settings = settings
        self.hpo_loader = hpo_loader
        # Get LIRICAL service URL from settings or use default
        self.lirical_url = getattr(settings, 'lirical_service_url', 'http://localhost:8083')
        self.timeout = getattr(settings, 'lirical_timeout', 30)
        
        logger.info(f"LIRICAL wrapper initialized with URL: {self.lirical_url}")
    
    def analyze_phenotypes(self, hpo_codes: List[str], top_k: int = 10) -> List[LiricalResult]:
        """
        Analyze HPO codes using LIRICAL service
        
        Args:
            hpo_codes: List of HPO IDs (e.g., ["HP:0001249", "HP:0000252"])
            top_k: Number of top results to return
            
        Returns:
            List of LIRICAL results with probabilities
        """
        if not hpo_codes:
            logger.warning("No HPO codes provided for LIRICAL analysis")
            return []
        
        logger.info(f"Calling LIRICAL service for {len(hpo_codes)} HPO codes: {hpo_codes}")
        
        try:
            # Call LIRICAL service
            url = f"{self.lirical_url}/api/v1/analyze"
            payload = {
                "hpo_codes": hpo_codes,
                "max_results": top_k
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"LIRICAL service error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            results = []
            
            # Parse LIRICAL response
            for item in data.get("results", []):
                disease_id = item.get("disease_id", "")
                disease_name = item.get("disease_name", "")
                
                # Используем правильное название из нашей базы данных
                if self.hpo_loader and disease_id.startswith("OMIM:"):
                    real_name = self.hpo_loader.get_disease_name(disease_id)
                    if real_name and not real_name.startswith("Unknown"):
                        logger.debug(f"Replacing '{disease_name}' with '{real_name}' for {disease_id}")
                        disease_name = real_name
                    else:
                        logger.debug(f"No name found for {disease_id}, keeping '{disease_name}'")
                
                result = LiricalResult(
                    disease_id=disease_id,
                    disease_name=disease_name,
                    post_test_probability=item.get("post_test_probability", 0.0),
                    maximum_likelihood_ratio=item.get("composite_lr", 0.0),
                    combined_likelihood_ratio=item.get("composite_lr", 0.0)
                )
                results.append(result)
            
            logger.info(f"LIRICAL analysis complete, returning {len(results)} results")
            
            # Log top result for debugging
            if results:
                top = results[0]
                logger.info(
                    f"Top result: {top.disease_id} ({top.disease_name}) "
                    f"with probability {top.post_test_probability:.3f}"
                )
            
            return results
            
        except requests.exceptions.Timeout:
            logger.error(f"LIRICAL service timeout after {self.timeout}s")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LIRICAL service at {self.lirical_url}")
            return []
        except Exception as e:
            logger.error(f"LIRICAL service error: {e}")
            return []