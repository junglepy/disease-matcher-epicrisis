#!/usr/bin/env python3
"""
Phenopacket Builder for LIRICAL

Creates phenopacket format JSON from HPO codes list
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PhenopacketBuilder:
    """Build phenopacket format for LIRICAL input"""
    
    def __init__(self, hpo_labels: Optional[Dict[str, str]] = None):
        """
        Initialize builder
        
        Args:
            hpo_labels: Optional dictionary mapping HPO IDs to labels
        """
        self.hpo_labels = hpo_labels or {}
        
    def get_hpo_label(self, hpo_id: str) -> str:
        """Get label for HPO term, fallback to ID if not found"""
        return self.hpo_labels.get(hpo_id, hpo_id)
        
    def create_phenopacket(self, 
                          hpo_codes: List[str],
                          patient_id: Optional[str] = None,
                          age_years: Optional[int] = None,
                          sex: Optional[str] = None) -> Dict:
        """
        Create phenopacket from HPO codes
        
        Args:
            hpo_codes: List of HPO IDs (e.g., ["HP:0001263", "HP:0001250"])
            patient_id: Optional patient identifier
            age_years: Optional patient age in years
            sex: Optional sex ("MALE", "FEMALE", "UNKNOWN")
            
        Returns:
            Phenopacket as dictionary
        """
        # Generate ID if not provided
        if not patient_id:
            patient_id = f"patient_{uuid.uuid4().hex[:8]}"
            
        # Create base structure
        phenopacket = {
            "id": patient_id,
            "subject": {
                "id": "proband",
            },
            "phenotypicFeatures": [],
            "metaData": {
                "created": datetime.utcnow().isoformat() + "Z",
                "createdBy": "LIRICAL-service",
                "resources": [
                    {
                        "id": "hp",
                        "name": "Human Phenotype Ontology",
                        "url": "http://purl.obolibrary.org/obo/hp.owl",
                        "version": "2024-01-01",  # Update as needed
                        "namespacePrefix": "HP"
                    }
                ]
            }
        }
        
        # Add age if provided
        if age_years is not None:
            phenopacket["subject"]["timeAtLastEncounter"] = {
                "age": f"P{age_years}Y"  # ISO 8601 duration
            }
            
        # Add sex if provided
        if sex:
            # Normalize sex value
            sex_map = {
                "male": "MALE",
                "m": "MALE", 
                "female": "FEMALE",
                "f": "FEMALE",
                "unknown": "UNKNOWN_SEX",
                "u": "UNKNOWN_SEX"
            }
            normalized_sex = sex_map.get(sex.lower(), "UNKNOWN_SEX")
            phenopacket["subject"]["sex"] = normalized_sex
        
        # Add phenotypic features
        for hpo_id in hpo_codes:
            if not hpo_id.startswith("HP:"):
                logger.warning(f"Invalid HPO ID format: {hpo_id}")
                continue
                
            feature = {
                "type": {
                    "id": hpo_id,
                    "label": self.get_hpo_label(hpo_id)
                },
                "excluded": False  # All features are observed (not negated)
            }
            phenopacket["phenotypicFeatures"].append(feature)
            
        return phenopacket
        
    def create_minimal_phenopacket(self, hpo_codes: List[str]) -> Dict:
        """
        Create minimal phenopacket with just HPO codes
        
        This is the simplest format that LIRICAL accepts
        """
        return {
            "subject": {},
            "phenotypicFeatures": [
                {"type": {"id": hpo_id}}
                for hpo_id in hpo_codes
                if hpo_id.startswith("HP:")
            ]
        }
        
    def save_phenopacket(self, phenopacket: Dict, file_path: str):
        """Save phenopacket to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(phenopacket, f, indent=2, ensure_ascii=False)
            
    def validate_phenopacket(self, phenopacket: Dict) -> bool:
        """
        Basic validation of phenopacket structure
        
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if "phenotypicFeatures" not in phenopacket:
            logger.error("Missing required field: phenotypicFeatures")
            return False
            
        # Check that we have at least one phenotype
        if not phenopacket["phenotypicFeatures"]:
            logger.error("No phenotypic features provided")
            return False
            
        # Check phenotype structure
        for feature in phenopacket["phenotypicFeatures"]:
            if "type" not in feature:
                logger.error("Feature missing 'type' field")
                return False
            if "id" not in feature["type"]:
                logger.error("Feature type missing 'id' field")
                return False
                
        return True


# Example usage and testing
if __name__ == "__main__":
    # Test the builder
    builder = PhenopacketBuilder()
    
    # Test case 1: Minimal phenopacket
    hpo_codes = ["HP:0001263", "HP:0001250", "HP:0000252"]
    minimal = builder.create_minimal_phenopacket(hpo_codes)
    print("Minimal phenopacket:")
    print(json.dumps(minimal, indent=2))
    
    # Test case 2: Full phenopacket
    full = builder.create_phenopacket(
        hpo_codes=hpo_codes,
        patient_id="test_001",
        age_years=5,
        sex="male"
    )
    print("\nFull phenopacket:")
    print(json.dumps(full, indent=2))
    
    # Validate
    print(f"\nValidation: {builder.validate_phenopacket(full)}")