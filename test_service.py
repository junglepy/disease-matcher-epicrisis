#!/usr/bin/env python3
"""Test epicrisis service with corrected dataset"""

import csv
import httpx
import json
from pathlib import Path
import time

def test_epicrisis(text: str, expected_omim: str, expected_hpo: list):
    """Test one epicrisis"""
    url = "http://localhost:8004/api/v1/match_epicrisis"
    
    payload = {
        "epicrisis_text": text,
        "language": "ru",
        "top_k": 10
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        result = response.json()
        
        # Check if we got results
        if result.get("results"):
            top_omim = result["results"][0].get("omim_id", "")
            matched = top_omim == expected_omim
            print(f"✓ Match: {matched} - Expected: {expected_omim}, Got: {top_omim}")
        else:
            print(f"✗ No results returned")
            
        # Check HPO codes
        extracted_hpo = result.get("extended", {}).get("hpo_codes", [])
        expected_hpo_set = set(expected_hpo)
        extracted_hpo_set = set(extracted_hpo)
        hpo_overlap = len(expected_hpo_set & extracted_hpo_set)
        print(f"  HPO overlap: {hpo_overlap}/{len(expected_hpo)} codes")
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    # Read test dataset
    dataset_path = Path("/Users/eduard_m/Yandex.Disk.localized/complex_case/test-datasets/test_epicrisis_hpo_correct_10.csv")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)[:3]  # Test first 3 cases
    
    print(f"Testing {len(test_cases)} epicrisis cases...\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Text: {case['epicrisis_ru'][:100]}...")
        
        expected_hpo = case['hpo_codes'].split(',') if case['hpo_codes'] else []
        result = test_epicrisis(
            case['epicrisis_ru'],
            case['omim_codes'],
            expected_hpo
        )
        
        time.sleep(2)  # Don't overwhelm the service


if __name__ == "__main__":
    main()