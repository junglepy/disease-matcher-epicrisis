#!/usr/bin/env python3
"""
Quick test of LIRICAL service with 30 test cases to get preliminary accuracy metrics
"""

import requests
import csv
import json
import time
from typing import List, Dict, Any

def load_test_cases(csv_file: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Load first N test cases from CSV"""
    test_cases = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
                
            # Parse HPO codes
            hpo_codes = [code.strip() for code in row['input_hpo_codes'].split(',') if code.strip()]
            
            test_cases.append({
                'test_id': row['test_id'],
                'hpo_codes': hpo_codes,
                'expected_omim': row['expected_omim'].strip(),
                'expected_mondo': row.get('expected_mondo', '').strip(),
                'coverage': row.get('coverage', '80%')
            })
    
    return test_cases

def test_lirical_case(hpo_codes: List[str], max_results: int = 10) -> Dict[str, Any]:
    """Test single case with LIRICAL"""
    url = "http://localhost:8083/api/v1/analyze"
    
    payload = {
        "hpo_codes": hpo_codes,
        "max_results": max_results
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate accuracy metrics"""
    total_cases = len(results)
    if total_cases == 0:
        return {}
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    hit_rate_10 = 0
    successful_cases = 0
    
    for result in results:
        if result.get('error'):
            continue
            
        successful_cases += 1
        expected_omim = result['expected_omim']
        predictions = result.get('predictions', [])
        
        if not predictions:
            continue
            
        # Extract OMIM IDs from predictions
        predicted_omims = []
        for pred in predictions:
            omim_id = pred.get('omim_id', '')
            if omim_id:
                predicted_omims.append(omim_id)
        
        # Check if expected OMIM is in predictions
        if expected_omim in predicted_omims:
            hit_rate_10 += 1
            
            # Find position of correct answer
            correct_position = predicted_omims.index(expected_omim) + 1
            
            if correct_position == 1:
                top1_correct += 1
            if correct_position <= 5:
                top5_correct += 1
            if correct_position <= 10:
                top10_correct += 1
    
    metrics = {
        'total_cases': total_cases,
        'successful_cases': successful_cases,
        'top1_accuracy': (top1_correct / successful_cases * 100) if successful_cases > 0 else 0,
        'top5_accuracy': (top5_correct / successful_cases * 100) if successful_cases > 0 else 0,
        'top10_accuracy': (top10_correct / successful_cases * 100) if successful_cases > 0 else 0,
        'hit_rate_10': (hit_rate_10 / successful_cases * 100) if successful_cases > 0 else 0,
        'top1_count': top1_correct,
        'top5_count': top5_correct,
        'top10_count': top10_correct,
        'hit_count': hit_rate_10
    }
    
    return metrics

def main():
    print("=== Quick LIRICAL Accuracy Test (30 cases) ===\n")
    
    # Check service health
    try:
        health = requests.get("http://localhost:8083/health", timeout=5)
        print(f"Service health: {health.json()}")
    except Exception as e:
        print(f"Service health check failed: {e}")
        return
    
    # Load test cases
    test_file = "tests/lirical_test_data_80pct.csv"
    test_cases = load_test_cases(test_file, limit=30)
    print(f"\nLoaded {len(test_cases)} test cases")
    
    # Run tests
    results = []
    start_time = time.time()
    
    for i, case in enumerate(test_cases, 1):
        print(f"Testing case {i}/30...", end="", flush=True)
        
        prediction = test_lirical_case(case['hpo_codes'], max_results=10)
        
        result = {
            'test_id': case['test_id'],
            'expected_omim': case['expected_omim'],
            'hpo_codes': case['hpo_codes'],
            'predictions': prediction.get('results', []),
            'error': prediction.get('error'),
            'processing_time': prediction.get('metadata', {}).get('processing_time_ms', 0)
        }
        
        results.append(result)
        print("✓")
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print(f"\n=== RESULTS ===")
    print(f"Total test time: {total_time:.1f}s")
    print(f"Average time per case: {total_time/30:.1f}s")
    print(f"\nTest Cases: {metrics['total_cases']}")
    print(f"Successful: {metrics['successful_cases']}")
    print(f"Failed: {metrics['total_cases'] - metrics['successful_cases']}")
    
    print(f"\n=== ACCURACY METRICS ===")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.1f}% ({metrics['top1_count']}/{metrics['successful_cases']})")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.1f}% ({metrics['top5_count']}/{metrics['successful_cases']})")
    print(f"Top-10 Accuracy: {metrics['top10_accuracy']:.1f}% ({metrics['top10_count']}/{metrics['successful_cases']})")
    print(f"Hit Rate @ 10: {metrics['hit_rate_10']:.1f}% ({metrics['hit_count']}/{metrics['successful_cases']})")
    
    # Show some examples
    print(f"\n=== SAMPLE RESULTS ===")
    for i, result in enumerate(results[:3]):
        if result.get('error'):
            print(f"Case {i+1}: ERROR - {result['error']}")
        else:
            predictions = result.get('predictions', [])
            top_pred = predictions[0] if predictions else None
            expected = result['expected_omim']
            
            if top_pred:
                top_omim = top_pred.get('omim_id', 'N/A')
                confidence = top_pred.get('post_test_probability', 0)
                correct = "✓" if top_omim == expected else "✗"
                print(f"Case {i+1}: Expected {expected} → Predicted {top_omim} ({confidence:.4f}) {correct}")
            else:
                print(f"Case {i+1}: Expected {expected} → No predictions")
    
    # Save results
    with open('quick_test_results_30.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
            'test_info': {
                'total_time': total_time,
                'test_cases': len(test_cases),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: quick_test_results_30.json")

if __name__ == "__main__":
    main()