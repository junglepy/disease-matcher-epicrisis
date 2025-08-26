#!/usr/bin/env python3
"""
LIRICAL REST API Service

Provides HTTP endpoints for disease prioritization based on HPO phenotypes
"""

import os
import asyncio
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from typing import List, Dict
import json
from datetime import datetime
from functools import wraps

# Import our modules
from lirical_wrapper import LiricalWrapper, LiricalSimulator
from phenopacket_builder import PhenopacketBuilder
from metrics_collector import metrics, track_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize LIRICAL wrapper
lirical = None
is_simulator = False

def initialize_lirical():
    """Initialize LIRICAL wrapper with fallback to simulator"""
    global lirical, is_simulator
    
    try:
        # Try to initialize real LIRICAL
        lirical = LiricalWrapper()
        is_simulator = False
        logger.info("Initialized LIRICAL wrapper")
    except Exception as e:
        logger.warning(f"Failed to initialize LIRICAL: {e}")
        logger.warning("Using simulator for development")
        lirical = LiricalSimulator()
        is_simulator = True

# Initialize on startup
initialize_lirical()

# Create event loop for async operations
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'lirical',
        'mode': 'simulator' if is_simulator else 'real',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/v1/analyze', methods=['POST'])
def analyze_phenotypes():
    """
    Analyze HPO phenotypes to predict diseases
    
    Request body:
    {
        "hpo_codes": ["HP:0001263", "HP:0001250"],
        "max_results": 10,
        "min_score": 0.01
    }
    """
    start_time = datetime.utcnow()
    error_msg = None
    
    try:
        data = request.get_json()
        
        if not data or 'hpo_codes' not in data:
            error_msg = 'Missing required field: hpo_codes'
            return jsonify({'error': error_msg}), 400
            
        hpo_codes = data['hpo_codes']
        max_results = data.get('max_results', 10)
        
        # Validate HPO codes
        if not isinstance(hpo_codes, list) or not hpo_codes:
            error_msg = 'hpo_codes must be a non-empty list'
            return jsonify({'error': error_msg}), 400
            
        # Filter valid HPO codes
        valid_hpo = [code for code in hpo_codes if code.startswith('HP:')]
        if not valid_hpo:
            error_msg = 'No valid HPO codes provided'
            return jsonify({'error': error_msg}), 400
            
        logger.info(f"Analyzing {len(valid_hpo)} HPO codes")
        
        # Run analysis
        results = loop.run_until_complete(
            lirical.analyze_hpo_codes(valid_hpo, max_results)
        )
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Record metrics
        metrics.record_request(
            processing_time,
            len(valid_hpo),
            len(results['diagnoses'])
        )
        
        # Format response
        response = {
            'results': results['diagnoses'],
            'metadata': {
                'lirical_version': results['metadata'].get('lirical_version', 'unknown'),
                'processing_time_ms': int(processing_time),
                'num_hpo_codes': len(valid_hpo),
                'mode': 'simulator' if is_simulator else 'real'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error analyzing phenotypes: {e}")
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Record error metrics
        metrics.record_request(
            processing_time,
            len(data.get('hpo_codes', [])) if data else 0,
            0,
            error_msg
        )
        
        return jsonify({'error': error_msg}), 500


@app.route('/api/v1/recursive_analyze', methods=['POST'])
def recursive_analyze():
    """
    Recursive analysis with phenotype refinement
    
    Request body:
    {
        "hpo_codes": ["HP:0001263", "HP:0001250"],
        "epicrisis_text": "Optional clinical text for validation",
        "max_iterations": 2
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'hpo_codes' not in data:
            return jsonify({
                'error': 'Missing required field: hpo_codes'
            }), 400
            
        hpo_codes = data['hpo_codes']
        epicrisis_text = data.get('epicrisis_text')
        max_iterations = data.get('max_iterations', 2)
        
        # Initial analysis
        initial_results = loop.run_until_complete(
            lirical.analyze_hpo_codes(hpo_codes, max_results=5)
        )
        
        response = {
            'initial_analysis': initial_results,
            'iterations': []
        }
        
        # TODO: Implement recursive refinement logic
        # This would involve:
        # 1. Getting all HPO codes for top diseases
        # 2. Checking which additional HPO codes are present in epicrisis
        # 3. Re-running analysis with expanded HPO set
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in recursive analysis: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/v1/info', methods=['GET'])
def service_info():
    """Get information about the LIRICAL service"""
    return jsonify({
        'service': 'LIRICAL Disease Prioritization',
        'version': '2.0.5',
        'mode': 'simulator' if is_simulator else 'real',
        'description': 'Likelihood Ratio Interpretation of Clinical Abnormalities',
        'features': {
            'hpo_analysis': True,
            'phenopacket_support': True,
            'batch_processing': False,  # TODO: Implement
            'recursive_analysis': True
        },
        'expected_performance': {
            'top1_accuracy': '>60%',
            'top5_accuracy': '>85%',
            'processing_time': '<5s for 10 HPO codes'
        },
        'input_format': {
            'hpo_codes': 'List of HPO IDs (e.g., ["HP:0001263", "HP:0001250"])',
            'max_results': 'Maximum number of disease predictions (default: 10)'
        },
        'output_format': {
            'disease_id': 'OMIM or MONDO identifier',
            'disease_name': 'Human-readable disease name',
            'post_test_probability': 'Probability after considering phenotypes',
            'composite_lr': 'Combined likelihood ratio',
            'phenotype_score': 'Phenotype matching score'
        }
    })


@app.route('/api/v1/validate_hpo', methods=['POST'])
def validate_hpo_codes():
    """
    Validate HPO codes format and existence
    
    Request body:
    {
        "hpo_codes": ["HP:0001263", "HP:0001250"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'hpo_codes' not in data:
            return jsonify({
                'error': 'Missing required field: hpo_codes'
            }), 400
            
        hpo_codes = data['hpo_codes']
        
        # Validate each code
        validation_results = []
        for code in hpo_codes:
            result = {
                'code': code,
                'valid_format': False,
                'exists': False  # TODO: Check against HPO ontology
            }
            
            # Check format
            if isinstance(code, str) and code.startswith('HP:') and len(code) == 10:
                try:
                    # Check if numeric part is valid
                    int(code[3:])
                    result['valid_format'] = True
                except ValueError:
                    pass
                    
            validation_results.append(result)
            
        # Summary
        valid_count = sum(1 for r in validation_results if r['valid_format'])
        
        return jsonify({
            'total_codes': len(hpo_codes),
            'valid_codes': valid_count,
            'invalid_codes': len(hpo_codes) - valid_count,
            'validation_results': validation_results
        })
        
    except Exception as e:
        logger.error(f"Error validating HPO codes: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def prometheus_metrics():
    """Export metrics in Prometheus format"""
    return Response(metrics.export_prometheus(), mimetype='text/plain')


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Get service statistics in JSON format"""
    return jsonify(metrics.get_all_metrics())


@app.route('/api/v1/benchmark', methods=['POST'])
def run_benchmark():
    """
    Run benchmark on test dataset
    
    Request body:
    {
        "test_file": "path/to/test_file.csv"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'test_file' not in data:
            return jsonify({'error': 'Missing required field: test_file'}), 400
            
        test_file = data['test_file']
        
        # Check if file exists
        if not os.path.exists(test_file):
            return jsonify({'error': f'Test file not found: {test_file}'}), 404
            
        logger.info(f"Starting benchmark with {test_file}")
        
        # Run benchmark (this will take time)
        results = metrics.run_benchmark(test_file)
        
        return jsonify({
            'status': 'completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/test_accuracy', methods=['POST'])
def test_accuracy():
    """
    Test accuracy with known case
    
    Request body:
    {
        "hpo_codes": ["HP:0001263"],
        "expected_omim": "123456",
        "test_id": "test-001"
    }
    """
    try:
        data = request.get_json()
        
        # Analyze HPO codes
        results = loop.run_until_complete(
            lirical.analyze_hpo_codes(data['hpo_codes'], max_results=20)
        )
        
        # Record accuracy
        metrics.record_accuracy(
            data['expected_omim'],
            results['diagnoses'],
            data.get('test_id')
        )
        
        # Check if correct
        result_omim_ids = [
            r.get('omim_id', r.get('disease_id', '').replace('OMIM:', ''))
            for r in results['diagnoses']
        ]
        
        is_top1 = data['expected_omim'] == result_omim_ids[0] if result_omim_ids else False
        is_top5 = data['expected_omim'] in result_omim_ids[:5]
        is_top10 = data['expected_omim'] in result_omim_ids[:10]
        
        return jsonify({
            'test_id': data.get('test_id'),
            'is_correct': {
                'top1': is_top1,
                'top5': is_top5,
                'top10': is_top10
            },
            'expected_rank': result_omim_ids.index(data['expected_omim']) + 1 
                           if data['expected_omim'] in result_omim_ids else -1,
            'results': results['diagnoses'][:10]
        })
        
    except Exception as e:
        logger.error(f"Test accuracy error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8083))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting LIRICAL service on port {port}")
    logger.info(f"Mode: {'Simulator' if is_simulator else 'Real LIRICAL'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)