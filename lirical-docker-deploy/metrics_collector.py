#!/usr/bin/env python3
"""
Metrics collection for LIRICAL service

Tracks:
- Performance metrics (latency, throughput)
- Accuracy metrics (top-k accuracy)
- Resource usage (memory, CPU)
- Error rates
"""

import time
import json
import psutil
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional
import threading
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics for LIRICAL service"""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            window_size: Number of recent requests to keep for metrics
        """
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # Performance metrics
        self.response_times = deque(maxlen=window_size)
        self.hpo_counts = deque(maxlen=window_size)
        self.result_counts = deque(maxlen=window_size)
        
        # Accuracy metrics
        self.accuracy_results = {
            'top1': deque(maxlen=window_size),
            'top5': deque(maxlen=window_size),
            'top10': deque(maxlen=window_size)
        }
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.successful_requests = 0
        
        # Resource metrics
        self.resource_samples = deque(maxlen=100)
        self.start_resource_monitoring()
        
        # Benchmark results
        self.benchmark_results = []
        
    def start_resource_monitoring(self):
        """Start background thread for resource monitoring"""
        def monitor():
            while True:
                try:
                    # Get current process and children
                    process = psutil.Process()
                    children = process.children(recursive=True)
                    
                    # Calculate total resource usage
                    cpu_percent = process.cpu_percent(interval=1)
                    memory_info = process.memory_info()
                    
                    # Add children resources
                    for child in children:
                        try:
                            cpu_percent += child.cpu_percent(interval=0)
                            child_mem = child.memory_info()
                            memory_info = type(memory_info)(
                                rss=memory_info.rss + child_mem.rss,
                                vms=memory_info.vms + child_mem.vms
                            )
                        except:
                            pass
                    
                    sample = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_info.rss / 1024 / 1024,
                        'memory_percent': process.memory_percent()
                    }
                    
                    with self.lock:
                        self.resource_samples.append(sample)
                        
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    
                time.sleep(10)  # Sample every 10 seconds
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        
    def record_request(self,
                      response_time: float,
                      hpo_count: int,
                      result_count: int,
                      error: Optional[str] = None):
        """
        Record metrics for a single request
        
        Args:
            response_time: Response time in milliseconds
            hpo_count: Number of HPO codes in request
            result_count: Number of results returned
            error: Error message if request failed
        """
        with self.lock:
            self.total_requests += 1
            
            if error:
                self.error_counts[error] += 1
            else:
                self.successful_requests += 1
                self.response_times.append(response_time)
                self.hpo_counts.append(hpo_count)
                self.result_counts.append(result_count)
                
    def record_accuracy(self,
                       expected_omim: str,
                       results: List[Dict],
                       test_id: Optional[str] = None):
        """
        Record accuracy metrics for a test case
        
        Args:
            expected_omim: Expected OMIM ID
            results: List of disease predictions
            test_id: Optional test case identifier
        """
        with self.lock:
            # Extract OMIM IDs from results
            result_omim_ids = []
            for r in results:
                omim_id = r.get('omim_id', '')
                if not omim_id and 'disease_id' in r:
                    # Try to extract from disease_id
                    if r['disease_id'].startswith('OMIM:'):
                        omim_id = r['disease_id'].replace('OMIM:', '')
                if omim_id:
                    result_omim_ids.append(omim_id)
                    
            # Check accuracy at different k values
            for k, queue in self.accuracy_results.items():
                k_value = int(k.replace('top', ''))
                is_correct = expected_omim in result_omim_ids[:k_value]
                queue.append(1 if is_correct else 0)
                
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        with self.lock:
            if not self.response_times:
                return {
                    'requests_total': self.total_requests,
                    'requests_successful': self.successful_requests,
                    'error_rate': 0.0
                }
                
            response_times = list(self.response_times)
            
            return {
                'requests_total': self.total_requests,
                'requests_successful': self.successful_requests,
                'error_rate': 1 - (self.successful_requests / max(1, self.total_requests)),
                'response_time': {
                    'mean': np.mean(response_times),
                    'p50': np.percentile(response_times, 50),
                    'p95': np.percentile(response_times, 95),
                    'p99': np.percentile(response_times, 99),
                    'min': np.min(response_times),
                    'max': np.max(response_times)
                },
                'hpo_count': {
                    'mean': np.mean(list(self.hpo_counts)) if self.hpo_counts else 0,
                    'max': max(self.hpo_counts) if self.hpo_counts else 0
                }
            }
            
    def get_accuracy_metrics(self) -> Dict:
        """Get current accuracy metrics"""
        with self.lock:
            metrics = {}
            
            for k, queue in self.accuracy_results.items():
                if queue:
                    accuracy = sum(queue) / len(queue)
                    metrics[f'{k}_accuracy'] = accuracy
                    metrics[f'{k}_count'] = len(queue)
                else:
                    metrics[f'{k}_accuracy'] = 0.0
                    metrics[f'{k}_count'] = 0
                    
            return metrics
            
    def get_resource_metrics(self) -> Dict:
        """Get current resource usage metrics"""
        with self.lock:
            if not self.resource_samples:
                return {}
                
            recent_samples = list(self.resource_samples)[-10:]  # Last 10 samples
            
            return {
                'cpu_percent': {
                    'current': recent_samples[-1]['cpu_percent'],
                    'mean': np.mean([s['cpu_percent'] for s in recent_samples]),
                    'max': max(s['cpu_percent'] for s in recent_samples)
                },
                'memory_mb': {
                    'current': recent_samples[-1]['memory_mb'],
                    'mean': np.mean([s['memory_mb'] for s in recent_samples]),
                    'max': max(s['memory_mb'] for s in recent_samples)
                }
            }
            
    def get_error_summary(self) -> Dict:
        """Get error counts by type"""
        with self.lock:
            return dict(self.error_counts)
            
    def get_all_metrics(self) -> Dict:
        """Get all metrics in a single call"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'performance': self.get_performance_metrics(),
            'accuracy': self.get_accuracy_metrics(),
            'resources': self.get_resource_metrics(),
            'errors': self.get_error_summary()
        }
        
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        metrics = self.get_all_metrics()
        lines = []
        
        # Helper to add metric
        def add_metric(name, value, labels=None):
            if labels:
                label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f'{name}{{{label_str}}} {value}')
            else:
                lines.append(f'{name} {value}')
                
        # Performance metrics
        perf = metrics['performance']
        add_metric('lirical_requests_total', perf['requests_total'])
        add_metric('lirical_requests_successful', perf['requests_successful'])
        add_metric('lirical_error_rate', perf['error_rate'])
        
        if 'response_time' in perf:
            for stat, value in perf['response_time'].items():
                add_metric(f'lirical_response_time_{stat}_ms', value)
                
        # Accuracy metrics
        for metric, value in metrics['accuracy'].items():
            if metric.endswith('_accuracy'):
                add_metric(f'lirical_{metric}', value)
                
        # Resource metrics
        if 'cpu_percent' in metrics['resources']:
            for stat, value in metrics['resources']['cpu_percent'].items():
                add_metric(f'lirical_cpu_{stat}_percent', value)
                
        if 'memory_mb' in metrics['resources']:
            for stat, value in metrics['resources']['memory_mb'].items():
                add_metric(f'lirical_memory_{stat}_mb', value)
                
        # Error metrics
        for error_type, count in metrics['errors'].items():
            add_metric('lirical_errors_total', count, {'type': error_type})
            
        return '\n'.join(lines)
        
    def run_benchmark(self, test_file: str) -> Dict:
        """
        Run benchmark on test dataset
        
        Args:
            test_file: Path to CSV test file
            
        Returns:
            Benchmark results
        """
        import csv
        import requests
        
        results = {
            'test_file': test_file,
            'start_time': datetime.utcnow().isoformat(),
            'test_cases': 0,
            'successful': 0,
            'failed': 0,
            'accuracy': {},
            'performance': []
        }
        
        # Reset accuracy metrics for clean benchmark
        with self.lock:
            for queue in self.accuracy_results.values():
                queue.clear()
                
        # Load test data
        test_cases = []
        with open(test_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'test_id': row['test_id'],
                    'hpo_codes': row['input_hpo_codes'].split(','),
                    'expected_omim': row['expected_omim'],
                    'coverage': float(row['coverage'])
                })
                
        results['test_cases'] = len(test_cases)
        
        # Run tests
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                
                # Call API
                response = requests.post(
                    'http://localhost:8083/api/v1/analyze',
                    json={
                        'hpo_codes': test_case['hpo_codes'],
                        'max_results': 20
                    },
                    timeout=30
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    results['successful'] += 1
                    
                    # Record metrics
                    self.record_request(
                        elapsed_ms,
                        len(test_case['hpo_codes']),
                        len(data['results'])
                    )
                    
                    self.record_accuracy(
                        test_case['expected_omim'],
                        data['results'],
                        test_case['test_id']
                    )
                    
                    results['performance'].append({
                        'test_id': test_case['test_id'],
                        'response_time_ms': elapsed_ms,
                        'coverage': test_case['coverage']
                    })
                else:
                    results['failed'] += 1
                    self.record_request(
                        elapsed_ms,
                        len(test_case['hpo_codes']),
                        0,
                        f"HTTP {response.status_code}"
                    )
                    
            except Exception as e:
                results['failed'] += 1
                logger.error(f"Benchmark error on {test_case['test_id']}: {e}")
                
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Benchmark progress: {i + 1}/{len(test_cases)}")
                
        # Get final accuracy metrics
        results['accuracy'] = self.get_accuracy_metrics()
        results['end_time'] = datetime.utcnow().isoformat()
        
        # Store benchmark results
        with self.lock:
            self.benchmark_results.append(results)
            
        return results


# Global metrics instance
metrics = MetricsCollector()


# Decorator for automatic metric collection
def track_metrics(func):
    """Decorator to automatically track metrics for API endpoints"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Try to extract request info
            from flask import request
            try:
                data = request.get_json()
                hpo_count = len(data.get('hpo_codes', []))
                result_count = len(result[0].get('results', [])) if not error else 0
            except:
                hpo_count = 0
                result_count = 0
                
            metrics.record_request(elapsed_ms, hpo_count, result_count, error)
            
    return wrapper