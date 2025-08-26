#!/usr/bin/env python3
"""
LIRICAL Wrapper

Python wrapper for LIRICAL Java application
"""

import os
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import asyncio
import aiofiles
from datetime import datetime

logger = logging.getLogger(__name__)


class LiricalWrapper:
    """Wrapper for LIRICAL command-line tool"""
    
    def __init__(self, 
                 lirical_jar: str = None,
                 data_dir: str = None,
                 temp_dir: str = "/tmp/lirical"):
        """
        Initialize LIRICAL wrapper
        
        Args:
            lirical_jar: Path to LIRICAL JAR file
            data_dir: Path to LIRICAL data directory
            temp_dir: Directory for temporary files
        """
        self.lirical_jar = lirical_jar or os.getenv('LIRICAL_JAR', 'lirical-cli-2.2.0.jar')
        self.data_dir = data_dir or os.getenv('LIRICAL_DATA', '/data')
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check if LIRICAL is available
        self._check_lirical()
        
    def _check_lirical(self):
        """Check if LIRICAL JAR exists and is executable"""
        if not os.path.exists(self.lirical_jar):
            raise FileNotFoundError(f"LIRICAL JAR not found: {self.lirical_jar}")
            
        # Try to run help command
        try:
            result = subprocess.run(
                ['java', '-jar', self.lirical_jar, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning(f"LIRICAL help command failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to run LIRICAL: {e}")
            
    async def analyze_phenopacket(self, 
                                 phenopacket_path: str,
                                 output_prefix: str = None) -> Dict:
        """
        Run LIRICAL analysis on a phenopacket file
        
        Args:
            phenopacket_path: Path to phenopacket JSON file
            output_prefix: Prefix for output files (optional)
            
        Returns:
            Dictionary with analysis results
        """
        if not output_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = self.temp_dir / f"lirical_{timestamp}"
            
        # Prepare command
        cmd = [
            'java', '-Xmx2g', '-Xms1g',
            '-jar', self.lirical_jar,
            'phenopacket',
            '--phenopacket', phenopacket_path,
            '--data', self.data_dir,
            '--prefix', str(output_prefix),
            '--output-format', 'json'  # Request JSON output
        ]
        
        logger.info(f"Running LIRICAL command: {' '.join(cmd)}")
        
        # Run LIRICAL
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"LIRICAL failed: {stderr.decode()}")
                raise RuntimeError(f"LIRICAL analysis failed: {stderr.decode()}")
                
            # Parse results
            json_output = f"{output_prefix}.json"
            if os.path.exists(json_output):
                async with aiofiles.open(json_output, 'r') as f:
                    results = json.loads(await f.read())
                    
                # Clean up temporary files
                for ext in ['.json', '.html', '.tsv']:
                    file_path = f"{output_prefix}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
                return self._parse_lirical_results(results)
            else:
                logger.error(f"LIRICAL output not found: {json_output}")
                raise FileNotFoundError(f"LIRICAL output not found: {json_output}")
                
        except asyncio.TimeoutError:
            logger.error("LIRICAL analysis timed out")
            raise TimeoutError("LIRICAL analysis timed out")
            
    def _parse_lirical_results(self, raw_results: Dict) -> Dict:
        """
        Parse LIRICAL JSON output into standardized format
        
        Args:
            raw_results: Raw JSON output from LIRICAL
            
        Returns:
            Standardized results dictionary
        """
        # Extract differential diagnoses
        diagnoses = []
        
        # LIRICAL 2.2.0 output structure uses 'analysisResults'
        if 'analysisResults' in raw_results:
            diff_diagnosis = raw_results['analysisResults']
        elif 'differentialDiagnosis' in raw_results:
            diff_diagnosis = raw_results['differentialDiagnosis']
        elif 'results' in raw_results:
            diff_diagnosis = raw_results['results']
        else:
            logger.warning("Could not find analysis results in LIRICAL output")
            logger.debug(f"Available keys in LIRICAL output: {list(raw_results.keys())}")
            diff_diagnosis = []
            
        # Sort results by posttestProbability (highest first)
        if diff_diagnosis:
            try:
                diff_diagnosis = sorted(diff_diagnosis, 
                                      key=lambda x: x.get('posttestProbability', 0.0), 
                                      reverse=True)
            except Exception as e:
                logger.warning(f"Could not sort results: {e}")
            
        for i, diagnosis in enumerate(diff_diagnosis[:20]):  # Top 20 results
            # Extract disease information
            disease_id = diagnosis.get('diseaseId', '')
            disease_name = self._get_disease_name(disease_id)  # Get name from disease ID
            
            disease_info = {
                'rank': i + 1,
                'disease_id': disease_id,
                'disease_name': disease_name,
                'post_test_probability': diagnosis.get('posttestProbability', 0.0),
                'pretest_probability': diagnosis.get('pretestProbability', 0.0),
                'composite_lr': diagnosis.get('compositeLR', 0.0),
                'phenotype_score': diagnosis.get('posttestProbability', 0.0)  # Use posttest as phenotype score
            }
            
            # Try to extract OMIM/MONDO IDs
            if disease_id.startswith('OMIM:'):
                disease_info['omim_id'] = disease_id.replace('OMIM:', '')
            elif disease_id.startswith('MONDO:'):
                disease_info['mondo_id'] = disease_id
                
            diagnoses.append(disease_info)
            
        # Extract metadata from new structure
        metadata = raw_results.get('analysisMetadata', {})
        
        # Build standardized response
        return {
            'diagnoses': diagnoses,
            'metadata': {
                'lirical_version': metadata.get('liricalVersion', 'unknown'),
                'analysis_date': metadata.get('analysisDate', datetime.utcnow().isoformat()),
                'hpo_version': metadata.get('hpoVersion', 'unknown'),
                'transcript_database': metadata.get('transcriptDatabase', 'unknown'),
                'num_diseases_analyzed': len(diff_diagnosis)
            }
        }
        
    def _get_disease_name(self, disease_id: str) -> str:
        """
        Get disease name from disease ID
        
        Args:
            disease_id: Disease ID (e.g., 'OMIM:212050')
            
        Returns:
            Disease name or fallback string
        """
        # Simple disease name mapping (could be expanded with real database)
        disease_names = {
            'OMIM:212050': 'Jervell and Lange-Nielsen syndrome',
            'OMIM:248000': 'Ataxia-telangiectasia',
            'OMIM:617895': 'Neurodevelopmental disorder',
            'OMIM:616564': 'Intellectual disability',
            'OMIM:615233': 'Developmental delay',
            'OMIM:209850': 'Autism spectrum disorder',
            'OMIM:100800': 'Achondroplasia',
            'OMIM:102900': 'Ehlers-Danlos syndrome',
            'OMIM:107250': 'Albinism'
        }
        
        # Try to get name from mapping
        if disease_id in disease_names:
            return disease_names[disease_id]
            
        # Extract ID and create descriptive name
        if disease_id.startswith('OMIM:'):
            omim_id = disease_id.replace('OMIM:', '')
            return f"OMIM disease {omim_id}"
        elif disease_id.startswith('MONDO:'):
            return f"MONDO disease {disease_id}"
        else:
            return f"Disease {disease_id}"
        
    async def analyze_hpo_codes(self, 
                              hpo_codes: List[str],
                              max_results: int = 10) -> Dict:
        """
        Convenience method to analyze HPO codes directly
        
        Args:
            hpo_codes: List of HPO IDs
            max_results: Maximum number of results to return
            
        Returns:
            Analysis results
        """
        from phenopacket_builder import PhenopacketBuilder
        
        # Create temporary phenopacket
        builder = PhenopacketBuilder()
        phenopacket = builder.create_minimal_phenopacket(hpo_codes)
        
        # Save to temporary file
        temp_file = self.temp_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.json"
        builder.save_phenopacket(phenopacket, str(temp_file))
        
        try:
            # Run analysis
            results = await self.analyze_phenopacket(str(temp_file))
            
            # Limit results
            if 'diagnoses' in results:
                results['diagnoses'] = results['diagnoses'][:max_results]
                
            return results
            
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()


class LiricalSimulator:
    """Simulator for testing without actual LIRICAL installation"""
    
    def __init__(self):
        logger.warning("Using LIRICAL simulator - results are mock data!")
        
    async def analyze_hpo_codes(self, 
                              hpo_codes: List[str],
                              max_results: int = 10) -> Dict:
        """Simulate LIRICAL analysis"""
        # Simple simulation based on HPO codes
        # In reality, LIRICAL uses complex likelihood calculations
        
        # Mock disease mapping (very simplified)
        mock_diseases = {
            frozenset(['HP:0001263', 'HP:0001250']): {
                'id': 'OMIM:209850',
                'name': 'Autism spectrum disorder',
                'mondo': 'MONDO:0005260'
            },
            frozenset(['HP:0000252', 'HP:0002510']): {
                'id': 'OMIM:100800',
                'name': 'Achondroplasia',
                'mondo': 'MONDO:0007037'
            },
            frozenset(['HP:0001249', 'HP:0001250']): {
                'id': 'OMIM:300486',
                'name': 'Intellectual disability',
                'mondo': 'MONDO:0001071'
            }
        }
        
        # Find best matches
        input_set = frozenset(hpo_codes[:3])  # Use first 3 for matching
        matches = []
        
        for disease_hpo, disease_info in mock_diseases.items():
            overlap = len(input_set & disease_hpo)
            if overlap > 0:
                score = overlap / len(disease_hpo)
                matches.append((score, disease_info))
                
        # Sort by score
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Build results
        diagnoses = []
        for i, (score, disease) in enumerate(matches[:max_results]):
            diagnoses.append({
                'rank': i + 1,
                'disease_id': disease['id'],
                'disease_name': disease['name'],
                'omim_id': disease['id'].replace('OMIM:', ''),
                'mondo_id': disease.get('mondo', ''),
                'post_test_probability': score * 0.9,  # Mock probability
                'composite_lr': score * 100,  # Mock likelihood ratio
                'phenotype_score': score
            })
            
        # Add some random diseases to fill results
        while len(diagnoses) < min(5, max_results):
            diagnoses.append({
                'rank': len(diagnoses) + 1,
                'disease_id': f'OMIM:{100000 + len(diagnoses)}',
                'disease_name': f'Mock disease {len(diagnoses) + 1}',
                'omim_id': str(100000 + len(diagnoses)),
                'post_test_probability': 0.1 / (len(diagnoses) + 1),
                'composite_lr': 1.0,
                'phenotype_score': 0.1
            })
            
        return {
            'diagnoses': diagnoses,
            'metadata': {
                'lirical_version': 'simulator',
                'analysis_date': datetime.utcnow().isoformat(),
                'mode': 'simulation'
            }
        }


# Test the wrapper
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test simulator
        simulator = LiricalSimulator()
        
        test_hpo = ["HP:0001263", "HP:0001250", "HP:0000252"]
        print(f"Testing with HPO codes: {test_hpo}")
        
        results = await simulator.analyze_hpo_codes(test_hpo)
        
        print("\nResults:")
        print(json.dumps(results, indent=2))
        
    asyncio.run(test())