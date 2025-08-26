import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from .models import (
    EpicrisisQuery, UnifiedMatchResponse, UnifiedDiseaseMatch,
    UnifiedMetadata, UnifiedExtended, LiricalResult
)
from .config import Settings
from .hpo_loader import HPOLoader
from .symptom_extractor import SymptomExtractor
from .enhanced_symptom_extractor import EnhancedSymptomExtractor
from .hpo_matcher import HPOMatcher, HPOTextPreprocessor
from .hpo_selector import HPOSelector
from .hpo_validator import HPOValidator
from .lirical_wrapper import LiricalWrapper
from .query_generator import QueryGenerator
from .epicrisis_reranker import EpicrisisReranker
from .recursive_hpo_analyzer import RecursiveHPOAnalyzer
from .disease_reranker import DiseaseReranker

logger = logging.getLogger(__name__)


class EpicrisisOrchestrator:
    """Главный координатор обработки эпикризов"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Инициализация компонентов
        logger.info("Initializing epicrisis orchestrator...")
        
        # Загрузка HPO данных
        self.hpo_loader = HPOLoader()  # Использует путь по умолчанию из HPOLoader
        self.hpo_loader.load_all()
        
        # Компоненты пайплайна
        # Use enhanced extractor if enabled
        self.use_enhanced_extraction = getattr(settings, 'use_enhanced_extraction', True)
        if self.use_enhanced_extraction:
            self.symptom_extractor = EnhancedSymptomExtractor(settings)
        else:
            self.symptom_extractor = SymptomExtractor(settings)
            
        self.query_generator = QueryGenerator(settings)
        self.hpo_matcher = HPOMatcher(self.hpo_loader)
        self.hpo_selector = HPOSelector(settings)
        self.hpo_validator = HPOValidator(settings)
        
        # LiricalWrapper теперь может использовать наш custom analyzer
        self.lirical_wrapper = LiricalWrapper(settings=settings, hpo_loader=self.hpo_loader)
        
        # Recursive analyzer if enabled
        self.recursive_analyzer = None
        if settings.recursive_analysis_enabled:
            self.recursive_analyzer = RecursiveHPOAnalyzer(settings, self.hpo_loader, self.lirical_wrapper)
        
        # Инициализация reranker если включен
        self.epicrisis_reranker = None
        if hasattr(settings, 'epicrisis_rerank_enabled') and settings.epicrisis_rerank_enabled:
            self.epicrisis_reranker = EpicrisisReranker(settings, self.hpo_loader)
            logger.info("Epicrisis reranker enabled")
            
        # Disease reranker - optional based on settings
        self.disease_reranker = None
        if settings.final_disease_rerank_enabled:
            self.disease_reranker = DiseaseReranker(settings, self.hpo_loader)
            logger.info("Disease reranker initialized with clinical notes generation")
        
        logger.info("Epicrisis orchestrator initialized")
        
    async def process_epicrisis(self, query: EpicrisisQuery) -> UnifiedMatchResponse:
        """
        Главный метод обработки эпикриза
        
        Пайплайн:
        1. Извлечение симптомов из текста
        2. Поиск HPO кандидатов для каждого симптома
        3. Выбор лучшего HPO для каждого симптома
        4. Анализ через LIRICAL
        5. Формирование ответа
        """
        start_time = time.time()
        
        try:
            # 1. Извлечение симптомов
            logger.info("Step 1: Extracting symptoms from epicrisis")
            
            # Check if we're using enhanced extraction
            if self.use_enhanced_extraction:
                symptoms_with_synonyms = await self.symptom_extractor.extract_symptoms_with_synonyms(query.epicrisis_text)
                extracted = await self.symptom_extractor.get_full_extraction()
            else:
                extracted = await self.symptom_extractor.extract_symptoms(query.epicrisis_text)
                symptoms_with_synonyms = None
            
            if not extracted.symptoms and not extracted.codes.get("hpo"):
                # Нет симптомов и HPO кодов
                return self._create_empty_response(
                    processing_time=time.time() - start_time,
                    message="No symptoms or HPO codes found in epicrisis"
                )
                
            # Собираем HPO коды
            hpo_codes = []
            
            # 2-3. Поиск и выбор HPO для симптомов (если есть)
            if extracted.symptoms:
                logger.info(f"Step 2-3: Finding HPO codes for {len(extracted.symptoms)} symptoms")
                
                if self.use_enhanced_extraction and symptoms_with_synonyms:
                    # Use enhanced extraction with pre-generated synonyms
                    logger.info("Using enhanced extraction with pre-generated synonyms")
                    symptom_candidates = {}
                    
                    for symptom, queries in symptoms_with_synonyms.items():
                        logger.debug(f"Searching HPO for '{symptom}' with {len(queries)} query variations")
                        candidates = self.hpo_matcher.search_hpo_with_multiple_queries(
                            queries,
                            top_k=10
                        )
                        symptom_candidates[symptom] = candidates
                        
                    # Validate candidates with LLM
                    logger.info("Step 2.5: Validating HPO candidates")
                    validated_hpo = await self.hpo_validator.validate_batch(
                        symptom_candidates,
                        query.epicrisis_text,
                        threshold=self.settings.hpo_validation_threshold
                    )
                    
                    # Add validated HPO codes
                    for symptom, hpo_list in validated_hpo.items():
                        for hpo_id, confidence in hpo_list:
                            if hpo_id not in hpo_codes:
                                hpo_codes.append(hpo_id)
                                logger.info(f"Added validated HPO {hpo_id} for '{symptom}' with confidence {confidence}")
                else:
                    # Original flow
                    logger.info("Generating multiple search queries for symptoms...")
                    symptom_queries = await self.query_generator.generate_batch_queries(extracted.symptoms)
                    
                    # Поиск кандидатов с множественными запросами
                    symptom_candidates = {}
                    for symptom, queries in symptom_queries.items():
                        logger.debug(f"Searching HPO for '{symptom}' with queries: {queries}")
                        candidates = self.hpo_matcher.search_hpo_with_multiple_queries(
                            queries,
                            top_k=10
                        )
                        symptom_candidates[symptom] = candidates
                    
                    # Выбор лучших
                    selected_hpo = await self.hpo_selector.select_batch(symptom_candidates)
                    
                    # Добавляем выбранные коды
                    for symptom, hpo_id in selected_hpo.items():
                        if hpo_id:
                            hpo_codes.append(hpo_id)
                        
            # Добавляем уже извлеченные HPO коды
            if "hpo" in extracted.codes:
                hpo_codes.extend(extracted.codes["hpo"])
                
            # Убираем дубликаты
            hpo_codes = list(dict.fromkeys(hpo_codes))
            
            if not hpo_codes:
                return self._create_empty_response(
                    processing_time=time.time() - start_time,
                    message="Could not map symptoms to HPO codes"
                )
                
            logger.info(f"Found {len(hpo_codes)} unique HPO codes: {hpo_codes}")
            
            # 4. Анализ через LIRICAL (или рекурсивный анализ если включен)
            if self.recursive_analyzer and self.settings.recursive_analysis_enabled:
                logger.info("Step 4: Running recursive HPO analysis with LIRICAL")
                final_hpo, lirical_results = await self.recursive_analyzer.perform_recursive_analysis(
                    hpo_codes,
                    query.epicrisis_text
                )
                # Update HPO codes with recursive results
                hpo_codes = final_hpo
            else:
                logger.info("Step 4: Running LIRICAL analysis")
                lirical_results = self.lirical_wrapper.analyze_phenotypes(
                    hpo_codes,
                    top_k=query.top_k
                )
            
            if not lirical_results:
                # Still return HPO codes even if LIRICAL fails
                processing_time = time.time() - start_time
                return self._create_response_with_hpo_only(
                    hpo_codes=hpo_codes,
                    extracted_symptoms=extracted.symptoms,
                    extracted_codes=extracted.codes,
                    processing_time=processing_time,
                    message="LIRICAL analysis returned no results"
                )
                
            # 5. LLM Reranking если включен и есть достаточно кандидатов
            if (self.epicrisis_reranker and 
                len(lirical_results) >= self.settings.epicrisis_rerank_min_candidates):
                
                logger.info("Step 5: LLM reranking of LIRICAL results")
                
                # Создаем маппинг симптом -> HPO для reranker
                symptom_to_hpo = {}
                
                # Handle both enhanced and original extraction flows
                if self.use_enhanced_extraction and 'validated_hpo' in locals():
                    # For enhanced extraction, use validated HPO mapping
                    for symptom, hpo_list in validated_hpo.items():
                        if hpo_list:
                            hpo_id, confidence = hpo_list[0]  # Take the best validated HPO
                            if hpo_id in self.hpo_loader.hpo_terms:
                                hpo_term = self.hpo_loader.hpo_terms[hpo_id]
                                symptom_to_hpo[symptom] = f"{hpo_id} ({hpo_term.name})"
                elif 'selected_hpo' in locals():
                    # For original extraction, use selected HPO mapping
                    for symptom, hpo_id in selected_hpo.items():
                        if hpo_id and hpo_id in self.hpo_loader.hpo_terms:
                            hpo_term = self.hpo_loader.hpo_terms[hpo_id]
                            symptom_to_hpo[symptom] = f"{hpo_id} ({hpo_term.name})"
                
                # Переранжирование
                reranking_result = await self.epicrisis_reranker.rerank_lirical_results(
                    lirical_results=lirical_results[:self.settings.epicrisis_rerank_top_k],
                    extracted_symptoms=extracted.symptoms,
                    hpo_codes=hpo_codes,
                    original_text=query.epicrisis_text,
                    symptom_to_hpo_mapping=symptom_to_hpo
                )
                
                lirical_results = reranking_result.ranked_candidates
                logger.info(f"Reranking complete. New top result: {lirical_results[0].disease_name if lirical_results else 'None'}")
                
            # 6. Final Disease Reranking (optional)
            clinical_notes = None
            if (self.disease_reranker and 
                lirical_results and 
                len(lirical_results) > self.settings.rerank_top_k):
                
                logger.info("Step 6: Final disease reranking with clinical notes generation")
                rerank_result = await self.disease_reranker.rerank_diseases(
                    epicrisis_text=query.epicrisis_text,
                    disease_candidates=lirical_results,
                    extracted_hpo_codes=hpo_codes,
                    top_k=self.settings.rerank_top_k,
                    generate_notes=self.settings.generate_clinical_notes
                )
                lirical_results = rerank_result['diseases']
                clinical_notes = rerank_result.get('clinical_notes')
            
            # 7. Формирование ответа
            processing_time = time.time() - start_time
            
            return self._create_response(
                lirical_results=lirical_results,
                extracted_symptoms=extracted.symptoms,
                hpo_codes=hpo_codes,
                extracted_codes=extracted.codes,
                processing_time=processing_time,
                clinical_notes=clinical_notes
            )
            
        except Exception as e:
            logger.error(f"Error processing epicrisis: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return self._create_error_response(str(e), processing_time)
            
    def _create_response(
        self,
        lirical_results: List[LiricalResult],
        extracted_symptoms: List[str],
        hpo_codes: List[str],
        extracted_codes: Dict[str, List[str]],
        processing_time: float,
        clinical_notes: Optional[Dict[str, Any]] = None
    ) -> UnifiedMatchResponse:
        """Создать унифицированный ответ из результатов LIRICAL"""
        
        # Конвертируем результаты LIRICAL в унифицированный формат
        results = []
        lirical_scores = {}
        
        for idx, lirical_result in enumerate(lirical_results):
            # Извлекаем OMIM ID из disease_id (формат может быть OMIM:123456)
            omim_id = None
            mondo_id = None
            
            if lirical_result.disease_id.startswith("OMIM:"):
                omim_id = lirical_result.disease_id.replace("OMIM:", "")
            elif lirical_result.disease_id.startswith("MONDO:"):
                mondo_id = lirical_result.disease_id
                
            # Получаем гены для заболевания
            genes = []
            if lirical_result.disease_id:
                genes = self.hpo_loader.get_genes_for_disease(lirical_result.disease_id)
                
            result = UnifiedDiseaseMatch(
                rank=idx + 1,
                omim_id=omim_id,
                mondo_id=mondo_id,
                name=lirical_result.disease_name,
                score=lirical_result.post_test_probability,
                genes=genes
            )
            results.append(result)
            
            # Сохраняем детальные scores
            lirical_scores[lirical_result.disease_id] = lirical_result.post_test_probability
            
        # Метаданные
        metadata = UnifiedMetadata(
            processing_time_ms=int(processing_time * 1000),
            architecture="epicrisis_lirical",
            model=self.settings.openai_model,
            total_results=len(results)
        )
        
        # Расширенная информация
        # Convert clinical notes if present
        clinical_notes_obj = None
        if clinical_notes:
            from .models import ClinicalNotes
            clinical_notes_obj = ClinicalNotes(
                summary=clinical_notes.get('summary', ''),
                key_findings=clinical_notes.get('key_findings', []),
                recommendations=clinical_notes.get('recommendations')
            )
        
        # Создаем список HPO терминов с названиями
        from .models import HPOTerm
        hpo_terms = []
        for hpo_code in hpo_codes:
            if hpo_code in self.hpo_loader.hpo_terms:
                term = self.hpo_loader.hpo_terms[hpo_code]
                hpo_terms.append(HPOTerm(code=hpo_code, name=term.name))
            else:
                hpo_terms.append(HPOTerm(code=hpo_code, name="Unknown"))
        
        extended = UnifiedExtended(
            extracted_symptoms=extracted_symptoms,
            hpo_codes=hpo_codes,
            hpo_terms=hpo_terms,
            lirical_scores=lirical_scores,
            extracted_codes=extracted_codes,
            clinical_notes=clinical_notes_obj
        )
        
        return UnifiedMatchResponse(
            results=results,
            metadata=metadata,
            extended=extended,
            error=None
        )
        
    def _create_empty_response(self, processing_time: float, message: str) -> UnifiedMatchResponse:
        """Создать пустой ответ"""
        metadata = UnifiedMetadata(
            processing_time_ms=int(processing_time * 1000),
            architecture="epicrisis_lirical",
            model=self.settings.openai_model,
            total_results=0
        )
        
        extended = UnifiedExtended(
            extracted_symptoms=[],
            hpo_codes=[],
            lirical_scores=None,
            extracted_codes={}  # Пустой словарь вместо сообщения
        )
        
        return UnifiedMatchResponse(
            results=[],
            metadata=metadata,
            extended=extended,
            error=None
        )
        
    def _create_error_response(self, error_message: str, processing_time: float) -> UnifiedMatchResponse:
        """Создать ответ с ошибкой"""
        metadata = UnifiedMetadata(
            processing_time_ms=int(processing_time * 1000),
            architecture="epicrisis_lirical",
            model=self.settings.openai_model,
            total_results=0
        )
        
        return UnifiedMatchResponse(
            results=[],
            metadata=metadata,
            extended=None,
            error={
                "code": "PROCESSING_ERROR",
                "message": error_message
            }
        )
        
    def _create_response_with_hpo_only(
        self,
        hpo_codes: List[str],
        extracted_symptoms: List[str],
        extracted_codes: Dict[str, List[str]],
        processing_time: float,
        message: str = ""
    ) -> UnifiedMatchResponse:
        """Создать ответ только с HPO кодами (когда LIRICAL не работает)"""
        metadata = UnifiedMetadata(
            processing_time_ms=int(processing_time * 1000),
            architecture="epicrisis_lirical",
            model=self.settings.openai_model,
            total_results=0
        )
        
        extended = UnifiedExtended(
            extracted_symptoms=extracted_symptoms,
            hpo_codes=hpo_codes,
            lirical_scores=None,
            extracted_codes=extracted_codes
        )
        
        return UnifiedMatchResponse(
            results=[],
            metadata=metadata,
            extended=extended,
            error={
                "code": "LIRICAL_UNAVAILABLE",
                "message": message
            } if message else None
        )