import logging
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re

from .models import HPOCandidate
from .hpo_loader import HPOLoader, HPOTerm
# from .symptom_translator import SymptomTranslator
# from .hpo_russian_synonyms import get_hpo_for_russian_term, RUSSIAN_TO_HPO_MAPPING

logger = logging.getLogger(__name__)


class HPOTextPreprocessor:
    """Препроцессор текста для HPO терминов"""
    
    def preprocess(self, text: str) -> str:
        """Базовая предобработка текста"""
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем лишние символы
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        return text.split()


class HPOMatcher:
    """BM25-based поиск HPO терминов по симптомам"""
    
    def __init__(self, hpo_loader: HPOLoader, preprocessor: HPOTextPreprocessor = None):
        self.hpo_loader = hpo_loader
        self.preprocessor = preprocessor or HPOTextPreprocessor()
        # self.translator = SymptomTranslator()
        
        # Данные для BM25
        self.hpo_ids: List[str] = []
        self.hpo_texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None
        
        # Параметры BM25
        self.k1 = 1.2
        self.b = 0.75
        
        # Строим индекс
        self._build_index()
        
    def _build_index(self):
        """Построить BM25 индекс из HPO терминов"""
        logger.info("Building BM25 index for HPO terms...")
        
        # from .hpo_russian_synonyms import HPO_TO_RUSSIAN_MAPPING
        HPO_TO_RUSSIAN_MAPPING = {}
        
        hpo_terms = self.hpo_loader.hpo_terms
        
        if not hpo_terms:
            logger.warning("No HPO terms loaded, index will be empty")
            return
            
        # Подготавливаем данные для индексации
        for hpo_id, term in hpo_terms.items():
            # Собираем все тексты для термина
            texts = [term.name] + term.synonyms
            
            # Добавляем русские синонимы если есть
            if hpo_id in HPO_TO_RUSSIAN_MAPPING:
                texts.extend(HPO_TO_RUSSIAN_MAPPING[hpo_id])
                logger.debug(f"Added {len(HPO_TO_RUSSIAN_MAPPING[hpo_id])} Russian synonyms for {hpo_id}")
            
            combined_text = " | ".join(texts)  # Разделитель для читаемости
            
            # Предобработка
            processed_text = self.preprocessor.preprocess(combined_text)
            
            self.hpo_ids.append(hpo_id)
            self.hpo_texts.append(combined_text)
            
            # Токенизация для BM25
            tokens = self.preprocessor.tokenize(processed_text)
            self.tokenized_corpus.append(tokens)
            
        # Создаем BM25 индекс
        self.bm25 = BM25Okapi(
            self.tokenized_corpus, 
            k1=self.k1, 
            b=self.b
        )
        
        logger.info(f"BM25 index built with {len(self.hpo_ids)} HPO terms")
        
    def search_hpo_for_symptom(self, symptom: str, top_k: int = 10) -> List[HPOCandidate]:
        """
        Поиск HPO кодов для одного симптома
        
        Args:
            symptom: Симптом на русском/английском языке
            top_k: Количество результатов
            
        Returns:
            Список кандидатов HPO с оценками
        """
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []
        
        # Сначала проверяем прямое соответствие в русском словаре
        # direct_hpo = get_hpo_for_russian_term(symptom)
        direct_hpo = None
        if direct_hpo and direct_hpo in self.hpo_loader.hpo_terms:
            # Нашли прямое соответствие
            term = self.hpo_loader.hpo_terms[direct_hpo]
            logger.info(f"Found direct mapping: '{symptom}' -> {direct_hpo} ({term.name})")
            
            # Возвращаем с максимальным score
            direct_candidate = HPOCandidate(
                hpo_id=direct_hpo,
                name=term.name,
                synonyms=term.synonyms,
                score=100.0  # Максимальный score для прямого соответствия
            )
            
            # Добавляем еще кандидатов через BM25 для полноты
            additional_candidates = self._search_via_bm25(symptom, top_k - 1)
            
            # Фильтруем дубликаты
            result = [direct_candidate]
            for candidate in additional_candidates:
                if candidate.hpo_id != direct_hpo:
                    result.append(candidate)
            
            return result[:top_k]
        
        # Если прямого соответствия нет, используем обычный поиск
        return self._search_via_bm25(symptom, top_k)
    
    def _search_via_bm25(self, symptom: str, top_k: int) -> List[HPOCandidate]:
        """Поиск через BM25 с переводом"""
        # Переводим симптом, если он на русском
        # if self.translator.is_russian(symptom):
        #     translated_symptom = self.translator.translate_symptom(symptom)
        #     logger.debug(f"Translated symptom: '{symptom}' -> '{translated_symptom}'")
        #     query_text = translated_symptom
        # else:
        query_text = symptom
            
        # Предобработка запроса
        processed_query = self.preprocessor.preprocess(query_text)
        query_tokens = self.preprocessor.tokenize(processed_query)
        
        # BM25 поиск
        scores = self.bm25.get_scores(query_tokens)
        
        # Получаем топ-K результатов
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        candidates = []
        for idx in top_indices:
            if scores[idx] > 0:  # Только релевантные результаты
                hpo_id = self.hpo_ids[idx]
                term = self.hpo_loader.hpo_terms[hpo_id]
                
                candidate = HPOCandidate(
                    hpo_id=hpo_id,
                    name=term.name,
                    synonyms=term.synonyms,
                    score=float(scores[idx])
                )
                candidates.append(candidate)
                
        logger.debug(f"Found {len(candidates)} HPO candidates for symptom: {symptom}")
        return candidates
        
    def search_hpo_with_multiple_queries(self, queries: List[str], top_k: int = 10) -> List[HPOCandidate]:
        """
        Поиск HPO кодов используя множественные запросы
        
        Args:
            queries: Список поисковых запросов (например, разные вариации одного симптома)
            top_k: Количество финальных результатов
            
        Returns:
            Объединенный и отранжированный список кандидатов
        """
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []
            
        # Словарь для аккумулирования оценок
        combined_scores = {}  # {hpo_id: max_score}
        hpo_id_to_term = {}
        
        # Поиск для каждого запроса
        for query in queries:
            # Предобработка запроса
            processed_query = self.preprocessor.preprocess(query)
            query_tokens = self.preprocessor.tokenize(processed_query)
            
            # BM25 поиск
            scores = self.bm25.get_scores(query_tokens)
            
            # Обновляем максимальные оценки
            for idx, score in enumerate(scores):
                if score > 0:
                    hpo_id = self.hpo_ids[idx]
                    if hpo_id not in combined_scores or score > combined_scores[hpo_id]:
                        combined_scores[hpo_id] = score
                        hpo_id_to_term[hpo_id] = self.hpo_loader.hpo_terms[hpo_id]
        
        # Сортируем по оценкам и берем топ-K
        sorted_hpo_ids = sorted(combined_scores.keys(), 
                               key=lambda x: combined_scores[x], 
                               reverse=True)[:top_k]
        
        # Создаем кандидатов
        candidates = []
        for hpo_id in sorted_hpo_ids:
            term = hpo_id_to_term[hpo_id]
            candidate = HPOCandidate(
                hpo_id=hpo_id,
                name=term.name,
                synonyms=term.synonyms,
                score=float(combined_scores[hpo_id])
            )
            candidates.append(candidate)
            
        logger.debug(f"Found {len(candidates)} HPO candidates from {len(queries)} queries")
        return candidates
        
    def search_batch(self, symptoms: List[str], top_k_per_symptom: int = 5) -> Dict[str, List[HPOCandidate]]:
        """
        Поиск HPO кодов для списка симптомов
        
        Args:
            symptoms: Список симптомов
            top_k_per_symptom: Количество результатов на симптом
            
        Returns:
            Словарь {симптом: [кандидаты]}
        """
        results = {}
        
        for symptom in symptoms:
            candidates = self.search_hpo_for_symptom(symptom, top_k_per_symptom)
            results[symptom] = candidates
            
        return results
        
    def get_all_unique_hpo_codes(self, symptom_candidates: Dict[str, List[HPOCandidate]]) -> List[str]:
        """Получить все уникальные HPO коды из результатов поиска"""
        unique_codes = set()
        
        for candidates in symptom_candidates.values():
            for candidate in candidates:
                unique_codes.add(candidate.hpo_id)
                
        return list(unique_codes)