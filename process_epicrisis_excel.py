#!/usr/bin/env python3
"""
Скрипт для обработки Excel файлов с эпикризами через API disease-matcher-epicrisis-min.

Использование:
    python3 process_epicrisis_excel.py input_file.xlsx [--output output_file.xlsx]
    python3 process_epicrisis_excel.py input_file.xlsx --start-from 10
    python3 process_epicrisis_excel.py input_file.xlsx --api-url http://localhost:8004
"""

import pandas as pd
import requests
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_epicrisis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EpicrisisProcessor:
    """Класс для обработки эпикризов через API"""
    
    def __init__(self, api_url: str = "http://localhost:8004/api/v1/match_epicrisis", 
                 delay_between_requests: float = 0.5):
        self.api_url = api_url
        self.delay_between_requests = delay_between_requests
        self.session = requests.Session()
        self.hpo_names_cache = {}  # Кеш для названий HPO
        
    def call_api(self, text: str, top_k: int = 10, max_retries: int = 3) -> Optional[Dict]:
        """
        Отправить текст на API и получить результат
        
        Args:
            text: Текст эпикриза
            top_k: Количество результатов
            max_retries: Максимальное количество попыток
            
        Returns:
            Словарь с результатами или None при ошибке
        """
        if not text or pd.isna(text) or str(text).strip() == '':
            logger.warning("Пустой текст, пропускаем")
            return None
            
        payload = {
            "epicrisis_text": str(text),
            "language": "ru",
            "top_k": top_k
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.api_url, 
                    json=payload,
                    timeout=180  # 3 минуты таймаут для длинных текстов
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"API вернул код {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут при обработке (попытка {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                logger.error(f"Ошибка соединения (попытка {attempt + 1}/{max_retries})")
                time.sleep(5)  # Подождать перед повторной попыткой
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(self.delay_between_requests * (attempt + 1))
                
        return None
    
    def get_hpo_name(self, hpo_code: str) -> str:
        """Получить название HPO по коду из кеша или API"""
        # Проверяем кеш
        if hpo_code in self.hpo_names_cache:
            return self.hpo_names_cache[hpo_code]
            
        # Если нет в кеше, пока просто возвращаем код
        # В будущем можно добавить запрос к API для получения названий
        return hpo_code
    
    def process_response(self, response: Optional[Dict]) -> Dict[str, str]:
        """
        Извлечь нужные данные из ответа API
        
        Args:
            response: Ответ от API
            
        Returns:
            Словарь с извлеченными данными
        """
        result = {
            'HPO_коды': '',
            'HPO_названия': '',
            'Заболевания_коды': '',
            'Заболевания_названия': '',
            'Топ1_заболевание': '',
            'Топ1_score': '',
            'Клинические_рекомендации': ''
        }
        
        if not response:
            return result
            
        try:
            # HPO коды и названия
            if 'extended' in response and response['extended']:
                hpo_codes = response['extended'].get('hpo_codes', [])
                result['HPO_коды'] = ', '.join(hpo_codes)
                
                # Используем HPO термины с названиями
                hpo_terms = response['extended'].get('hpo_terms', [])
                if hpo_terms:
                    # Собираем названия HPO терминов
                    hpo_names = []
                    for term in hpo_terms:
                        hpo_names.append(f"{term['code']}: {term['name']}")
                    result['HPO_названия'] = '; '.join(hpo_names)
                else:
                    # Fallback на extracted_symptoms если нет hpo_terms
                    symptoms = response['extended'].get('extracted_symptoms', [])
                    result['HPO_названия'] = '; '.join(symptoms)
            
            # Заболевания
            if 'results' in response and response['results']:
                diseases = response['results'][:10]  # Топ-10
                
                disease_codes = []
                disease_names = []
                
                for disease in diseases:
                    # Формируем код
                    if disease.get('omim_id'):
                        disease_codes.append(f"OMIM:{disease['omim_id']}")
                    elif disease.get('mondo_id'):
                        disease_codes.append(disease['mondo_id'])
                        
                    # Название
                    disease_names.append(disease.get('name', 'Unknown'))
                
                result['Заболевания_коды'] = ', '.join(disease_codes)
                result['Заболевания_названия'] = '; '.join(disease_names)
                
                # Топ-1 результат
                if len(diseases) > 0:
                    top_disease = diseases[0]
                    if top_disease.get('omim_id'):
                        result['Топ1_заболевание'] = f"OMIM:{top_disease['omim_id']} - {top_disease.get('name', '')}"
                    else:
                        result['Топ1_заболевание'] = top_disease.get('name', '')
                    result['Топ1_score'] = f"{top_disease.get('score', 0):.3f}"
            
            # Клинические рекомендации
            if 'extended' in response and response['extended']:
                clinical_notes = response['extended'].get('clinical_notes')
                if clinical_notes:
                    recommendations = []
                    
                    if clinical_notes.get('summary'):
                        recommendations.append(f"РЕЗЮМЕ: {clinical_notes['summary']}")
                        
                    if clinical_notes.get('key_findings'):
                        recommendations.append(f"КЛЮЧЕВЫЕ НАХОДКИ: {'; '.join(clinical_notes['key_findings'])}")
                        
                    if clinical_notes.get('recommendations'):
                        recommendations.append(f"РЕКОМЕНДАЦИИ: {clinical_notes['recommendations']}")
                        
                    result['Клинические_рекомендации'] = '\n'.join(recommendations)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке ответа: {e}")
            
        return result
    
    def process_excel_file(self, input_file: str, output_file: Optional[str] = None,
                          start_from: int = 0) -> None:
        """
        Обработать Excel файл
        
        Args:
            input_file: Путь к входному файлу
            output_file: Путь к выходному файлу (если не указан, добавит _processed к имени)
            start_from: С какой строки начать обработку
        """
        # Проверяем входной файл
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")
            
        # Определяем выходной файл
        if output_file is None:
            output_file = input_path.stem + "_processed" + input_path.suffix
            
        logger.info(f"Начинаем обработку файла: {input_file}")
        logger.info(f"Результаты будут сохранены в: {output_file}")
        
        # Читаем Excel
        try:
            df = pd.read_excel(input_file)
            logger.info(f"Загружено {len(df)} строк")
        except Exception as e:
            logger.error(f"Ошибка при чтении Excel файла: {e}")
            raise
            
        # Проверяем наличие первой колонки
        if df.empty:
            logger.error("Файл пуст")
            return
            
        first_column_name = df.columns[0]
        logger.info(f"Обрабатываем колонку: {first_column_name}")
        
        # Добавляем новые колонки если их еще нет
        new_columns = [
            'HPO_коды', 'HPO_названия', 'Заболевания_коды', 
            'Заболевания_названия', 'Топ1_заболевание', 'Топ1_score',
            'Клинические_рекомендации'
        ]
        
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''
                
        # Проверяем доступность API
        try:
            health_check = self.session.get(self.api_url.replace('/match_epicrisis', '/health'), timeout=5)
            if health_check.status_code != 200:
                logger.warning("API health check failed, но продолжаем попытку")
        except:
            logger.warning("Не удалось проверить доступность API, продолжаем")
            
        # Обрабатываем каждую строку
        total_rows = len(df)
        processed = 0
        errors = 0
        
        # Создаем прогресс-бар
        with tqdm(total=total_rows - start_from, desc="Обработка эпикризов") as pbar:
            for idx, row in df.iterrows():
                if idx < start_from:
                    continue
                    
                epicrisis_text = row[first_column_name]
                
                # Пропускаем пустые строки
                if pd.isna(epicrisis_text) or str(epicrisis_text).strip() == '':
                    pbar.update(1)
                    continue
                    
                logger.info(f"Обработка строки {idx + 1}/{total_rows}")
                
                # Вызываем API
                response = self.call_api(epicrisis_text)
                
                if response:
                    # Обрабатываем ответ
                    extracted_data = self.process_response(response)
                    
                    # Обновляем DataFrame
                    for col, value in extracted_data.items():
                        df.at[idx, col] = value
                        
                    processed += 1
                    logger.info(f"Строка {idx + 1} обработана успешно")
                else:
                    errors += 1
                    logger.error(f"Ошибка при обработке строки {idx + 1}")
                    
                # Сохраняем промежуточный результат каждые 10 строк
                if (processed + errors) % 10 == 0:
                    df.to_excel(output_file, index=False)
                    logger.info(f"Промежуточное сохранение после {processed + errors} строк")
                    
                # Задержка между запросами
                time.sleep(self.delay_between_requests)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Обработано': processed,
                    'Ошибок': errors
                })
        
        # Сохраняем финальный результат
        df.to_excel(output_file, index=False)
        logger.info(f"Обработка завершена. Обработано: {processed}, Ошибок: {errors}")
        logger.info(f"Результаты сохранены в: {output_file}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Обработка Excel файлов с эпикризами через API'
    )
    parser.add_argument('input_file', help='Путь к входному Excel файлу')
    parser.add_argument('--output', '-o', help='Путь к выходному файлу')
    parser.add_argument('--api-url', default='http://localhost:8004/api/v1/match_epicrisis',
                        help='URL API эпикризов')
    parser.add_argument('--start-from', type=int, default=0,
                        help='С какой строки начать обработку (0-based)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Задержка между запросами в секундах')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Количество заболеваний для возврата')
    
    args = parser.parse_args()
    
    # Создаем процессор
    processor = EpicrisisProcessor(
        api_url=args.api_url,
        delay_between_requests=args.delay
    )
    
    # Обрабатываем файл
    try:
        processor.process_excel_file(
            input_file=args.input_file,
            output_file=args.output,
            start_from=args.start_from
        )
    except KeyboardInterrupt:
        logger.info("Обработка прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == '__main__':
    main()