import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

logger = logging.getLogger(__name__)


class HPOTerm:
    """Представление HPO термина"""
    def __init__(self, hpo_id: str, name: str):
        self.id = hpo_id
        self.name = name
        self.synonyms: List[str] = []
        self.parents: List[str] = []
        self.namespace = "phenotypic_abnormality"
        
    def get_all_names(self) -> List[str]:
        """Получить все названия включая синонимы"""
        return [self.name] + self.synonyms


class HPOLoader:
    """Загрузчик HPO онтологии и связей с заболеваниями"""
    
    def __init__(self, data_dir: str = "/datasets"):
        self.data_dir = Path(data_dir)
        self.hpo_terms: Dict[str, HPOTerm] = {}
        self.hpo_to_diseases: Dict[str, Set[Tuple[str, float]]] = {}  # HPO -> [(disease, frequency)]
        self.disease_to_hpo: Dict[str, Set[str]] = {}  # Disease -> [HPO]
        self.disease_to_genes: Dict[str, Set[str]] = {}  # Disease -> [Genes]
        self.disease_names: Dict[str, str] = {}  # Disease ID -> Name
        
    def load_all(self) -> Dict[str, HPOTerm]:
        """Загрузить все данные HPO"""
        logger.info("Loading HPO data...")
        
        # Пробуем загрузить из JSON
        json_path = self.data_dir / "hp.json"
        if json_path.exists():
            self._load_hpo_json()
        else:
            # Загружаем онтологию из OBO
            self._load_hpo_obo()
        
        # Загружаем аннотации заболеваний
        self._load_phenotype_hpoa()
        
        # Загружаем связи генов с заболеваниями
        self._load_genes_to_disease()
        
        # Загружаем названия заболеваний из OMIM
        self._load_disease_names()
        
        logger.info(f"Loaded {len(self.hpo_terms)} HPO terms")
        logger.info(f"Loaded disease annotations for {len(self.disease_to_hpo)} diseases")
        logger.info(f"Loaded gene associations for {len(self.disease_to_genes)} diseases")
        logger.info(f"Loaded names for {len(self.disease_names)} diseases")
        
        return self.hpo_terms
        
    def _load_hpo_obo(self):
        """Загрузить HPO термины из OBO файла"""
        obo_path = self.data_dir / "hp.obo"
        
        if not obo_path.exists():
            logger.warning(f"HPO OBO file not found at {obo_path}")
            return
            
        current_term = None
        
        with open(obo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line == "[Term]":
                    # Сохраняем предыдущий термин
                    if current_term and current_term.namespace == "phenotypic_abnormality":
                        self.hpo_terms[current_term.id] = current_term
                    current_term = None
                    
                elif line.startswith("id: HP:"):
                    hpo_id = line[4:]
                    current_term = HPOTerm(hpo_id, "")
                    
                elif current_term and line.startswith("name: "):
                    current_term.name = line[6:]
                    
                elif current_term and line.startswith("synonym: "):
                    # Извлекаем синоним из кавычек
                    match = re.match(r'synonym: "([^"]+)"', line)
                    if match:
                        synonym = match.group(1)
                        current_term.synonyms.append(synonym)
                        
                elif current_term and line.startswith("is_a: HP:"):
                    parent_id = line[6:].split()[0]
                    current_term.parents.append(parent_id)
                    
                elif current_term and line.startswith("namespace: "):
                    current_term.namespace = line[11:]
                    
        # Сохраняем последний термин
        if current_term and current_term.namespace == "phenotypic_abnormality":
            self.hpo_terms[current_term.id] = current_term
            
        logger.info(f"Loaded {len(self.hpo_terms)} phenotypic abnormality terms from hp.obo")
        
    def _load_phenotype_hpoa(self):
        """Загрузить связи HPO-заболевания из phenotype.hpoa"""
        hpoa_path = self.data_dir / "phenotype.hpoa"
        
        if not hpoa_path.exists():
            logger.warning(f"phenotype.hpoa file not found at {hpoa_path}")
            return
            
        with open(hpoa_path, 'r', encoding='utf-8') as f:
            # Пропускаем заголовки
            for line in f:
                if not line.startswith('#'):
                    break
                    
            # Читаем данные
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                    
                disease_id = parts[0]  # OMIM:123456 или ORPHA:123
                hpo_id = parts[3]      # HP:0001234
                frequency_str = parts[7] if len(parts) > 7 else ""
                
                # Парсим частоту (например, "5/8" -> 0.625)
                frequency = self._parse_frequency(frequency_str)
                
                # Сохраняем связи
                if hpo_id not in self.hpo_to_diseases:
                    self.hpo_to_diseases[hpo_id] = set()
                self.hpo_to_diseases[hpo_id].add((disease_id, frequency))
                
                if disease_id not in self.disease_to_hpo:
                    self.disease_to_hpo[disease_id] = set()
                self.disease_to_hpo[disease_id].add(hpo_id)
                
        logger.info(f"Loaded HPO annotations for {len(self.disease_to_hpo)} diseases")
        
    def _parse_frequency(self, freq_str: str) -> float:
        """Парсинг строки частоты в число"""
        if not freq_str or freq_str == "HP:0040283":  # HP:0040283 = Occasional
            return 0.1
            
        # Парсим формат "n/m"
        match = re.match(r'(\d+)/(\d+)', freq_str)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return numerator / denominator if denominator > 0 else 0.0
            
        # HP термины частоты
        frequency_map = {
            "HP:0040281": 0.99,   # Very frequent
            "HP:0040282": 0.70,   # Frequent
            "HP:0040283": 0.30,   # Occasional
            "HP:0040284": 0.05,   # Very rare
            "HP:0012831": 1.00,   # Obligate (100%)
        }
        
        return frequency_map.get(freq_str, 0.5)  # По умолчанию 50%
        
    def search_terms(self, query: str, limit: int = 10) -> List[HPOTerm]:
        """Простой поиск HPO терминов по названию"""
        query_lower = query.lower()
        results = []
        
        for hpo_id, term in self.hpo_terms.items():
            # Поиск в названии и синонимах
            if query_lower in term.name.lower():
                results.append((term, 1.0))  # Точное совпадение в названии
            else:
                for synonym in term.synonyms:
                    if query_lower in synonym.lower():
                        results.append((term, 0.8))  # Совпадение в синониме
                        break
                        
        # Сортируем по релевантности и возвращаем
        results.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in results[:limit]]
        
    def get_diseases_for_hpo(self, hpo_id: str) -> List[Tuple[str, float]]:
        """Получить заболевания для данного HPO кода"""
        return list(self.hpo_to_diseases.get(hpo_id, []))
        
    def get_hpo_for_disease(self, disease_id: str) -> Set[str]:
        """Получить HPO коды для заболевания"""
        return self.disease_to_hpo.get(disease_id, set())
        
    def get_disease_hpo_with_frequencies(self, disease_id: str) -> Dict[str, float]:
        """
        Получить HPO коды заболевания с их частотами
        
        Args:
            disease_id: ID заболевания (например, "OMIM:123456")
            
        Returns:
            Словарь {HPO_ID: частота}, например {"HP:0001249": 0.95}
        """
        hpo_with_freq = {}
        
        # Получаем все HPO коды для заболевания
        hpo_codes = self.get_hpo_for_disease(disease_id)
        
        # Для каждого HPO кода находим частоту для данного заболевания
        for hpo_id in hpo_codes:
            # Ищем в hpo_to_diseases частоту для конкретного заболевания
            if hpo_id in self.hpo_to_diseases:
                for dis_id, freq in self.hpo_to_diseases[hpo_id]:
                    if dis_id == disease_id:
                        hpo_with_freq[hpo_id] = freq
                        break
                else:
                    # Если частота не найдена, используем значение по умолчанию
                    hpo_with_freq[hpo_id] = 0.5
            else:
                hpo_with_freq[hpo_id] = 0.5
                
        return hpo_with_freq
    
    def _load_hpo_json(self):
        """Загрузить HPO термины из JSON файла"""
        import json
        json_path = self.data_dir / "hp.json"
        
        logger.info(f"Loading HPO from JSON: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # JSON structure: {"graphs": [{"nodes": [...]}]}
        if "graphs" in data and len(data["graphs"]) > 0:
            nodes = data["graphs"][0].get("nodes", [])
            
            for node in nodes:
                if node.get("type") == "CLASS":
                    hpo_id = node.get("id", "").replace("http://purl.obolibrary.org/obo/HP_", "HP:")
                    
                    if hpo_id.startswith("HP:"):
                        term = HPOTerm(hpo_id, node.get("lbl", ""))
                        
                        # Add synonyms
                        meta = node.get("meta", {})
                        if "synonyms" in meta:
                            for syn in meta["synonyms"]:
                                term.synonyms.append(syn.get("val", ""))
                        
                        # Only keep phenotypic abnormality terms
                        if "xrefs" in meta:
                            for xref in meta["xrefs"]:
                                if xref.get("val", "").startswith("UMLS:"):
                                    term.namespace = "phenotypic_abnormality"
                                    break
                        else:
                            # If no xrefs, assume it's a phenotypic abnormality
                            term.namespace = "phenotypic_abnormality"
                        
                        self.hpo_terms[hpo_id] = term
            
        logger.info(f"Loaded {len(self.hpo_terms)} HPO terms from JSON")
    
    def _load_genes_to_disease(self):
        """Загрузить связи генов и заболеваний из genes_to_disease.txt"""
        genes_file = self.data_dir / "genes_to_disease.txt"
        
        if not genes_file.exists():
            logger.warning(f"genes_to_disease.txt not found at {genes_file}")
            return
            
        logger.info(f"Loading genes_to_disease from {genes_file}")
        
        with open(genes_file, 'r', encoding='utf-8') as f:
            # Пропускаем заголовок
            next(f)
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    gene_symbol = parts[1]
                    disease_id = parts[3]  # Уже в формате OMIM:123456
                    
                    if disease_id not in self.disease_to_genes:
                        self.disease_to_genes[disease_id] = set()
                    self.disease_to_genes[disease_id].add(gene_symbol)
                    
        logger.info(f"Loaded gene associations for {len(self.disease_to_genes)} diseases")
    
    def get_genes_for_disease(self, disease_id: str) -> List[str]:
        """Получить список генов для заболевания"""
        return sorted(list(self.disease_to_genes.get(disease_id, [])))
    
    def _load_disease_names(self):
        """Загрузить названия заболеваний из mimTitles.txt"""
        mim_titles_path = self.data_dir / "mimTitles.txt"
        
        if not mim_titles_path.exists():
            logger.warning(f"mimTitles.txt not found at {mim_titles_path}")
            return
            
        logger.info(f"Loading disease names from {mim_titles_path}")
        
        with open(mim_titles_path, 'r', encoding='utf-8') as f:
            # Пропускаем заголовки
            for line in f:
                if not line.startswith('#'):
                    break
                    
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    prefix = parts[0]  # Prefix (asterisk, plus, etc.)
                    mim_number = parts[1]
                    preferred_title = parts[2]
                    
                    # Название уже чистое, просто убираем лишние пробелы
                    clean_name = preferred_title.strip()
                    
                    disease_id = f"OMIM:{mim_number}"
                    self.disease_names[disease_id] = clean_name
                    
        logger.info(f"Loaded {len(self.disease_names)} disease names")
    
    def get_disease_name(self, disease_id: str) -> str:
        """Получить название заболевания по ID"""
        return self.disease_names.get(disease_id, f"Unknown disease {disease_id}")