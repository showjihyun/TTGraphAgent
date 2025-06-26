"""
Advanced NLP Processor for Knowledge Graph Construction
5-Stage Pipeline: Preprocessing → NER → Relation Extraction → Attribute Parsing → LLM Validation
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# NLP Libraries
import spacy
import nltk
from konlpy.tag import Okt, Mecab
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """엔티티 클래스"""
    id: str
    text: str
    label: str
    start: int
    end: int
    confidence: float
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class Relation:
    """관계 클래스"""
    source: str
    target: str
    relation: str
    confidence: float
    context: str
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

class AdvancedNLPProcessor:
    """고급 NLP 처리기"""
    
    def __init__(self, language: str = "korean"):
        self.language = language
        self.device_info = self._get_device_info()
        self.setup_models()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """GPU/CPU 장치 정보 확인"""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            device_info = {
                "torch_available": True,
                "cuda_available": cuda_available,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "current_device": torch.cuda.current_device() if cuda_available else None,
                "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
                "cuda_version": torch.version.cuda if cuda_available else None,
                "pytorch_version": torch.__version__
            }
            
            if cuda_available:
                device_info["gpu_memory"] = {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "allocated": torch.cuda.memory_allocated(0),
                    "cached": torch.cuda.memory_reserved(0)
                }
                logger.info(f"🚀 GPU DETECTED: {device_info['device_name']}")
                logger.info(f"GPU Memory: {device_info['gpu_memory']['total'] / 1024**3:.1f}GB total")
            else:
                logger.warning("⚠️  No GPU detected, will use CPU")
            
            logger.info(f"PyTorch version: {device_info['pytorch_version']}")
            return device_info
            
        except ImportError:
            logger.warning("PyTorch not available, using CPU only")
            return {"torch_available": False, "cuda_available": False}
        
    def setup_models(self):
        """모델 초기화"""
        try:
            # 1. SpaCy 모델 (영어만)
            if self.language == "english":
                try:
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                    logger.info("English SpaCy model loaded successfully")
                except OSError:
                    logger.warning("en_core_web_sm not found, using blank English model")
                    self.spacy_nlp = spacy.blank("en")
            else:
                # 한국어의 경우 SpaCy 사용하지 않음 (토크나이저 의존성 문제 완전 회피)
                logger.info("Korean language: Skipping SpaCy to avoid tokenizer dependencies")
                self.spacy_nlp = None
            
            # 2. 한국어 형태소 분석기 (선택적)
            if self.language == "korean":
                try:
                    # Okt만 시도 (MeCab 의존성 완전 회피)
                    from konlpy.tag import Okt
                    self.korean_analyzer = Okt()
                    logger.info("Okt Korean analyzer initialized successfully")
                except Exception as okt_error:
                    logger.warning(f"Korean analyzer initialization failed: {okt_error}")
                    logger.info("Will use rule-based processing only")
                    self.korean_analyzer = None
            else:
                self.korean_analyzer = None
            
            # 3. HuggingFace NER 모델 (선택적)
            try:
                if self.language == "korean":
                    model_name = "klue/bert-base-korean-ner"
                else:
                    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                
                # 강화된 GPU 우선 사용 설정
                import torch
                
                if torch.cuda.is_available():
                    device = 0  # GPU device 0
                    torch.cuda.set_device(0)
                    logger.info(f"NER will use GPU: {torch.cuda.get_device_name(0)}")
                else:
                    device = -1  # CPU
                    logger.warning("NER will use CPU")
                
                # 모델 다운로드 시도 (GPU 우선)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    aggregation_strategy="simple",
                    device=device,
                    torch_dtype=torch.float16 if device == 0 else torch.float32  # GPU에서 float16 사용
                )
                device_name = "GPU" if device == 0 else "CPU"
                logger.info(f"NER pipeline initialized with {model_name} on {device_name}")
                    
            except Exception as e:
                logger.warning(f"NER pipeline failed to load: {e}")
                self.ner_pipeline = None
            
            # 4. 문장 임베딩 모델 (선택적)
            try:
                if self.language == "korean":
                    # 더 가벼운 한국어 모델 사용
                    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                else:
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                
                # 강화된 GPU 우선 사용 설정
                import torch
                import os
                
                # CUDA 환경 변수 설정
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
                # GPU 강제 우선 설정
                if torch.cuda.is_available():
                    device = "cuda:0"
                    torch.cuda.set_device(0)
                    # GPU 메모리 정리
                    torch.cuda.empty_cache()
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                else:
                    device = "cpu"
                    logger.warning("No GPU available, using CPU")
                
                # SentenceTransformer 초기화 시 명시적 device 설정
                self.sentence_model = SentenceTransformer(model_name)
                self.sentence_model = self.sentence_model.to(device)
                
                # 실제 사용 중인 device 확인
                actual_device = next(self.sentence_model.parameters()).device
                logger.info(f"Sentence transformer initialized with {model_name}")
                logger.info(f"Target device: {device}, Actual device: {actual_device}")
                
            except Exception as e:
                logger.warning(f"Sentence transformer failed to load: {e}")
                self.sentence_model = None
                
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            self.setup_fallback_models()
    
    def setup_fallback_models(self):
        """폴백 모델 설정"""
        self.spacy_nlp = None
        self.korean_analyzer = None
        self.ner_pipeline = None
        self.sentence_model = None
        logger.info("Using fallback rule-based processing only")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """5단계 파이프라인 실행"""
        try:
            # 1️⃣ 전처리
            preprocessed = self.preprocess_text(text)
            
            # 2️⃣ 엔티티 추출 (NER)
            entities = self.extract_entities(preprocessed)
            
            # 3️⃣ 관계 추출 (RE)
            relations = self.extract_relations(preprocessed, entities)
            
            # 4️⃣ 속성 추출
            entities_with_attributes = self.extract_attributes(preprocessed, entities)
            
            # 5️⃣ 구조화 + 검증 (LLM 보정은 별도 호출)
            structured_graph = self.structure_and_validate(
                entities_with_attributes, relations, preprocessed
            )
            
            return {
                "original_text": text,
                "preprocessed_text": preprocessed,
                "entities": [self._entity_to_dict(e) for e in entities_with_attributes],
                "relations": [self._relation_to_dict(r) for r in relations],
                "graph": structured_graph,
                "processing_stats": self._get_processing_stats(entities_with_attributes, relations)
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return self._fallback_processing(text)
    
    def preprocess_text(self, text: str) -> str:
        """1️⃣ 전처리: 문장 분리 및 정제"""
        try:
            # 기본 정제
            text = re.sub(r'\s+', ' ', text)  # 다중 공백 제거
            text = re.sub(r'[^\w\s가-힣.,!?;:()\-\[\]{}"]', '', text)  # 특수문자 제거
            text = text.strip()
            
            # 문장 분리
            if self.language == "korean" and self.korean_analyzer:
                # 한국어 문장 분리
                sentences = self._split_korean_sentences(text)
            else:
                # 영어 또는 폴백 문장 분리
                sentences = self._split_sentences_basic(text)
            
            # 문장 정제 및 재결합
            cleaned_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:  # 너무 짧은 문장 제외
                    cleaned_sentences.append(sent)
            
            return ' '.join(cleaned_sentences)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return text
    
    def extract_entities(self, text: str) -> List[Entity]:
        """2️⃣ 엔티티 추출 (NER)"""
        entities = []
        
        try:
            # HuggingFace NER 사용
            if self.ner_pipeline:
                ner_results = self.ner_pipeline(text)
                for i, result in enumerate(ner_results):
                    entity = Entity(
                        id=f"entity_{i}",
                        text=result['word'],
                        label=result['entity_group'],
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score']
                    )
                    entities.append(entity)
            
            # SpaCy NER 보완
            if self.spacy_nlp and len(entities) < 3:
                doc = self.spacy_nlp(text)
                for i, ent in enumerate(doc.ents):
                    entity = Entity(
                        id=f"spacy_entity_{i}",
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8
                    )
                    entities.append(entity)
            
            # 폴백: 규칙 기반 엔티티 추출
            if len(entities) < 2:
                entities.extend(self._extract_entities_rule_based(text))
            
            # 중복 제거 및 정제
            entities = self._deduplicate_entities(entities)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            entities = self._extract_entities_rule_based(text)
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """3️⃣ 관계 추출 (RE)"""
        relations = []
        
        try:
            # 패턴 기반 관계 추출 (중복 방지를 위해 하나의 방법만 사용)
            pattern_relations = self._extract_relations_by_patterns(text, entities)
            relations.extend(pattern_relations)
            
            # 임베딩 기반 관계 추출 (선택적, 패턴 기반으로 충분하지 않을 때만)
            if self.sentence_model and len(relations) == 0:
                embedding_relations = self._extract_relations_by_embeddings(text, entities)
                relations.extend(embedding_relations)
            
            # 관계 중복 제거
            relations = self._deduplicate_relations(relations)
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            relations = self._extract_relations_fallback(entities)
        
        return relations
    
    def extract_attributes(self, text: str, entities: List[Entity]) -> List[Entity]:
        """4️⃣ 속성 추출"""
        try:
            for entity in entities:
                # 날짜 속성
                entity.attributes.update(self._extract_date_attributes(text, entity))
                
                # 수치 속성
                entity.attributes.update(self._extract_numeric_attributes(text, entity))
                
                # 위치 속성
                entity.attributes.update(self._extract_location_attributes(text, entity))
                
                # 카테고리 속성
                entity.attributes.update(self._extract_category_attributes(text, entity))
                
        except Exception as e:
            logger.error(f"Attribute extraction failed: {e}")
        
        return entities
    
    def structure_and_validate(self, entities: List[Entity], relations: List[Relation], text: str) -> Dict[str, Any]:
        """5️⃣ 구조화 + 검증"""
        try:
            # 노드 생성 (프론트엔드 형식에 맞게)
            nodes = []
            for entity in entities:
                node = {
                    "id": entity.text,  # 프론트엔드에서 텍스트를 ID로 사용
                    "label": entity.text,
                    "group": entity.label,  # 프론트엔드에서 group 필드 사용
                    "confidence": entity.confidence,
                    "attributes": entity.attributes,
                    "x": np.random.randint(100, 700),
                    "y": np.random.randint(100, 600)
                }
                nodes.append(node)
            
            # 엣지 생성 (프론트엔드 형식에 맞게)
            edges = []
            
            for i, relation in enumerate(relations):
                edge = {
                    "id": f"edge_{i}",
                    "from": relation.source,  # 이미 텍스트로 설정됨
                    "to": relation.target,    # 이미 텍스트로 설정됨
                    "label": relation.relation,
                    "confidence": relation.confidence,
                    "context": relation.context,
                    "attributes": relation.attributes
                }
                edges.append(edge)
            
            # 그래프 통계
            summary = {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_types": list(set([n["group"] for n in nodes])),  # group 필드 사용
                "processing_language": self.language,
                "confidence_avg": np.mean([n["confidence"] for n in nodes]) if nodes else 0
            }
            
            return {
                "nodes": nodes,
                "edges": edges,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Structuring failed: {e}")
            return {"nodes": [], "edges": [], "summary": {}}
    
    # Helper Methods
    def _split_korean_sentences(self, text: str) -> List[str]:
        """한국어 문장 분리"""
        sentences = re.split(r'[.!?]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_sentences_basic(self, text: str) -> List[str]:
        """기본 문장 분리"""
        try:
            if self.spacy_nlp:
                doc = self.spacy_nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            else:
                return nltk.sent_tokenize(text)
        except:
            return re.split(r'[.!?]+\s*', text)
    
    def _extract_entities_rule_based(self, text: str) -> List[Entity]:
        """규칙 기반 엔티티 추출"""
        entities = []
        
        try:
            # 한국어 패턴 (더 정확하게)
            if self.language == "korean":
                patterns = [
                    # 인명 패턴 (명확한 호칭이 있는 경우만)
                    (r'[가-힣]{2,4}(?:\s*(?:씨|님|교수|박사|대표|회장|사장|선생|부장|과장|팀장))', "PERSON", 0.8),
                    # 조직명 패턴 (명확한 조직 접미사가 있는 경우만)
                    (r'[가-힣A-Za-z]{2,}(?:회사|기업|대학교|대학|연구소|재단|협회|센터|청|처|부서)', "ORG", 0.7),
                    # 장소 패턴 (명확한 지명 접미사가 있는 경우만)
                    (r'[가-힣]{2,}(?:시|도|구|군|동|읍|면|리|역|공항|병원|학교)', "LOC", 0.7),
                    # 날짜 패턴
                    (r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}월\s*\d{1,2}일|\d{4}년', "DATE", 0.8),
                    # 숫자 + 단위 (명확한 단위가 있는 경우만)
                    (r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:원|달러|만원|억원|명|개|시간|분|초|미터|킬로)', "QUANTITY", 0.7),
                    # 제품/서비스명 (따옴표나 특수 문자로 둘러싸인 경우)
                    (r'["\'][가-힣A-Za-z0-9\s]{2,}["\']', "PRODUCT", 0.6)
                ]
            else:
                # 영어 패턴
                patterns = [
                    # 대문자로 시작하는 단어들 (인명, 조직명 등)
                    (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', "MISC", 0.5),
                    # 날짜 패턴
                    (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', "DATE", 0.8),
                    # 숫자 + 단위
                    (r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars?|years?|months?|days?|hours?|minutes?)\b', "QUANTITY", 0.7)
                ]
            
            # 패턴 매칭
            for pattern, label, confidence in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # 중복 체크
                    overlap = False
                    for existing in entities:
                        if (match.start() < existing.end and match.end() > existing.start):
                            overlap = True
                            break
                    
                    if not overlap and len(match.group().strip()) > 1:
                        entities.append(Entity(
                            id=f"rule_{label.lower()}_{len(entities)}",
                            text=match.group().strip(),
                            label=label,
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence
                        ))
            
            # 의미 있는 엔티티만 추출 (최소 보장 로직 제거)
            logger.info(f"Rule-based extraction found {len(entities)} entities")
                        
        except Exception as e:
            logger.error(f"Rule-based entity extraction failed: {e}")
            # 폴백 시에도 의미 있는 엔티티만 추출
            entities = []
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """엔티티 중복 제거"""
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            if entity.text.lower() not in seen_texts:
                seen_texts.add(entity.text.lower())
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """관계 중복 제거"""
        unique_relations = []
        seen_relations = set()
        
        for relation in relations:
            # 소스-타겟-관계 조합으로 중복 체크
            relation_key = (relation.source.lower(), relation.target.lower(), relation.relation.lower())
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _extract_relation_between_entities(self, text: str, entity1: Entity, entity2: Entity) -> Optional[Relation]:
        """두 엔티티 간 관계 추출"""
        # 전체 텍스트에서 두 엔티티가 포함된 문맥 추출
        entity1_pos = text.find(entity1.text)
        entity2_pos = text.find(entity2.text)
        
        if entity1_pos == -1 or entity2_pos == -1:
            return None
        
        # 두 엔티티 사이의 거리 확인
        distance = abs(entity1_pos - entity2_pos)
        if distance > 100:  # 너무 멀리 떨어진 엔티티
            return None
        
        # 관계 패턴을 전체 텍스트에서 검색
        relation_patterns = {
            "소속": [
                r"([가-힣A-Za-z\s]+)\s*(?:교수|박사|대표|회장|사장|선생|부장|과장|팀장).*?([가-힣A-Za-z]+(?:대학교|대학|회사|기업|연구소|센터))",
                r"([가-힣A-Za-z\s]+).*?(?:에서|에게|에)\s*(?:소속|근무|일하|재직).*?([가-힣A-Za-z]+)"
            ],
            "연구": [
                r"([가-힣A-Za-z\s]+).*?(?:에서|에게|에)\s*([가-힣A-Za-z]+).*?(?:연구|개발|공부)"
            ],
            "관련": [
                r"([가-힣A-Za-z\s]+).*?(?:와|과|에|의)\s*(?:관련|연관|관계|대해).*?([가-힣A-Za-z]+)"
            ]
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # 매치된 그룹이 두 엔티티를 포함하는지 확인
                    group1, group2 = match.groups()
                    if ((entity1.text in group1 and entity2.text in group2) or 
                        (entity1.text in group2 and entity2.text in group1)):
                        return Relation(
                            source=entity1.text,  # 텍스트를 ID로 사용
                            target=entity2.text,  # 텍스트를 ID로 사용
                            relation=relation_type,
                            confidence=0.8,
                            context=match.group()
                        )
        
        # 기본 패턴으로 폴백
        if entity1.text in text and entity2.text in text:
            return Relation(
                source=entity1.text,  # 텍스트를 ID로 사용
                target=entity2.text,  # 텍스트를 ID로 사용
                relation="관련",
                confidence=0.5,
                context=f"{entity1.text}와 {entity2.text}가 같은 문맥에 등장"
            )
        
        return None
    
    def _extract_relations_by_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """패턴 기반 관계 추출"""
        relations = []
        
        # 의미 있는 관계만 추출 (기본 연결 제거)
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relation = self._extract_relation_between_entities(text, entity1, entity2)
                if relation:
                    relations.append(relation)
        
        return relations
    
    def _extract_relations_by_embeddings(self, text: str, entities: List[Entity]) -> List[Relation]:
        """임베딩 기반 관계 추출 (GPU 가속)"""
        relations = []
        
        try:
            if not self.sentence_model or len(entities) < 2:
                return relations
            
            # GPU 사용 확인
            device = next(self.sentence_model.parameters()).device
            logger.info(f"Embedding extraction using device: {device}")
            
            # 엔티티 쌍별 유사도 계산 (GPU에서 수행)
            entity_texts = [e.text for e in entities]
            embeddings = self.sentence_model.encode(
                entity_texts,
                convert_to_tensor=True,  # GPU tensor로 변환
                show_progress_bar=False
            )
            
            # GPU에서 유사도 계산
            import torch
            if torch.cuda.is_available() and device.type == 'cuda':
                embeddings = embeddings.to(device)
            
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # GPU에서 코사인 유사도 계산
                    similarity = torch.cosine_similarity(
                        embeddings[i].unsqueeze(0), 
                        embeddings[j].unsqueeze(0)
                    ).item()
                    
                    if similarity > 0.7:  # 높은 유사도
                        relations.append(Relation(
                            source=entity1.text,  # 텍스트를 ID로 사용
                            target=entity2.text,  # 텍스트를 ID로 사용
                            relation="유사",
                            confidence=float(similarity),
                            context=f"임베딩 유사도 (GPU 가속, 유사도: {similarity:.3f})"
                        ))
        
        except Exception as e:
            logger.error(f"Embedding-based relation extraction failed: {e}")
        
        return relations
    
    def _extract_relations_fallback(self, entities: List[Entity]) -> List[Relation]:
        """폴백 관계 추출 - 의미 있는 관계만"""
        relations = []
        
        # 폴백에서도 무의미한 관계 생성하지 않음
        logger.info("Using fallback relation extraction - no automatic relations")
        
        return relations
    
    def _extract_date_attributes(self, text: str, entity: Entity) -> Dict[str, Any]:
        """날짜 속성 추출"""
        attributes = {}
        
        # 엔티티 주변 텍스트에서 날짜 찾기
        context_start = max(0, entity.start - 50)
        context_end = min(len(text), entity.end + 50)
        context = text[context_start:context_end]
        
        date_patterns = [
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, context)
            if matches:
                attributes['date'] = matches[0]
                break
        
        return attributes
    
    def _extract_numeric_attributes(self, text: str, entity: Entity) -> Dict[str, Any]:
        """수치 속성 추출"""
        attributes = {}
        
        context_start = max(0, entity.start - 30)
        context_end = min(len(text), entity.end + 30)
        context = text[context_start:context_end]
        
        # 숫자 + 단위 패턴
        numeric_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:원|달러|명|개|년|월|일|시간|분|초|미터|킬로|톤)',
            r'\d+(?:,\d{3})*(?:\.\d+)?%',
            r'\d+(?:,\d{3})*(?:\.\d+)?'
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, context)
            if matches:
                attributes['numeric_value'] = matches[0]
                break
        
        return attributes
    
    def _extract_location_attributes(self, text: str, entity: Entity) -> Dict[str, Any]:
        """위치 속성 추출"""
        attributes = {}
        
        if entity.label in ["LOC", "GPE", "위치", "장소"]:
            attributes['location_type'] = entity.label
            
            # 주소 패턴
            context_start = max(0, entity.start - 50)
            context_end = min(len(text), entity.end + 50)
            context = text[context_start:context_end]
            
            address_pattern = r'[가-힣]+(?:시|도|구|군|동|읍|면|리)\s*[가-힣0-9\-\s]*'
            matches = re.findall(address_pattern, context)
            if matches:
                attributes['address'] = matches[0]
        
        return attributes
    
    def _extract_category_attributes(self, text: str, entity: Entity) -> Dict[str, Any]:
        """카테고리 속성 추출"""
        attributes = {}
        
        # 엔티티 타입별 카테고리
        category_mapping = {
            "PERSON": "인물",
            "ORG": "조직",
            "LOC": "위치",
            "DATE": "날짜",
            "MONEY": "금액",
            "PERCENT": "비율"
        }
        
        if entity.label in category_mapping:
            attributes['category'] = category_mapping[entity.label]
        
        return attributes
    
    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """엔티티를 딕셔너리로 변환"""
        return {
            "id": entity.id,
            "text": entity.text,
            "label": entity.label,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence,
            "attributes": entity.attributes
        }
    
    def _relation_to_dict(self, relation: Relation) -> Dict[str, Any]:
        """관계를 딕셔너리로 변환"""
        return {
            "source": relation.source,
            "target": relation.target,
            "relation": relation.relation,
            "confidence": relation.confidence,
            "context": relation.context,
            "attributes": relation.attributes
        }
    
    def _get_processing_stats(self, entities: List[Entity], relations: List[Relation]) -> Dict[str, Any]:
        """처리 통계"""
        return {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_types": list(set([e.label for e in entities])),
            "relation_types": list(set([r.relation for r in relations])),
            "avg_entity_confidence": np.mean([e.confidence for e in entities]) if entities else 0,
            "avg_relation_confidence": np.mean([r.confidence for r in relations]) if relations else 0
        }
    
    def _fallback_processing(self, text: str) -> Dict[str, Any]:
        """폴백 처리"""
        logger.warning("Using fallback processing")
        
        # 기본 엔티티 추출
        words = text.split()
        entities = []
        for i, word in enumerate(words[:5]):  # 처음 5개 단어를 엔티티로
            entities.append({
                "id": f"fallback_entity_{i}",
                "text": word,
                "label": "MISC",
                "start": 0,
                "end": len(word),
                "confidence": 0.3,
                "attributes": {}
            })
        
        # 기본 관계
        relations = []
        if len(entities) >= 2:
            relations.append({
                "source": entities[0]["id"],
                "target": entities[1]["id"],
                "relation": "관련",
                "confidence": 0.3,
                "context": "폴백 처리",
                "attributes": {}
            })
        
        return {
            "original_text": text,
            "preprocessed_text": text,
            "entities": entities,
            "relations": relations,
            "graph": {
                "nodes": [
                    {
                        **e,
                        "node_type": e["label"],
                        "x": np.random.randint(100, 700),
                        "y": np.random.randint(100, 600)
                    } for e in entities
                ],
                "edges": [
                    {
                        "id": f"edge_{i}",
                        **r
                    } for i, r in enumerate(relations)
                ],
                "summary": {
                    "node_count": len(entities),
                    "edge_count": len(relations),
                    "node_types": ["MISC"],
                    "processing_language": self.language,
                    "confidence_avg": 0.3
                }
            },
            "processing_stats": {
                "total_entities": len(entities),
                "total_relations": len(relations),
                "entity_types": ["MISC"],
                "relation_types": ["관련"],
                "avg_entity_confidence": 0.3,
                "avg_relation_confidence": 0.3
            }
        } 