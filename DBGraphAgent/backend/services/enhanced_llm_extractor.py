"""
Enhanced LLM Extractor with 5-Stage NLP Pipeline Integration
Combines advanced NLP processing with LLM-based validation and correction
"""

import json
import logging
from typing import Dict, Any, List, Optional
from services.advanced_nlp_processor import AdvancedNLPProcessor
from services.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

class EnhancedLLMExtractor:
    """향상된 LLM 추출기 - NLP + LLM 하이브리드"""
    
    def __init__(self):
        self.nlp_processor = None
        self.llm_extractor = LLMExtractor()
        
    def initialize_nlp_processor(self, language: str = "korean"):
        """NLP 프로세서 초기화"""
        try:
            self.nlp_processor = AdvancedNLPProcessor(language)
            logger.info(f"NLP processor initialized for {language}")
        except Exception as e:
            logger.error(f"Failed to initialize NLP processor: {e}")
            self.nlp_processor = None
    
    async def extract_knowledge_graph(self, text: str, language: str = "korean") -> Dict[str, Any]:
        """
        5단계 + LLM 보정 파이프라인
        1-4단계: NLP 처리 → 5단계: LLM 보정
        """
        try:
            # NLP 프로세서 초기화 (필요시)
            if self.nlp_processor is None:
                self.initialize_nlp_processor(language)
            
            # 1-4단계: Advanced NLP Processing
            if self.nlp_processor:
                nlp_result = self.nlp_processor.process_text(text)
                logger.info(f"NLP processing completed: {nlp_result['processing_stats']}")
                
                # 5단계: LLM 보정
                enhanced_result = await self._llm_enhancement(text, nlp_result, language)
                return enhanced_result
            else:
                # NLP 처리 실패시 LLM만 사용
                logger.warning("NLP processor unavailable, using LLM only")
                return await self.llm_extractor.extract_entities_and_relations(text, language)
                
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}")
            # 폴백: 기존 LLM 추출기 사용
            return await self.llm_extractor.extract_entities_and_relations(text, language)
    
    async def _llm_enhancement(self, original_text: str, nlp_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """5단계: LLM 기반 보정 및 검증"""
        try:
            # NLP 결과를 LLM 프롬프트에 포함
            enhancement_prompt = self._create_enhancement_prompt(original_text, nlp_result, language)
            
            # LLM으로 보정 요청
            llm_correction = await self.llm_extractor.extract_entities_and_relations(
                enhancement_prompt, language
            )
            
            # NLP 결과와 LLM 보정 결과 융합
            final_result = self._merge_nlp_and_llm_results(nlp_result, llm_correction)
            
            return final_result
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return self._format_nlp_result_as_final(nlp_result)
    
    def _create_enhancement_prompt(self, original_text: str, nlp_result: Dict[str, Any], language: str) -> str:
        """LLM 보정을 위한 프롬프트 생성"""
        
        if language == "korean":
            prompt = f"""
다음 텍스트에서 중요한 엔티티(인물, 조직, 장소, 개념 등)와 그들 간의 관계를 추출해주세요.

텍스트:
{original_text}

요구사항:
1. 텍스트에 명시적으로 언급된 엔티티만 추출
2. 실제로 의미 있는 관계만 추출
3. 추측이나 가정은 하지 말고 텍스트 기반으로만 추출
4. 단순한 단어가 아닌 의미 있는 개체만 포함

JSON 형식으로 결과를 출력해주세요:
"""
        else:
            prompt = f"""
Extract important entities (people, organizations, places, concepts) and their relationships from the following text.

Text:
{original_text}

Requirements:
1. Extract only entities explicitly mentioned in the text
2. Extract only meaningful relationships
3. Base extraction on text content only, no assumptions
4. Include only significant entities, not simple words

Output the results in JSON format:
"""
        
        return prompt
    
    def _format_entities_for_prompt(self, entities: List[Dict]) -> str:
        """엔티티를 프롬프트용으로 포맷팅"""
        if not entities:
            return "없음"
        
        formatted = []
        for entity in entities:
            formatted.append(
                f"- {entity.get('text', '')} ({entity.get('label', '')}) "
                f"[신뢰도: {entity.get('confidence', 0):.2f}]"
            )
        return "\n".join(formatted)
    
    def _format_relations_for_prompt(self, relations: List[Dict]) -> str:
        """관계를 프롬프트용으로 포맷팅"""
        if not relations:
            return "없음"
        
        formatted = []
        for relation in relations:
            formatted.append(
                f"- {relation.get('source', '')} --[{relation.get('relation', '')}]--> "
                f"{relation.get('target', '')} [신뢰도: {relation.get('confidence', 0):.2f}]"
            )
        return "\n".join(formatted)
    
    def _merge_nlp_and_llm_results(self, nlp_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """NLP 결과와 LLM 보정 결과 융합"""
        try:
            # LLM 결과가 유효한 경우 우선 사용
            if self._is_valid_llm_result(llm_result):
                # LLM 결과에 NLP의 상세 정보 추가
                enhanced_result = llm_result.copy()
                
                # NLP 속성 정보 추가
                enhanced_result = self._add_nlp_attributes(enhanced_result, nlp_result)
                
                # 메타데이터 추가
                enhanced_result['processing_method'] = 'nlp_llm_hybrid'
                enhanced_result['nlp_stats'] = nlp_result.get('processing_stats', {})
                
                return enhanced_result
            else:
                # LLM 보정 실패시 NLP 결과 사용
                return self._format_nlp_result_as_final(nlp_result)
                
        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            return self._format_nlp_result_as_final(nlp_result)
    
    def _is_valid_llm_result(self, llm_result: Dict[str, Any]) -> bool:
        """LLM 결과 유효성 검증"""
        try:
            if not (isinstance(llm_result, dict) and
                    'nodes' in llm_result and
                    'edges' in llm_result and
                    isinstance(llm_result['nodes'], list) and
                    isinstance(llm_result['edges'], list)):
                return False
            
            # fallback 결과는 유효하지 않은 것으로 간주
            nodes = llm_result['nodes']
            if len(nodes) == 0:
                return False
                
            # fallback_node가 포함된 경우 무효 처리
            for node in nodes:
                if (isinstance(node.get('id'), str) and 
                    node['id'].startswith('fallback_node')):
                    logger.warning("LLM result contains fallback nodes, treating as invalid")
                    return False
                    
                # undefined 값이 있는 경우도 무효 처리
                if (node.get('label') is None or 
                    node.get('node_type') is None or
                    str(node.get('label')).strip() == '' or
                    str(node.get('node_type')).strip() == ''):
                    logger.warning("LLM result contains undefined values, treating as invalid")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"LLM result validation failed: {e}")
            return False
    
    def _add_nlp_attributes(self, llm_result: Dict[str, Any], nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 결과에 NLP 속성 정보 추가"""
        try:
            # 엔티티별 속성 매핑
            nlp_entities = {e.get('text', ''): e for e in nlp_result.get('entities', [])}
            
            for node in llm_result.get('nodes', []):
                node_label = node.get('label', '')
                if node_label in nlp_entities:
                    nlp_entity = nlp_entities[node_label]
                    # NLP 속성 추가
                    node['nlp_attributes'] = nlp_entity.get('attributes', {})
                    node['nlp_confidence'] = nlp_entity.get('confidence', 0)
                    node['character_position'] = {
                        'start': nlp_entity.get('start', 0),
                        'end': nlp_entity.get('end', 0)
                    }
            
            return llm_result
            
        except Exception as e:
            logger.error(f"Adding NLP attributes failed: {e}")
            return llm_result
    
    def _format_nlp_result_as_final(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """NLP 결과를 최종 형식으로 변환"""
        try:
            graph = nlp_result.get('graph', {})
            
            # 노드와 엣지 추출
            nodes = graph.get('nodes', [])
            edges = graph.get('edges', [])
            
            # 통계 정보 계산
            processing_stats = nlp_result.get('processing_stats', {})
            
            # 기본 구조 보장
            final_result = {
                'nodes': nodes,
                'edges': edges,
                'summary': {
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'node_types': list(set([node.get('group', 'MISC') for node in nodes])),
                    'processing_language': 'korean',
                    'total_entities': processing_stats.get('total_entities', len(nodes)),
                    'total_relations': processing_stats.get('total_relations', len(edges)),
                    'entity_types': processing_stats.get('entity_types', []),
                    'relation_types': processing_stats.get('relation_types', [])
                },
                'processing_method': 'hybrid',  # hybrid 모드에서 NLP 결과 사용
                'nlp_stats': processing_stats,
                'original_text': nlp_result.get('original_text', ''),
                'preprocessed_text': nlp_result.get('preprocessed_text', '')
            }
            
            logger.info(f"NLP result formatted: {len(nodes)} nodes, {len(edges)} edges")
            return final_result
            
        except Exception as e:
            logger.error(f"NLP result formatting failed: {e}")
            return {
                'nodes': [],
                'edges': [],
                'summary': {
                    'node_count': 0,
                    'edge_count': 0,
                    'node_types': [],
                    'processing_language': 'korean',
                    'total_entities': 0,
                    'total_relations': 0
                },
                'processing_method': 'hybrid'
            }
    
    def get_processing_info(self) -> Dict[str, Any]:
        """처리 정보 반환"""
        return {
            'nlp_processor_available': self.nlp_processor is not None,
            'llm_extractor_available': self.llm_extractor is not None,
            'supported_languages': ['korean', 'english'],
            'processing_stages': [
                '1. 전처리 (문장 분리 및 정제)',
                '2. 엔티티 추출 (NER)',
                '3. 관계 추출 (RE)',
                '4. 속성 추출',
                '5. LLM 보정 및 검증'
            ]
        } 