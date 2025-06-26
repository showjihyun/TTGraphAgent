import os
import json
import logging
from typing import Dict, Any, List
from llm.llama3_handler import Llama3Handler

logger = logging.getLogger(__name__)

# 환경변수에서 OpenAI API 키를 읽어옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EXTRACT_PROMPT = """
아래 문서에서 엔티티, 관계, 이벤트를 JSON 형태로 추출해 주세요.
- entities: 주요 명사(인물, 조직, 장소 등)
- relations: 주어-관계-목적 구조
- events: 시점/이벤트 정보

문서:
"""

llama3 = Llama3Handler()

def extract_entities_relations_events(text: str) -> dict:
    prompt = EXTRACT_PROMPT + text + "\nJSON 결과만 출력하세요."
    try:
        content = llama3.chat(prompt)
        import json
        result = json.loads(content)
        return result
    except Exception as e:
        return {"error": str(e), "raw": content if 'content' in locals() else None}

class LLMExtractor:
    """LLM 기반 엔티티 및 관계 추출기"""
    
    def __init__(self):
        self.llama3 = Llama3Handler()
        
    async def extract_entities_and_relations(self, text: str, language: str = "korean") -> Dict[str, Any]:
        """텍스트에서 엔티티와 관계를 추출하여 그래프 형식으로 반환"""
        try:
            if language == "korean":
                prompt = f"""
다음 텍스트에서 엔티티와 관계를 추출하여 지식 그래프를 생성해주세요.

텍스트:
{text}

다음 JSON 형식으로 결과를 출력해주세요:
{{
    "nodes": [
        {{
            "id": "node_1",
            "label": "엔티티명",
            "node_type": "PERSON|ORG|LOC|MISC",
            "x": 100,
            "y": 100
        }}
    ],
    "edges": [
        {{
            "id": "edge_1",
            "source": "node_1",
            "target": "node_2",
            "relation": "관계명"
        }}
    ],
    "summary": {{
        "node_count": 2,
        "edge_count": 1,
        "node_types": ["PERSON", "ORG"],
        "processing_language": "korean"
    }}
}}

JSON 결과만 출력하세요:
"""
            else:
                prompt = f"""
Extract entities and relations from the following text to create a knowledge graph.

Text:
{text}

Output the result in the following JSON format:
{{
    "nodes": [
        {{
            "id": "node_1",
            "label": "entity_name",
            "node_type": "PERSON|ORG|LOC|MISC",
            "x": 100,
            "y": 100
        }}
    ],
    "edges": [
        {{
            "id": "edge_1",
            "source": "node_1",
            "target": "node_2",
            "relation": "relation_name"
        }}
    ],
    "summary": {{
        "node_count": 2,
        "edge_count": 1,
        "node_types": ["PERSON", "ORG"],
        "processing_language": "english"
    }}
}}

Output only JSON result:
"""
            
            # LLM 호출
            content = self.llama3.chat(prompt)
            logger.info(f"LLM raw response: {content[:200]}...")  # 처음 200자만 로깅
            
            # JSON 파싱
            try:
                result = json.loads(content)
                
                # 기본 구조 검증 및 보정
                if not isinstance(result, dict):
                    raise ValueError("Result is not a dictionary")
                
                # 필수 필드 확인
                if "nodes" not in result:
                    result["nodes"] = []
                if "edges" not in result:
                    result["edges"] = []
                if "summary" not in result:
                    result["summary"] = {
                        "node_count": len(result["nodes"]),
                        "edge_count": len(result["edges"]),
                        "node_types": [],
                        "processing_language": language
                    }
                
                # 노드 좌표 보정
                for i, node in enumerate(result["nodes"]):
                    if "x" not in node or "y" not in node:
                        import random
                        node["x"] = random.randint(100, 700)
                        node["y"] = random.randint(100, 600)
                
                logger.info(f"LLM extraction completed: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                return self._create_fallback_result(text, language)
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._create_fallback_result(text, language)
    
    def _create_fallback_result(self, text: str, language: str) -> Dict[str, Any]:
        """LLM 추출 실패시 폴백 결과 생성"""
        logger.warning(f"Creating fallback result for LLM extraction failure")
        
        # 규칙 기반 엔티티 추출 (한국어 패턴)
        nodes = []
        edges = []
        
        if language == "korean":
            import re
            
            # 한국어 엔티티 패턴
            person_pattern = r'([가-힣]{2,4})\s*(?:교수|박사|선생|씨|님|대표|회장|사장)'
            org_pattern = r'([가-힣]{2,10}(?:대학교|대학|회사|기업|연구소|학교|병원))'
            
            # 인물 추출
            persons = re.findall(person_pattern, text)
            for i, person in enumerate(persons):
                nodes.append({
                    "id": f"person_{i}",
                    "label": f"{person} 교수" if "교수" in text else person,
                    "group": "PERSON",
                    "x": 100 + i * 200,
                    "y": 200
                })
            
            # 조직 추출
            orgs = re.findall(org_pattern, text)
            for i, org in enumerate(orgs):
                nodes.append({
                    "id": f"org_{i}",
                    "label": org,
                    "group": "ORG", 
                    "x": 100 + i * 200,
                    "y": 400
                })
            
            # 관계 생성 (인물-조직)
            if len(nodes) >= 2:
                for i, person_node in enumerate([n for n in nodes if n["group"] == "PERSON"]):
                    for j, org_node in enumerate([n for n in nodes if n["group"] == "ORG"]):
                        edges.append({
                            "id": f"relation_{i}_{j}",
                            "from": person_node["label"],
                            "to": org_node["label"],
                            "label": "소속"
                        })
        
        # 결과가 없으면 빈 결과 반환 (fallback_node 사용 안함)
        if not nodes:
            logger.info("No entities found, returning empty result")
            return {
                "nodes": [],
                "edges": [],
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "node_types": [],
                    "processing_language": language,
                    "total_entities": 0,
                    "total_relations": 0
                }
            }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_types": list(set([n["group"] for n in nodes])),
                "processing_language": language,
                "total_entities": len(nodes),
                "total_relations": len(edges)
            }
        } 