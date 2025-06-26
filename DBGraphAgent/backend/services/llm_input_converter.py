from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import networkx as nx
from typing import Dict, List, Optional
import re
from dataclasses import dataclass, asdict
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """그래프 노드를 나타내는 클래스"""
    id: str
    label: str
    node_type: str  # entity, concept, action 등
    x: Optional[float] = None
    y: Optional[float] = None

@dataclass
class GraphEdge:
    """그래프 엣지를 나타내는 클래스"""
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0

class GraphResponse(BaseModel):
    """API 응답 모델"""
    nodes: List[Dict]
    edges: List[Dict]
    summary: Dict

class TextInput(BaseModel):
    """텍스트 입력 모델"""
    text: str
    language: str = "korean"

EXTRACT_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction expert. "
    "Extract entities, relations, and events from the given text and return ONLY valid JSON. "
    "Do not include any explanation, markdown, or extra text. "
    "Return only a valid JSON object as specified."
)

EXTRACT_USER_PROMPT = (
    "아래 문서에서 엔티티, 관계, 이벤트를 JSON 형태로 추출해 주세요.\n"
    "- entities: 주요 명사(인물, 조직, 장소 등)\n"
    "- relations: 주어-관계-목적 구조\n"
    "- events: 시점/이벤트 정보\n"
    "문서:\n"
)

class TextToGraphConverter:
    """한글 텍스트를 그래프로 변환하는 클래스"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model
        
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Ollama API 호출"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API 오류: {response.status_code}")
                
        except Exception as e:
            logger.error(f"LLM 호출 오류: {e}")
            raise
    
    def translate_to_english(self, korean_text: str) -> str:
        """한글 텍스트를 영어로 번역"""
        system_prompt = """You are a professional Korean-English translator. 
        Translate the given Korean text to English accurately while preserving the meaning and context.
        Only return the English translation, nothing else."""
        
        prompt = f"Translate this Korean text to English:\n\n{korean_text}"
        
        english_text = self._call_ollama(prompt, system_prompt)
        return english_text.strip()
    
    def translate_to_korean(self, english_text: str) -> str:
        """영어 텍스트를 한글로 번역"""
        system_prompt = """You are a professional English-Korean translator. 
        Translate the given English text to Korean accurately while preserving the meaning and context.
        Only return the Korean translation, nothing else."""
        
        prompt = f"Translate this English text to Korean:\n\n{english_text}"
        
        korean_text = self._call_ollama(prompt, system_prompt)
        return korean_text.strip()
    
    def extract_graph_structure(self, text: str) -> tuple[List[GraphNode], List[GraphEdge]]:
        """텍스트에서 그래프 구조 추출"""
        system_prompt = """You are an expert in knowledge graph extraction. 
        Extract entities, concepts, and relationships from the given text and format them as a structured graph.
        
        Return the result in the following JSON format:
        {
            "nodes": [
                {"id": "node1", "label": "Entity Name", "type": "entity"},
                {"id": "node2", "label": "Concept Name", "type": "concept"},
                {"id": "node3", "label": "Action Name", "type": "action"}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "relation": "relationship_type", "weight": 0.8}
            ]
        }
        
        Guidelines:
        - Extract key entities (people, places, organizations, objects)
        - Extract important concepts and ideas
        - Extract actions and processes
        - Identify relationships between entities and concepts
        - Use clear, descriptive relationship labels
        - Assign appropriate weights (0.1 to 1.0) based on relationship strength
        - Create meaningful node IDs (use descriptive names)
        - Limit to most important nodes (max 15) and edges (max 25)
        - Ensure all edge sources and targets exist in nodes
        """
        
        prompt = f"""Extract a knowledge graph from this text:

{text}

Return only the JSON structure as specified. Make sure the JSON is valid and complete."""
        
        response = self._call_ollama(prompt, system_prompt)
        
        try:
            # JSON 부분만 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                graph_data = json.loads(json_str)
            else:
                raise ValueError("JSON 구조를 찾을 수 없습니다.")
            
            nodes = []
            edges = []
            
            # 노드 처리
            for i, node in enumerate(graph_data.get("nodes", [])):
                nodes.append(GraphNode(
                    id=node.get("id", f"node_{i+1}"),
                    label=node.get("label", f"Node {i+1}"),
                    node_type=node.get("type", "entity")
                ))
            
            # 엣지 처리 (노드 ID 검증)
            node_ids = {node.id for node in nodes}
            for i, edge in enumerate(graph_data.get("edges", [])):
                source = edge.get("source", "")
                target = edge.get("target", "")
                
                if source in node_ids and target in node_ids:
                    edges.append(GraphEdge(
                        id=f"edge_{i+1}",
                        source=source,
                        target=target,
                        relation=edge.get("relation", "관련"),
                        weight=edge.get("weight", 1.0)
                    ))
            
            return nodes, edges
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON 파싱 오류: {e}. 대체 방법 사용...")
            return self._fallback_extraction(response, text)
    
    def _fallback_extraction(self, response: str, original_text: str) -> tuple[List[GraphNode], List[GraphEdge]]:
        """JSON 파싱 실패시 대체 추출 방법"""
        nodes = []
        edges = []
        
        # 원본 텍스트에서 키워드 추출
        words = re.findall(r'\b\w+\b', original_text)
        important_words = [word for word in words if len(word) > 2][:10]
        
        # 노드 생성
        for i, word in enumerate(set(important_words)):
            nodes.append(GraphNode(
                id=f"node_{i+1}",
                label=word,
                node_type="entity"
            ))
        
        # 간단한 엣지 생성 (순차적 연결)
        for i in range(len(nodes) - 1):
            edges.append(GraphEdge(
                id=f"edge_{i+1}",
                source=nodes[i].id,
                target=nodes[i+1].id,
                relation="관련",
                weight=0.5
            ))
        
        return nodes, edges
    
    def calculate_layout(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> List[GraphNode]:
        """NetworkX를 사용하여 레이아웃 계산"""
        if not nodes:
            return nodes
            
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        for node in nodes:
            G.add_node(node.id)
        
        for edge in edges:
            if edge.source in G.nodes and edge.target in G.nodes:
                G.add_edge(edge.source, edge.target)
        
        # 레이아웃 계산
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=3, iterations=50, scale=500)
            
            # 노드에 위치 정보 추가
            for node in nodes:
                if node.id in pos:
                    node.x = float(pos[node.id][0])
                    node.y = float(pos[node.id][1])
                else:
                    node.x = 0.0
                    node.y = 0.0
        
        return nodes
    
    def process_text(self, text: str, language: str = "korean") -> tuple[List[GraphNode], List[GraphEdge]]:
        """텍스트 처리 메인 함수"""
        try:
            if language == "korean":
                logger.info("한글 텍스트 처리 시작")
                
                # 1. 한글 → 영어 번역
                logger.info("1. 한글 → 영어 번역 중...")
                english_text = self.translate_to_english(text)
                logger.info(f"번역 완료: {english_text[:100]}...")
                
                # 2. 영어 텍스트에서 그래프 구조 추출
                logger.info("2. 그래프 구조 추출 중...")
                nodes, edges = self.extract_graph_structure(english_text)
                
                # 3. 노드와 엣지 라벨을 한글로 번역
                logger.info("3. 라벨 한글 번역 중...")
                for node in nodes:
                    if node.label and len(node.label.strip()) > 0:
                        korean_label = self.translate_to_korean(node.label)
                        node.label = korean_label.strip()
                
                for edge in edges:
                    if edge.relation and len(edge.relation.strip()) > 0:
                        korean_relation = self.translate_to_korean(edge.relation)
                        edge.relation = korean_relation.strip()
                
            else:
                # 영어 텍스트 직접 처리
                logger.info("영어 텍스트 직접 처리")
                nodes, edges = self.extract_graph_structure(text)
            
            # 4. 레이아웃 계산
            logger.info("4. 레이아웃 계산 중...")
            nodes = self.calculate_layout(nodes, edges)
            
            logger.info(f"처리 완료: 노드 {len(nodes)}개, 엣지 {len(edges)}개")
            return nodes, edges
            
        except Exception as e:
            logger.error(f"텍스트 처리 오류: {e}")
            raise

    def extract_entities_relations_events(self, text: str) -> dict:
        prompt = EXTRACT_USER_PROMPT + text + "\n반드시 JSON만, 설명 없이 출력하세요."
        # Llama3Handler 등 LLM handler 사용
        content = self._call_ollama(prompt)
        try:
            result = json.loads(content)
            return result
        except Exception as e:
            # fallback
            return {"error": str(e), "raw": content}

# FastAPI 앱 생성
app = FastAPI(title="TextToGraph API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 컨버터 인스턴스
converter = TextToGraphConverter()

@app.get("/")
async def root():
    return {"message": "TextToGraph API Server"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # Ollama 연결 테스트
        response = requests.get(f"{converter.ollama_url}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "model": converter.model
    }

@app.post("/api/text-to-graph", response_model=GraphResponse)
async def text_to_graph(input_data: TextInput):
    """텍스트를 그래프로 변환하는 API"""
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
        
        logger.info(f"텍스트 처리 요청: {input_data.text[:100]}...")
        
        # 그래프 구조 추출
        nodes, edges = converter.process_text(input_data.text, input_data.language)
        
        # 응답 데이터 준비
        nodes_dict = [asdict(node) for node in nodes]
        edges_dict = [asdict(edge) for edge in edges]
        
        summary = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": list(set(node.node_type for node in nodes)),
            "processing_language": input_data.language
        }
        
        return GraphResponse(
            nodes=nodes_dict,
            edges=edges_dict,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_available_models():
    """사용 가능한 모델 목록 조회"""
    try:
        response = requests.get(f"{converter.ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return {"models": model_names, "current_model": converter.model}
        else:
            return {"models": [], "current_model": converter.model, "error": "Ollama 연결 실패"}
    except Exception as e:
        return {"models": [], "current_model": converter.model, "error": str(e)}

@app.post("/api/set-model")
async def set_model(model_data: dict):
    """사용할 모델 변경"""
    try:
        new_model = model_data.get("model")
        if not new_model:
            raise HTTPException(status_code=400, detail="모델명이 필요합니다.")
        
        converter.model = new_model
        return {"message": f"모델이 {new_model}로 변경되었습니다.", "current_model": converter.model}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
