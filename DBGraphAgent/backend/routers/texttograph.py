from fastapi import APIRouter, HTTPException
from services.llm_input_converter import TextToGraphConverter, GraphResponse, TextInput
import requests
import logging
from pydantic import BaseModel
from typing import Optional

from services.enhanced_llm_extractor import EnhancedLLMExtractor

logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 컨버터 인스턴스
converter = TextToGraphConverter()

# 전역 추출기 인스턴스
enhanced_extractor = EnhancedLLMExtractor()

class TextToGraphRequest(BaseModel):
    text: str
    language: Optional[str] = "korean"
    processing_method: Optional[str] = "hybrid"  # "hybrid", "nlp_only", "llm_only"

class ProcessingInfoResponse(BaseModel):
    nlp_available: bool
    llm_available: bool
    supported_languages: list
    processing_stages: list
    supported_methods: list

@router.get("/api/models")
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

@router.post("/api/set-model")
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

@router.post("/api/text-to-graph")
async def convert_text_to_graph(request: TextToGraphRequest):
    """
    텍스트를 지식 그래프로 변환
    
    Processing Methods:
    - hybrid: NLP 파이프라인 + LLM 보정 (기본값, 가장 정확)
    - nlp_only: NLP 파이프라인만 사용 (빠름)
    - llm_only: LLM만 사용 (기존 방식)
    """
    try:
        logger.info(f"Processing text with method: {request.processing_method}")
        logger.info(f"Text length: {len(request.text)} characters")
        
        if request.processing_method == "llm_only":
            # 기존 LLM 전용 처리
            result = await enhanced_extractor.llm_extractor.extract_entities_and_relations(
                request.text, request.language
            )
            result['processing_method'] = 'llm_only'
            
        elif request.processing_method == "nlp_only":
            # NLP 전용 처리 (LLM 보정 없음)
            if enhanced_extractor.nlp_processor is None:
                enhanced_extractor.initialize_nlp_processor(request.language)
            
            if enhanced_extractor.nlp_processor:
                nlp_result = enhanced_extractor.nlp_processor.process_text(request.text)
                result = enhanced_extractor._format_nlp_result_as_final(nlp_result)
                result['processing_method'] = 'nlp_only'
            else:
                raise HTTPException(
                    status_code=503, 
                    detail="NLP processor not available. Try 'llm_only' method."
                )
        
        else:  # hybrid (기본값)
            # 5단계 하이브리드 처리
            result = await enhanced_extractor.extract_knowledge_graph(
                request.text, request.language
            )
        
        logger.info(f"Processing completed: {result.get('summary', {})}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text to graph conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/api/processing-info", response_model=ProcessingInfoResponse)
async def get_processing_info():
    """처리 방법 및 시스템 정보 조회"""
    try:
        info = enhanced_extractor.get_processing_info()
        
        return ProcessingInfoResponse(
            nlp_available=info['nlp_processor_available'],
            llm_available=info['llm_extractor_available'],
            supported_languages=info['supported_languages'],
            processing_stages=info['processing_stages'],
            supported_methods=[
                "hybrid - NLP 파이프라인 + LLM 보정 (권장)",
                "nlp_only - NLP 파이프라인만 사용",
                "llm_only - LLM만 사용 (기존 방식)"
            ]
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

@router.post("/api/initialize-nlp")
async def initialize_nlp_processor(language: str = "korean"):
    """NLP 프로세서 수동 초기화"""
    try:
        enhanced_extractor.initialize_nlp_processor(language)
        
        return {
            "message": f"NLP processor initialized for {language}",
            "available": enhanced_extractor.nlp_processor is not None
        }
        
    except Exception as e:
        logger.error(f"NLP initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.get("/api/health")
async def health_check():
    """시스템 상태 확인"""
    try:
        info = enhanced_extractor.get_processing_info()
        
        return {
            "status": "healthy",
            "nlp_processor": "available" if info['nlp_processor_available'] else "unavailable",
            "llm_extractor": "available" if info['llm_extractor_available'] else "unavailable",
            "recommended_method": "hybrid" if info['nlp_processor_available'] else "llm_only"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "recommended_method": "llm_only"
        } 