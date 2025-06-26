from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from services.llm_extractor import extract_entities_relations_events

router = APIRouter()

@router.post("/extract")
async def extract_entities(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")
    text = (await file.read()).decode("utf-8")
    result = await run_in_threadpool(extract_entities_relations_events, text)
    return result 