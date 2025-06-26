from fastapi import APIRouter

router = APIRouter()

@router.post("/graph/upload")
async def upload_graph(data: dict):
    # TODO: Neo4j에 그래프 저장 구현
    return {"status": "success"}

@router.get("/graph/query")
async def query_graph():
    # TODO: Neo4j에서 그래프 조회 구현
    return {"nodes": [], "edges": []} 