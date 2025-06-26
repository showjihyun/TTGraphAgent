from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import extract, graph, texttograph

app = FastAPI()

# CORS 설정 (프론트엔드와 연동 시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router)
app.include_router(graph.router)
app.include_router(texttograph.router)

@app.get("/")
def root():
    return {"message": "DBGraphAgent Backend is running"} 