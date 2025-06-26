# DBGraphAgent

**DBGraphAgent**는 자연어 텍스트를 엔티티-관계 기반의 그래프 구조로 변환하는 AI 파이프라인 플랫폼입니다.  
한국어/영어 입력을 지원하며, GPU 가속, Hybrid(NLP+LLM) 모드, LLM Only 모드, 웹 기반 시각화, LLM 연동을 제공합니다.
현재, 프로포 타입이니 참고하시기 바랍니다.

---
## 테스트 화면

![image](https://github.com/user-attachments/assets/6d23a7e2-66fc-47b1-a5ac-3e4e38acc1e0)

---
---

## 주요 기능

- **5단계 NLP 파이프라인**: 전처리 → 엔티티 추출(NER) → 관계 추출(RE) → 속성 추출 → 구조화/LLM 보정
- **Hybrid 모드**: 고성능 NLP + LLM 보정 결합 (정확도/일관성 ↑)
- **GPU 가속**: SentenceTransformer, HuggingFace NER 등에서 GPU 자동 사용
- **Ollama LLM 연동**: llama3 등 로컬 LLM 활용 (LLM Only 모드)
- **React + Vite 프론트엔드**: 입력/결과 시각화, 다양한 처리 옵션 제공
- **REST API**: FastAPI 기반, `/api/text-to-graph` 등 엔드포인트 제공

---

## 폴더 구조

```
DBGraphAgent/
  backend/         # FastAPI 서버, NLP/LLM 파이프라인, 서비스 로직
    llm/           # LLM 핸들러 (llama3, chatgpt 등)
    routers/       # API 라우터
    services/      # NLP/LLM 서비스
    test_gpu_web.html  # GPU/Hybrid/LLM 테스트용 웹페이지
    requirements.txt
    main.py
  frontend/        # React + TypeScript + Vite 프론트엔드
    src/
    public/
    package.json
    ...
  docs/            # 기획/설계 문서
```

---

## 설치 및 실행

### 1. Python 백엔드 (FastAPI)

```bash
cd DBGraphAgent/backend
pip install -r requirements.txt
# (CUDA GPU 사용 권장, torch는 cu121 등 GPU 버전 설치)
python main.py
```

### 2. Ollama LLM 서버 (로컬 LLM)

```bash
ollama serve
ollama pull llama3
```

### 3. 프론트엔드 (React + Vite)

```bash
cd DBGraphAgent/frontend
npm install
npm run dev
```

### 4. 테스트 웹페이지 (로컬 HTML)

- `DBGraphAgent/backend/test_gpu_web.html` 파일을 브라우저에서 열기

---

## 주요 API

- `POST /api/text-to-graph`
  - 입력: `{ "text": "텍스트", "language": "korean", "processing_method": "hybrid|nlp_only|llm_only" }`
  - 출력: 엔티티/관계 그래프 JSON

- `GET /api/health`  
  - 시스템 상태/가용성 확인

---

## 처리 모드

- **hybrid**: NLP + LLM 보정 (권장, GPU 가속)
- **nlp_only**: NLP 파이프라인만 (가장 빠름, GPU 가속)
- **llm_only**: LLM만 사용 (Ollama/llama3 필요)

---

## 예시 결과

- 입력:  
  `김철수 교수는 서울대학교에서 인공지능을 연구하고 있다.`
- Hybrid/NLP Only 결과:  
  - 엔티티: `김철수 교수 (PERSON)`, `서울대학교 (ORG)`
  - 관계: `김철수 교수 → 서울대학교 (소속/관련)`

---

## 참고/기타

- Ollama LLM 서버가 반드시 실행 중이어야 LLM Only/Hybrid 모드가 정상 동작합니다.
- GPU가 없으면 CPU로 자동 폴백됩니다.
- 반박시 니말이 맞음.

---

## 라이선스

MIT License

---
