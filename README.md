# AI Text to Graph

**AI Text to Graph**는 자연어 텍스트를 엔터티-관계 기반 그래프 구조로 변환하는 AI 파이프라인 플랫폼입니다.  
한국어와 영어를 모두 지원하며, GPU 가속, 하이브리드(NLP+LLM) 모드, LLM Only 모드, 웹 기반 시각화, LLM 통합을 제공합니다.

---

## 주요 특징

- **5단계 NLP 파이프라인**: 전처리 → 엔터티 추출(NER) → 관계 추출(RE) → 속성 추출 → 구조화/LLM 보정
- **하이브리드 모드**: 고성능 NLP와 LLM 보정 결합 (정확도/일관성 향상)
- **GPU 가속**: SentenceTransformer, HuggingFace NER 등에서 자동 GPU 사용
- **Ollama LLM 통합**: llama3 등 로컬 LLM 지원 (LLM Only 모드)
- **현대적 UI/UX**: React + Vite 기반, 다크 테마, 카드형 디자인, 직관적 그래프 인터랙션
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
```

---

## 설치 및 실행

### 1. Python 백엔드 (FastAPI)

```bash
cd DBGraphAgent/backend
pip install -r requirements.txt
# (GPU 사용 시, CUDA 버전에 맞는 torch 별도 설치 권장)
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
  - 입력: `{ "text": "텍스트", "language": "korean|english", "processing_method": "hybrid|nlp_only|llm_only" }`
  - 출력: 엔터티/관계 그래프 JSON

- `GET /api/health`
  - 시스템 상태/가용성 체크

---

## 처리 모드

- **hybrid**: NLP + LLM 보정 (권장, GPU 가속)
- **nlp_only**: NLP 파이프라인만 사용 (가장 빠름, GPU 가속)
- **llm_only**: LLM만 사용 (Ollama/llama3 필요)

---

## 예시 출력

- 입력:  
  `Professor Kim Chulsoo is researching artificial intelligence at Seoul National University.`
- 출력(엔터티/관계):  
  - 엔터티: `Professor Kim Chulsoo (PERSON)`, `Seoul National University (ORG)`
  - 관계: `Professor Kim Chulsoo → Seoul National University (affiliation/related)`

---

## 프론트엔드 주요 기능

- 현대적 다크 테마, 중앙 집중형 레이아웃, 카드 기반 UI
- 그래프 영역이 항상 가장 크고 우측에 위치
- 노드 클릭 시 JSON 정보 패널 표시, 드래그로 노드 위치 조정 가능
- 줌/다운로드/로딩바 등 현대적 인터랙션 제공
- 반응형 레이아웃, 커스텀 favicon 및 브라우저 탭 타이틀

---

## 참고/기타

- LLM Only/Hybrid 모드 사용 시 Ollama LLM 서버가 반드시 실행 중이어야 합니다.
- GPU 미탑재 시 자동으로 CPU로 전환됩니다.
- torch GPU 버전 설치는 [공식 PyTorch 사이트](https://pytorch.org/get-started/locally/) 참고.
- 문의/기여 환영!

---

## License

MIT License

---
