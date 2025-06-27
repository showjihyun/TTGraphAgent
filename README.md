# DBGraphAgent

**DBGraphAgent** is an AI pipeline platform that converts natural language text into an entity-relationship-based graph structure.
It supports both Korean and English input, GPU acceleration, Hybrid (NLP+LLM) mode, LLM Only mode, web-based visualization, and LLM integration.
Currently, this is a prototype version.

---
## Demo Screenshot

![image](https://github.com/user-attachments/assets/6d23a7e2-66fc-47b1-a5ac-3e4e38acc1e0)

---

## Key Features

- **5-Stage NLP Pipeline**: Preprocessing → Entity Extraction (NER) → Relation Extraction (RE) → Attribute Extraction → Structuring/LLM Correction
- **Hybrid Mode**: Combines high-performance NLP with LLM correction (improves accuracy/consistency)
- **GPU Acceleration**: Automatic GPU usage for SentenceTransformer, HuggingFace NER, etc.
- **Ollama LLM Integration**: Utilizes local LLMs such as llama3 (LLM Only mode)
- **React + Vite Frontend**: Input/result visualization and various processing options
- **REST API**: FastAPI-based, provides endpoints such as `/api/text-to-graph`

---

## Folder Structure

```
DBGraphAgent/
  backend/         # FastAPI server, NLP/LLM pipeline, service logic
    llm/           # LLM handlers (llama3, chatgpt, etc.)
    routers/       # API routers
    services/      # NLP/LLM services
    test_gpu_web.html  # Webpage for GPU/Hybrid/LLM testing
    requirements.txt
    main.py
  frontend/        # React + TypeScript + Vite frontend
    src/
    public/
    package.json
    ...
  docs/            # Planning/design documents
```

---

## Installation & Usage

### 1. Python Backend (FastAPI)

```bash
cd DBGraphAgent/backend
pip install -r requirements.txt
# (CUDA GPU recommended, install torch with cu121 or other GPU version)
python main.py
```

### 2. Ollama LLM Server (Local LLM)

```bash
ollama serve
ollama pull llama3
```

### 3. Frontend (React + Vite)

```bash
cd DBGraphAgent/frontend
npm install
npm run dev
```

### 4. Test Webpage (Local HTML)

- Open the `DBGraphAgent/backend/test_gpu_web.html` file in your browser

---

## Main APIs

- `POST /api/text-to-graph`
  - Input: `{ "text": "Your text", "language": "korean", "processing_method": "hybrid|nlp_only|llm_only" }`
  - Output: Entity/relationship graph JSON

- `GET /api/health`
  - Check system status/availability

---

## Processing Modes

- **hybrid**: NLP + LLM correction (recommended, GPU accelerated)
- **nlp_only**: NLP pipeline only (fastest, GPU accelerated)
- **llm_only**: LLM only (requires Ollama/llama3)

---

## Example Output

- Input:
  `Professor Kim Chulsoo is researching artificial intelligence at Seoul National University.`
- Hybrid/NLP Only Output:
  - Entities: `Professor Kim Chulsoo (PERSON)`, `Seoul National University (ORG)`
  - Relation: `Professor Kim Chulsoo → Seoul National University (affiliation/related)`

---

## Notes/Others

- The Ollama LLM server must be running for LLM Only/Hybrid modes to work properly.
- If no GPU is available, it will automatically fall back to CPU.
- If you disagree, you are probably right.

---

## License

MIT License

---
