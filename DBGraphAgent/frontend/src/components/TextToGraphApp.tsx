import React, { useRef, useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import * as pdfjsLib from 'pdfjs-dist';
// @ts-ignore
(pdfjsLib as any).GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${(pdfjsLib as any).version}/pdf.worker.min.js`;

// JSON 파싱에 문제될 수 있는 문자열을 정제하는 함수
function sanitizeForJson(text: string): string {
  return text
    .replace(/[`]/g, "'") // 백틱 → 작은따옴표
    .replace(/\"{2,}/g, "'") // 연속 큰따옴표 → 작은따옴표
    .replace(/[\x00-\x1F\x7F]/g, ' ') // 제어문자 → 공백
    .replace(/[\\]/g, ''); // 역슬래시 → 제거
}

interface GraphNode {
  id: string;
  label: string;
  node_type: string;
  x?: number;
  y?: number;
  confidence?: number;
  nlp_attributes?: Record<string, any>;
  nlp_confidence?: number;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  weight?: number;
  confidence?: number;
}

interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: {
    node_count: number;
    edge_count: number;
    node_types: string[];
    processing_language: string;
    confidence_avg?: number;
  };
  processing_method?: string;
  nlp_stats?: Record<string, any>;
}

interface ProcessingInfo {
  nlp_available: boolean;
  llm_available: boolean;
  supported_languages: string[];
  processing_stages: string[];
  supported_methods: string[];
}

// 그래프 시각화 컴포넌트
interface GraphVisualizationProps {
  graph: GraphResponse;
  onDownload: () => void;
}

function GraphVisualization({ graph, onDownload }: GraphVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const HEIGHT = 800;
  const WIDTH = 800;

  // 노드 id로 좌표 lookup
  const nodePos = (id: string) => {
    const node = graph?.nodes.find((n) => n.id === id);
    return node ? { x: node.x ?? WIDTH / 2, y: node.y ?? HEIGHT / 2 } : { x: WIDTH / 2, y: HEIGHT / 2 };
  };

  // 그래프 중앙 정렬
  let offsetX = 0, offsetY = 0;
  if (graph && graph.nodes.length > 0) {
    const xs = graph.nodes.map(n => n.x ?? WIDTH / 2);
    const ys = graph.nodes.map(n => n.y ?? HEIGHT / 2);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    offsetX = WIDTH / 2 - centerX;
    offsetY = HEIGHT / 2 - centerY;
  }

  // 줌 조작 함수들
  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(prev * 1.2, 3));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  }, []);

  const handleZoomReset = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // 마우스 휠 줌
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.min(Math.max(prev * delta, 0.1), 3));
  }, []);

  // 마우스 드래그 팬
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [pan]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  React.useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const handleDownloadGraph = () => {
    if (!svgRef.current) return;
    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svgRef.current);
    const svgBlob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'knowledge-graph.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* 그래프 헤더 */}
      <div className="flex items-center justify-between mb-8 p-6 bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20">
        <div className="flex items-center gap-8 text-white">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-blue-400 rounded-full"></div>
            <span className="font-semibold text-lg">노드: {graph.summary.node_count}</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-green-400 rounded-full"></div>
            <span className="font-semibold text-lg">관계: {graph.summary.edge_count}</span>
          </div>
          <div className="flex items-center gap-2 text-white/70">
            <span className="text-sm">줌: {(zoom * 100).toFixed(0)}%</span>
          </div>
          {graph.processing_method && (
            <div className="flex items-center gap-2 text-purple-200">
              <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
              <span className="text-sm font-medium">
                {graph.processing_method === 'nlp_llm_hybrid' ? '🧠 하이브리드' :
                 graph.processing_method === 'nlp_only' ? '⚡ NLP' : '🤖 LLM'}
              </span>
            </div>
          )}
        </div>
        
        {/* 줌 컨트롤 */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-xl p-2 border border-white/20">
            <button
              onClick={handleZoomOut}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              title="축소"
            >
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
              </svg>
            </button>
            <button
              onClick={handleZoomReset}
              className="px-3 py-1 hover:bg-white/20 rounded-lg transition-colors text-white text-sm font-medium"
              title="원래 크기"
            >
              100%
            </button>
            <button
              onClick={handleZoomIn}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              title="확대"
            >
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
          
          <button
            onClick={handleDownloadGraph}
            className="flex items-center gap-3 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200 transform hover:scale-105 text-lg"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            다운로드
          </button>
        </div>
      </div>
      
      {/* 그래프 컨테이너 */}
      <div 
        ref={containerRef}
        className="flex-1 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/20 p-8 overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        style={{ userSelect: 'none' }}
      >
        <div className="w-full h-full flex items-center justify-center">
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="w-full h-full"
            style={{ minHeight: HEIGHT }}
          >
            <defs>
              {/* 노드 그라데이션 */}
              <radialGradient id="nodeGradient" cx="50%" cy="30%" r="70%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
                <stop offset="50%" stopColor="#a855f7" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9" />
              </radialGradient>
              
              {/* 화살표 */}
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#ec4899" />
              </marker>
              
              {/* 그림자 효과 */}
              <filter id="nodeShadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="0" dy="4" stdDeviation="8" floodColor="#7c3aed" floodOpacity="0.3" />
              </filter>
              
              <filter id="edgeShadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="0" dy="2" stdDeviation="4" floodColor="#ec4899" floodOpacity="0.2" />
              </filter>
              
              {/* 엣지 그라데이션 */}
              <linearGradient id="edgeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ec4899" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#a855f7" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.6" />
              </linearGradient>
            </defs>
            
            <g transform={`translate(${offsetX + pan.x}, ${offsetY + pan.y}) scale(${zoom})`}>
              {/* 엣지 렌더링 */}
              {graph.edges.map((edge) => {
                const src = nodePos(edge.source);
                const tgt = nodePos(edge.target);
                const mx = (src.x + tgt.x) / 2;
                const my = (src.y + tgt.y) / 2;
                const offset = 40;
                
                return (
                  <g key={edge.id}>
                    {/* 곡선 연결선 */}
                    <path
                      d={`M${src.x},${src.y} Q${mx},${my - offset} ${tgt.x},${tgt.y}`}
                      stroke="url(#edgeGradient)"
                      strokeWidth={Math.max(2, (edge.weight ?? 1) * 2)}
                      fill="none"
                      markerEnd="url(#arrowhead)"
                      filter="url(#edgeShadow)"
                      opacity="0.8"
                    />
                    
                    {/* 관계 라벨 */}
                    <g>
                      <rect
                        x={mx - edge.relation.length * 4}
                        y={my - offset - 12}
                        width={edge.relation.length * 8}
                        height={20}
                        rx="10"
                        fill="rgba(255, 255, 255, 0.9)"
                        stroke="rgba(236, 72, 153, 0.3)"
                        strokeWidth="1"
                      />
                      <text
                        x={mx}
                        y={my - offset - 2}
                        textAnchor="middle"
                        fontSize="12"
                        fill="#7c3aed"
                        fontWeight="600"
                      >
                        {edge.relation}
                      </text>
                    </g>
                  </g>
                );
              })}
              
              {/* 노드 렌더링 */}
              {graph.nodes.map((node) => {
                const x = node.x ?? WIDTH / 2;
                const y = node.y ?? HEIGHT / 2;
                
                return (
                  <g key={node.id} className="node-group">
                    {/* 노드 원 */}
                    <circle
                      cx={x}
                      cy={y}
                      r="40"
                      fill="url(#nodeGradient)"
                      stroke="#ffffff"
                      strokeWidth="3"
                      filter="url(#nodeShadow)"
                      className="hover:r-45 transition-all duration-200"
                    />
                    
                    {/* 노드 라벨 */}
                    <text
                      x={x}
                      y={y - 3}
                      textAnchor="middle"
                      fontSize="16"
                      fill="#1e293b"
                      fontWeight="700"
                    >
                      {node.label.length > 10 ? node.label.substring(0, 10) + '...' : node.label}
                    </text>
                    
                    {/* 노드 타입 */}
                    <text
                      x={x}
                      y={y + 15}
                      textAnchor="middle"
                      fontSize="12"
                      fill="#64748b"
                      fontWeight="500"
                    >
                      {node.node_type}
                    </text>
                    
                    {/* 신뢰도 표시 (선택적) */}
                    {node.confidence && (
                      <text
                        x={x}
                        y={y + 28}
                        textAnchor="middle"
                        fontSize="10"
                        fill="#10b981"
                        fontWeight="600"
                      >
                        {(node.confidence * 100).toFixed(0)}%
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        </div>
        
        {/* 줌/팬 안내 */}
        <div className="absolute bottom-4 left-4 text-white/60 text-sm bg-black/20 backdrop-blur-sm rounded-lg p-2">
          <div>마우스 휠: 줌</div>
          <div>드래그: 이동</div>
        </div>
      </div>
    </div>
  );
}

const MIN_LEFT = 400;
const MAX_LEFT = 800;
const DEFAULT_LEFT = 500;

function TextToGraphApp() {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState<'korean' | 'english'>('korean');
  const [processingMethod, setProcessingMethod] = useState<'hybrid' | 'nlp_only' | 'llm_only'>('hybrid');
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [leftWidth, setLeftWidth] = useState(DEFAULT_LEFT);
  const [dragging, setDragging] = useState(false);
  const [processingInfo, setProcessingInfo] = useState<ProcessingInfo | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 시스템 정보 로드
  useEffect(() => {
    const loadProcessingInfo = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/processing-info');
        setProcessingInfo(response.data);
      } catch (error) {
        console.error('Failed to load processing info:', error);
      }
    };
    loadProcessingInfo();
  }, []);

  // PDF/txt 파일 업로드 핸들러
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    if (file.type === 'application/pdf') {
      const reader = new FileReader();
      reader.onload = async function () {
        const typedarray = new Uint8Array(reader.result as ArrayBuffer);
        // @ts-ignore
        const pdf = await pdfjsLib.getDocument({ data: typedarray }).promise;
        let fullText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const pageText = content.items.map((item: any) => item.str).join(' ');
          fullText += pageText + '\n';
        }
        setText(fullText);
      };
      reader.readAsArrayBuffer(file);
    } else if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
      const reader = new FileReader();
      reader.onload = function () {
        setText(reader.result as string);
      };
      reader.readAsText(file, 'utf-8');
    } else {
      setError('PDF 또는 TXT 파일만 지원합니다.');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setGraph(null);
    try {
      const sanitized = sanitizeForJson(text);
      const res = await axios.post('http://localhost:8000/api/text-to-graph', {
        text: sanitized,
        language,
        processing_method: processingMethod,
      });
      setGraph(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || '변환 중 오류 발생');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    // 이 함수는 GraphVisualization 컴포넌트에서 처리됨
  };

  const isButtonEnabled = !loading && (!!text.trim() || !!fileName);

  // Split Line 핸들러
  const handleMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    document.body.style.cursor = 'col-resize';
  };
  const handleMouseUp = () => {
    setDragging(false);
    document.body.style.cursor = '';
  };
  const handleMouseMove = (e: MouseEvent) => {
    if (!dragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    let px = e.clientX - rect.left;
    px = Math.max(MIN_LEFT, Math.min(MAX_LEFT, px));
    setLeftWidth(px);
  };
  React.useEffect(() => {
    if (dragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragging]);

  const getProcessingMethodInfo = (method: string) => {
    switch (method) {
      case 'hybrid':
        return {
          icon: '🧠',
          name: '하이브리드',
          desc: 'NLP + LLM 보정 (권장)',
          color: 'from-purple-500 to-pink-500'
        };
      case 'nlp_only':
        return {
          icon: '⚡',
          name: 'NLP 전용',
          desc: '빠른 처리',
          color: 'from-blue-500 to-cyan-500'
        };
      case 'llm_only':
        return {
          icon: '🤖',
          name: 'LLM 전용',
          desc: '기존 방식',
          color: 'from-green-500 to-emerald-500'
        };
      default:
        return {
          icon: '🧠',
          name: '하이브리드',
          desc: 'NLP + LLM 보정',
          color: 'from-purple-500 to-pink-500'
        };
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* 배경 애니메이션 */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900/20 via-slate-900/40 to-slate-900"></div>
      <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
      <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
      <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      
      <div ref={containerRef} className="relative z-10 flex w-full px-8 py-6 min-h-screen">
        {/* 좌측: 입력/설정 패널 */}
        <div 
          className="flex flex-col backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl border border-white/20 p-10 mr-6 transition-all duration-300 hover:bg-white/15"
          style={{ width: leftWidth, minWidth: MIN_LEFT, maxWidth: MAX_LEFT }}
        >
          <div className="flex items-center mb-10">
            <div className="w-4 h-4 bg-gradient-to-r from-pink-500 to-violet-500 rounded-full mr-4 animate-pulse"></div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
              Knowledge Graph AI
            </h1>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-8 flex-1 flex flex-col">
            <div className="relative">
              <textarea
                className="w-full border-0 rounded-2xl p-6 h-64 bg-white/10 backdrop-blur-sm text-white placeholder:text-white/60 focus:ring-2 focus:ring-purple-400/50 focus:outline-none resize-none shadow-inner transition-all duration-300 hover:bg-white/15 text-lg leading-relaxed"
                placeholder="텍스트를 입력하거나 파일을 업로드하세요..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="absolute top-4 right-4 text-white/40 text-sm font-medium">
                {text.length} chars
              </div>
            </div>
            
            {/* 언어 선택 */}
            <div className="flex flex-wrap items-center gap-6">
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-white/90 cursor-pointer group">
                  <div className="relative">
                    <input
                      type="radio"
                      name="language"
                      value="korean"
                      checked={language === 'korean'}
                      onChange={() => setLanguage('korean')}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded-full border-2 transition-all duration-200 ${language === 'korean' ? 'border-purple-400 bg-purple-400' : 'border-white/40 group-hover:border-white/60'}`}>
                      {language === 'korean' && <div className="w-2 h-2 bg-white rounded-full absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>}
                    </div>
                  </div>
                  한국어
                </label>
                <label className="flex items-center gap-2 text-white/90 cursor-pointer group">
                  <div className="relative">
                    <input
                      type="radio"
                      name="language"
                      value="english"
                      checked={language === 'english'}
                      onChange={() => setLanguage('english')}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded-full border-2 transition-all duration-200 ${language === 'english' ? 'border-purple-400 bg-purple-400' : 'border-white/40 group-hover:border-white/60'}`}>
                      {language === 'english' && <div className="w-2 h-2 bg-white rounded-full absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>}
                    </div>
                  </div>
                  English
                </label>
              </div>
              
              <div className="relative">
                <input
                  type="file"
                  accept=".pdf,.txt,text/plain,application/pdf"
                  onChange={handleFileUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="inline-flex items-center px-6 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-white/90 cursor-pointer transition-all duration-200 backdrop-blur-sm border border-white/20 font-medium">
                  <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  파일 업로드
                </label>
              </div>
            </div>

            {/* 처리 방법 선택 */}
            <div className="space-y-4">
              <h3 className="text-white/90 font-semibold text-lg">처리 방법 선택</h3>
              <div className="grid grid-cols-1 gap-3">
                {(['hybrid', 'nlp_only', 'llm_only'] as const).map((method) => {
                  const info = getProcessingMethodInfo(method);
                  const isAvailable = method === 'llm_only' || (method === 'hybrid' && processingInfo?.nlp_available) || (method === 'nlp_only' && processingInfo?.nlp_available);
                  
                  return (
                    <label
                      key={method}
                      className={`flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 cursor-pointer ${
                        processingMethod === method
                          ? 'bg-white/20 border-purple-400/50'
                          : 'bg-white/5 border-white/20 hover:bg-white/10'
                      } ${!isAvailable ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="relative">
                        <input
                          type="radio"
                          name="processing_method"
                          value={method}
                          checked={processingMethod === method}
                          onChange={() => setProcessingMethod(method)}
                          disabled={!isAvailable}
                          className="sr-only"
                        />
                        <div className={`w-5 h-5 rounded-full border-2 transition-all duration-200 ${
                          processingMethod === method ? 'border-purple-400 bg-purple-400' : 'border-white/40'
                        }`}>
                          {processingMethod === method && <div className="w-2 h-2 bg-white rounded-full absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>}
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xl">{info.icon}</span>
                          <span className="text-white font-medium">{info.name}</span>
                          {!isAvailable && <span className="text-red-400 text-sm">(비활성)</span>}
                        </div>
                        <div className="text-white/60 text-sm">{info.desc}</div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
            
            {fileName && (
              <div className="flex items-center gap-2 text-purple-200 text-sm bg-purple-500/20 rounded-lg p-3 backdrop-blur-sm">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                {fileName}
              </div>
            )}
            
            <button
              type="submit"
              disabled={!isButtonEnabled}
              className={`relative mt-auto bg-gradient-to-r ${getProcessingMethodInfo(processingMethod).color} hover:opacity-90 text-white font-bold py-5 px-10 rounded-2xl shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95 text-lg`}
            >
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                </div>
              )}
              <span className={loading ? 'opacity-0' : 'opacity-100'}>
                {loading ? '분석 중...' : `${getProcessingMethodInfo(processingMethod).icon} 그래프 생성`}
              </span>
            </button>
          </form>
          
          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-200 backdrop-blur-sm">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            </div>
          )}

          {/* 5단계 파이프라인 정보 */}
          {processingInfo && (
            <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl text-blue-200 backdrop-blur-sm">
              <h4 className="font-semibold mb-2">🔬 5단계 NLP 파이프라인</h4>
              <div className="text-sm space-y-1">
                {processingInfo.processing_stages.map((stage, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span>{stage}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* 분할선 */}
        <div
          className="w-1 bg-gradient-to-b from-purple-400/50 to-pink-400/50 hover:from-purple-400/80 hover:to-pink-400/80 cursor-col-resize absolute top-0 bottom-0 z-20 rounded-full transition-all duration-200"
          style={{ left: `${leftWidth}px`, transform: 'translateX(-50%)' }}
          onMouseDown={handleMouseDown}
        />
        
        {/* 우측: 그래프 시각화 Region */}
        <div className="flex-1 flex flex-col ml-6 min-w-0">
          {graph ? (
            <GraphVisualization graph={graph} onDownload={handleDownload} />
          ) : (
            <div className="flex-1 flex items-center justify-center bg-white/5 backdrop-blur-xl rounded-3xl border border-white/20">
              <div className="text-center">
                <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full flex items-center justify-center">
                  <svg className="w-12 h-12 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-white/80 mb-2">지식 그래프 대기 중</h3>
                <p className="text-white/60">텍스트를 입력하고 분석을 시작하세요</p>
                {processingInfo && (
                  <div className="mt-4 text-sm text-white/50">
                    NLP 프로세서: {processingInfo.nlp_available ? '✅ 사용 가능' : '❌ 비활성'}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      <style>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  );
}

export default TextToGraphApp; 