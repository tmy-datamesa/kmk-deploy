# FastAPI Backend API
# Bu dosya, RAG ajanını bir REST API servisi olarak sunar.
# Cloud Run üzerinde çalıştırılmak üzere tasarlanmıştır.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import LegalRAG
import logging

# --- Logging Ayarları ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Uygulaması ---
app = FastAPI(
    title="Legal RAG API",
    description="Apartman ve Site Hukuku Asistanı API",
    version="1.0.0"
)

# CORS: Streamlit Cloud ve diğer frontend'lerin erişimine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prod'da Streamlit Cloud URL'si ile sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Veri Modelleri (Request / Response) ---
class QuestionRequest(BaseModel):
    """Kullanıcı sorusu."""
    question: str

class SourceItem(BaseModel):
    """Tek bir kaynak belgesi."""
    doc_name: str
    content: str

class AnswerResponse(BaseModel):
    """API cevabı: yanıt metni + başvurulan kaynaklar."""
    answer: str
    sources: list[SourceItem]

# --- RAG Sistemi (Uygulama başlatılırken bir kez yüklenir) ---
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başlarken RAG sistemini hazırla."""
    global rag_system
    try:
        logger.info("RAG sistemi başlatılıyor...")
        rag_system = LegalRAG()
        logger.info("RAG sistemi hazır.")
    except Exception as e:
        logger.error(f"RAG sistemi başlatılamadı: {e}")
        raise e

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """
    Sağlık Kontrolü
    ----------------
    Cloud Run ve load balancer'ların kullanacağı basit bir endpoint.
    """
    return {"status": "healthy", "rag_ready": rag_system is not None}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Soru-Cevap Endpoint'i
    ---------------------
    Kullanıcının hukuki sorusunu alır, RAG ajanını çalıştırır
    ve cevabı kaynaklarıyla birlikte döndürür.
    
    Girdi: {"question": "Aidat ödemezsem ne olur?"}
    Çıktı: {"answer": "...", "sources": [...]}
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG sistemi henüz hazır değil.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")
    
    try:
        logger.info(f"Soru alındı: {request.question[:80]}...")
        answer, raw_sources = rag_system.generate_answer(request.question)
        
        # Ham kaynak verilerini API formatına dönüştür
        sources = []
        for src in raw_sources:
            sources.append(SourceItem(
                doc_name=src.get("metadata", {}).get("doc_name", "Bilinmiyor"),
                content=src.get("content", "")[:600]  # Uzun metinleri kırp
            ))
        
        logger.info(f"Cevap hazır. Kaynak sayısı: {len(sources)}")
        return AnswerResponse(answer=answer, sources=sources)
    
    except Exception as e:
        logger.error(f"Soru işlenirken hata: {e}")
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}")
