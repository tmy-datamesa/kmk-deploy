# ⚖️ Multi-Law Legal RAG Agent — Deployment

**Komşuluk & Apartman Hukuku Yapay Zeka Asistanı**

> 🔗 Canlı Uygulama: [https://kmk-deploy-csnu3xbifrnaixgypfsa93.streamlit.app](https://kmk-deploy-csnu3xbifrnaixgypfsa93.streamlit.app)
> 🔗 Backend API: [https://legal-rag-api-232706383774.europe-west1.run.app/docs](https://legal-rag-api-232706383774.europe-west1.run.app/docs)

---

## 1. Proje Hakkında

Bu proje, **Kat Mülkiyeti Kanunu (KMK)** başta olmak üzere **6 farklı hukuk kaynağını** kullanarak apartman ve site yönetimiyle ilgili hukuki soruları yanıtlayan bir **Yapay Zeka Asistanı**dır.

Sistem, **Retrieval-Augmented Generation (RAG)** tekniğini ve **Agentic (Ajan) mimarisini** kullanarak:
1. Kullanıcının sorusunu analiz eder,
2. Hangi kanunun uzmanlık alanına girdiğine karar verir,
3. İlgili kanun maddelerini vektör veritabanından bulur,
4. Bulunan bilgilere dayanarak **doğru ve kaynaklı** bir cevap üretir.

---

## 2. Kullanılan Teknolojiler

### 🤖 Yapay Zeka & NLP
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **OpenAI GPT-4o-mini** | Metin üretimi (LLM) — Soruları anlama ve cevap oluşturma |
| **OpenAI text-embedding-3-small** | Metin vektörleştirme — Kanun maddelerini sayısal vektörlere çevirme |
| **OpenAI Function Calling (Tools)** | Ajan mimarisi — LLM'in hangi kanunu arayacağına karar vermesi |

### 💾 Veri & Veritabanı
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **ChromaDB (Cloud)** | Vektör veritabanı — Kanun maddelerinin embedding'lerini saklar ve benzerlik araması yapar |
| **PyPDF** | PDF dosyalarından metin çıkarma |
| **LangChain Text Splitters** | Metni anlamlı parçalara (chunks) bölme |

### 🌐 Backend (API)
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **FastAPI** | REST API sunucusu — RAG motorunu HTTP endpoint'i olarak sunar |
| **Uvicorn** | ASGI sunucusu — FastAPI'yi çalıştırır |
| **Pydantic** | Veri doğrulama — API request/response modellerini tanımlar |

### 🎨 Frontend
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **Streamlit** | Web arayüzü — Sohbet tabanlı kullanıcı deneyimi |
| **Streamlit Cloud** | Frontend hosting — Uygulamayı internete açar |

### ☁️ Bulut & DevOps
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **Docker** | Konteynerizasyon — Uygulamayı paketler ve taşınabilir hale getirir |
| **Google Cloud Build** | CI/CD — Docker image'ını bulutta oluşturur |
| **Google Cloud Run** | Sunucusuz (Serverless) hosting — Backend API'yi çalıştırır |
| **Google Artifact Registry** | Docker image deposu — Image'ları saklar ve versiyonlar |

### 📊 MLOps (Deney Takibi)
| Teknoloji | Ne İçin Kullanıldı? |
|-----------|-------------------|
| **MLflow** | Deney loglama — Her soruyu, modeli ve parametreleri kaydeder |
| **RAGAS** | Değerlendirme — RAG sisteminin doğruluğunu ölçer (Faithfulness, Answer Relevancy) |

---

## 3. Sistem Mimarisi

```
┌─────────────────┐         ┌──────────────────────────────────┐
│  Streamlit Cloud │  HTTP   │     Google Cloud Run             │
│  (Frontend)      │────────▶│     FastAPI Backend              │
│                  │  /ask   │                                  │
│  app.py          │◀────────│  app_api.py                      │
│                  │  JSON   │    │                              │
└─────────────────┘         │    ▼                              │
                            │  LegalRAG Agent (agent.py)        │
                            │    │                              │
                            │    ├── OpenAI API (GPT-4o-mini)   │
                            │    │   └── Function Calling       │
                            │    │                              │
                            │    └── RAG Engine (rag_engine.py) │
                            │        └── ChromaDB Cloud         │
                            │            (6 Kanun Koleksiyonu)  │
                            └──────────────────────────────────┘
```

### Veri Akışı (Bir Soru Sorulduğunda):

```
1. Kullanıcı soru yazar  →  Streamlit Cloud (app.py)
2. HTTP POST /ask        →  Cloud Run (app_api.py)
3. LegalRAG.generate_answer() çalışır:
   a. Soru GPT-4o-mini'ye gönderilir
   b. GPT, hangi kanunu arayacağına karar verir (Function Calling)
      Örn: "search_kmk" veya "search_tbk"
   c. İlgili kanunun ChromaDB koleksiyonunda vektör araması yapılır
   d. Bulunan maddeler GPT'ye geri gönderilir
   e. GPT, kaynaklara dayanarak nihai cevabı üretir
4. Cevap + Kaynaklar JSON olarak döner
5. Streamlit ekranda gösterir
```

---

## 4. Proje Dosya Yapısı

```
kmk-deploy/
│
├── app.py                  # Streamlit Frontend (API çağrısı yapar)
├── app_api.py              # FastAPI Backend (POST /ask, GET /health)
├── Dockerfile              # Docker container tarifi
├── .dockerignore           # Docker build'den hariç tutulan dosyalar
├── requirements.txt        # Python bağımlılıkları
├── Makefile                # Kısayol komutları (make setup, make run)
├── .env.example            # Ortam değişkenleri şablonu
│
├── src/                    # Ana Python paketi
│   ├── __init__.py
│   ├── config.py           # Merkezi konfigürasyon (modeller, parametreler)
│   ├── utils.py            # ChromaDB & Embedding bağlantıları
│   ├── ingestion.py        # ETL: PDF → Chunk → ChromaDB
│   ├── rag_engine.py       # Vektör arama motoru (Retriever)
│   ├── agent.py            # Ajan: Router + RAG + LLM (Beyin)
│   └── evaluation.py       # RAGAS + MLflow ile değerlendirme
│
└── data/                   # Hukuk kaynakları (PDF dosyaları)
    ├── kat-mulkiyeti.pdf
    ├── borclar-kanunu.pdf
    ├── anayasa.pdf
    ├── medeni_kanun.pdf
    ├── asansor_yonetmeligi.pdf
    └── yangin_yonetmeligi.pdf
```

---

## 5. Hukuk Kaynakları

| # | Kaynak | Koleksiyon | Kapsam |
|---|--------|-----------|--------|
| 1 | Kat Mülkiyeti Kanunu (KMK) | `law_kmk` | Aidat, site yönetimi, kat malikleri kurulu |
| 2 | Türk Borçlar Kanunu (TBK) | `law_tbk` | Kira sözleşmeleri, kiracı hakları |
| 3 | T.C. Anayasası | `law_anayasa` | Konut dokunulmazlığı, mülkiyet hakkı |
| 4 | Türk Medeni Kanunu (TMK) | `law_tmk` | Genel mülkiyet ve komşuluk hakları |
| 5 | Asansör Yönetmeliği | `reg_asansor` | Asansör bakım, kırmızı etiket |
| 6 | Yangın Yönetmeliği | `reg_yangin` | Yangın merdiveni, kaçış yolları |

---

## 6. RAG Pipeline Detayları

### 6.1 Veri Hazırlama (Ingestion — ETL)

```
PDF Dosyası → Metin Çıkarma → Parçalama (Chunking) → Vektörleştirme → ChromaDB'ye Kayıt
```

- **Chunk Size:** 2000 karakter
- **Chunk Overlap:** 400 karakter (bağlam kaybını önlemek için)
- **Ayırıcılar:** Hukuki yapıya uygun (KISIM, BÖLÜM, Madde, Ek Madde)

### 6.2 Ajan Mimarisi (Agentic RAG)

Klasik RAG'den farkı: Sistem **tek bir veritabanında** arama yapmak yerine, **önce hangi kanunu arayacağına karar verir**.

```
Kullanıcı Sorusu
      │
      ▼
  GPT-4o-mini (Function Calling)
      │
      ├── "Aidat ödemezsem?"     → search_kmk() → KMK koleksiyonu
      ├── "Kiracı depozitosu?"   → search_tbk() → TBK koleksiyonu
      ├── "Asansör arızası?"     → search_asansor() → Asansör koleksiyonu
      └── "Yangın merdiveni?"    → search_yangin() → Yangın koleksiyonu
```

### 6.3 RAG Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| LLM Model | `gpt-4o-mini` | Maliyet/performans dengesi |
| Embedding Model | `text-embedding-3-small` | Hızlı ve verimli vektörleştirme |
| Top-K | 6 | Her aramada döndürülen sonuç sayısı |
| Temperature | 0.0 | Deterministik cevaplar (yaratıcılık yok) |
| Chunk Size | 2000 | Metin parçalama boyutu (karakter) |

---

## 7. Deployment Mimarisi

### Neden Backend ve Frontend Ayrıldı?

| Monolitik (Eski) | Microservice (Yeni) |
|-------------------|-------------------|
| Streamlit → doğrudan LegalRAG çağırır | Streamlit → HTTP → FastAPI → LegalRAG |
| Tek sunucuda çalışır | Frontend ve Backend bağımsız ölçeklenir |
| Ölçeklenemez | Cloud Run otomatik ölçeklenir |

### Deployment Adımları

```
1. FastAPI Backend yazıldı (app_api.py)
       ↓
2. Dockerfile ile paketlendi
       ↓
3. Google Cloud Build ile image oluşturuldu (gcloud builds submit)
       ↓
4. Google Cloud Run'a deploy edildi (gcloud run deploy)
       ↓
5. Streamlit app.py güncellendi (requests.post ile API çağrısı)
       ↓
6. GitHub'a push edildi → Streamlit Cloud otomatik deploy etti
```

### Ortam Değişkenleri (Environment Variables)

| Değişken | Nerede? | Açıklama |
|----------|---------|----------|
| `OPENAI_API_KEY` | Cloud Run | GPT ve Embedding API erişimi |
| `CHROMA_HOST` | Cloud Run | ChromaDB sunucu adresi |
| `CHROMA_API_KEY` | Cloud Run | ChromaDB kimlik doğrulama |
| `CHROMA_TENANT` | Cloud Run | ChromaDB kiracı ID'si |
| `CHROMA_DATABASE` | Cloud Run | ChromaDB veritabanı adı |
| `BACKEND_URL` | Streamlit Cloud | Cloud Run API adresi |

---

## 8. API Endpoint'leri

| Method | Endpoint | Açıklama | Örnek |
|--------|----------|----------|-------|
| `GET` | `/health` | Health kontrolü | `{"status": "healthy", "rag_ready": true}` |
| `POST` | `/ask` | Soru-cevap | `{"question": "Aidat ödemezsem ne olur?"}` |
| `GET` | `/docs` | Swagger arayüzü | Otomatik API dokümantasyonu |

### Örnek API Çağrısı:
```bash
curl -X POST https://legal-rag-api-232706383774.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Aidat ödemezsem ne olur?"}'
```

### Örnek Cevap:
```json
{
  "answer": "Kat Mülkiyeti Kanunu Madde 20 uyarınca, aidat borcunuz nedeniyle icra takibi başlatılabilir...",
  "sources": [
    {"doc_name": "Kat Mülkiyeti Kanunu", "content": "Madde 20 – Kat malikleri..."}
  ]
}
```

---

## 9. Değerlendirme (MLOps)

Sistemin performansı **RAGAS** framework'ü ile ölçülmüş ve **MLflow** ile loglanmıştır.

| Metrik | Açıklama | Ne Ölçüyor? |
|--------|----------|-------------|
| **Faithfulness** | Cevap, kaynaklara sadık mı? | Halüsinasyon kontrolü |
| **Answer Relevancy** | Cevap soruyla alakalı mı? | Konu dışı cevap kontrolü |

Değerlendirme komutu:
```bash
make eval          # RAGAS testlerini çalıştır
mlflow ui          # Sonuçları görüntüle (http://127.0.0.1:5000)
```

---

## 10. Kurulum ve Çalıştırma

### Yerel Geliştirme
```bash
make setup         # Sanal ortam + bağımlılıkları kur
make ingest        # PDF'leri ChromaDB'ye yükle
make run           # Streamlit uygulamasını başlat
```

### Backend (Docker)
```bash
docker build -t legal-rag-api .
docker run -p 8080:8080 --env-file .env legal-rag-api
```

### Google Cloud'a Deploy
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/legal-rag-api
gcloud run deploy legal-rag-api --image gcr.io/PROJECT-ID/legal-rag-api --platform managed --region europe-west1
```

---

## 11. Kısıtlamalar

- Sistem **sadece** apartman, site ve komşuluk hukuku bağlamında çalışır.
- Ceza hukuku, ticaret hukuku gibi farklı alanlar kapsam dışıdır.
- Cevaplar hukuki tavsiye niteliği taşımaz, bilgilendirme amaçlıdır.
- LLM'in ürettiği cevaplar her zaman %100 doğru olmayabilir.

---

## 12. Gelecek Planları

- [ ] CI/CD pipeline (GitHub Actions → otomatik Cloud Run deploy)
- [ ] Kullanıcı geri bildirim sistemi (cevap kalitesi takibi)
- [ ] Daha fazla hukuk kaynağı eklenmesi
- [ ] Cevaplarda ilgili mahkeme kararlarına referans verilmesi
