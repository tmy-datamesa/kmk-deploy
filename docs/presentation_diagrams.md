# 📊 Sunum Şemaları — Multi-Law Legal RAG Agent

---

## Sistem Mimarisi

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#555', 'primaryTextColor': '#111', 'edgeLabelBackground': '#fff'}}}%%
graph TD
    subgraph INPUT["📥 GİRİŞ"]
        QUESTION(["❓ Kullanıcı Sorusu<br/>(Prompt)"])
    end

    subgraph FE["🎨 FRONTEND — Streamlit Cloud"]
        APP["app.py<br/>Sohbet Arayüzü"]
    end

    subgraph BE["☁️ BACKEND — Google Cloud Run"]
        API["app_api.py<br/>FastAPI Sunucusu"]
        API --> AGENT["agent.py<br/>LegalRAG Agent"]
    end

    QUESTION -->|"1. Soru gönderilir"| APP
    
    APP -->|"2. HTTP POST /ask"| API
    API -.->|"5. JSON Response"| APP

    AGENT -->|"3. Function Calling"| GPT["🤖 OpenAI<br/>GPT-4o"]
    AGENT -->|"4. Vektör Araması"| CHROMA["💾 ChromaDB<br/>Cloud"]

    APP -.->|"6. Cevap gösterilir"| QUESTION

    style INPUT fill:#1a1a2e,stroke:#888,color:#fff
    style QUESTION fill:#1a1a2e,stroke:#888,color:#fff
    style FE fill:#0f3460,stroke:#888,color:#fff
    style BE fill:#533483,stroke:#888,color:#fff
    style APP fill:#2c2c54,stroke:#888,color:#fff
    style API fill:#2c2c54,stroke:#888,color:#fff
    style AGENT fill:#474787,stroke:#888,color:#fff
    style GPT fill:#1a1a2e,stroke:#888,color:#fff
    style CHROMA fill:#1a1a2e,stroke:#888,color:#fff
```

---

## Ajan Yönlendirme Mantığı (Router)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#888'}}}%%
flowchart TD
    Q["❓ Kullanıcı Sorusu"] --> LLM["🤖 GPT-4o — Function Calling"]

    LLM -->|"search_kmk()"| KMK["📕 Kat Mülkiyeti Kanunu<br/>Aidat, yönetim, gürültü"]
    LLM -->|"search_tbk()"| TBK["📗 Türk Borçlar Kanunu<br/>Kira, depozito, tahliye"]
    LLM -->|"search_anayasa()"| ANA["📘 Anayasa<br/>Konut dokunulmazlığı"]
    LLM -->|"search_tmk()"| TMK["📙 Türk Medeni Kanunu<br/>Mülkiyet, komşuluk"]
    LLM -->|"search_asansor()"| ASN["📓 Asansör Yönetmeliği<br/>Bakım, kırmızı etiket"]
    LLM -->|"search_yangin()"| YNG["📔 Yangın Yönetmeliği<br/>Söndürücü, kaçış yolu"]

    KMK --> DB["💾 ChromaDB Cloud"]
    TBK --> DB
    ANA --> DB
    TMK --> DB
    ASN --> DB
    YNG --> DB

    DB --> RES["📄 İlgili Maddeler — Top-K=6"]

    style Q fill:#1a1a2e,stroke:#888,color:#fff
    style LLM fill:#533483,stroke:#888,color:#fff
    style KMK fill:#2c2c54,stroke:#888,color:#fff
    style TBK fill:#2c2c54,stroke:#888,color:#fff
    style ANA fill:#2c2c54,stroke:#888,color:#fff
    style TMK fill:#2c2c54,stroke:#888,color:#fff
    style ASN fill:#2c2c54,stroke:#888,color:#fff
    style YNG fill:#2c2c54,stroke:#888,color:#fff
    style DB fill:#0f3460,stroke:#888,color:#fff
    style RES fill:#16213e,stroke:#888,color:#fff
```

---

## Veri Yükleme Pipeline'ı (ETL / Ingestion)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#888'}}}%%
flowchart LR
    subgraph Extract["1. EXTRACT"]
        PDF["📄 6 PDF Dosyası"] --> Read["PyPDF<br/>Metin Çıkarma"]
    end

    subgraph Transform["2. TRANSFORM"]
        Read --> Chunk["LangChain<br/>Chunking<br/>2000 kar / 400 overlap"]
        Chunk --> Embed["OpenAI<br/>Embedding<br/>text-embedding-3-small"]
    end

    subgraph Load["3. LOAD"]
        Embed --> Store["ChromaDB Cloud<br/>6 Koleksiyon"]
    end

    style Extract fill:#1a1a2e,stroke:#888,color:#fff
    style Transform fill:#1a1a2e,stroke:#888,color:#fff
    style Load fill:#1a1a2e,stroke:#888,color:#fff
    style PDF fill:#2c2c54,stroke:#888,color:#fff
    style Read fill:#2c2c54,stroke:#888,color:#fff
    style Chunk fill:#533483,stroke:#888,color:#fff
    style Embed fill:#533483,stroke:#888,color:#fff
    style Store fill:#0f3460,stroke:#888,color:#fff
```
---

## 7. MLOps / Değerlendirme Pipeline'ı

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#888'}}}%%
flowchart TD
    subgraph Eval["📊 Değerlendirme"]
        DATA["eval_data.json<br/>15 Soru-Cevap"] --> RAG["LegalRAG<br/>Cevap Üret"]
        RAG --> RAGAS["RAGAS Framework<br/>3 Metrik Hesapla"]
    end

    subgraph Metrics["📈 Metrikler"]
        RAGAS --> F["Faithfulness: 0.59<br/>Kaynaklara sadakat"]
        RAGAS --> R["Answer Relevancy: 0.51<br/>Soruyla alakalılık"]
        RAGAS --> C["Answer Correctness: 0.57<br/>Doğruluk"]
    end

    subgraph Track["🔍 MLflow Tracking"]
        F --> ML["MLflow Dashboard"]
        R --> ML
        C --> ML
        ML --> Params["Parametreler<br/>model, top_k, temp"]
        ML --> Arts["Artifacts<br/>CSV sonuçlar"]
    end

    style Eval fill:#1a1a2e,stroke:#888,color:#fff
    style Metrics fill:#1a1a2e,stroke:#888,color:#fff
    style Track fill:#1a1a2e,stroke:#888,color:#fff
    style DATA fill:#2c2c54,stroke:#888,color:#fff
    style RAG fill:#2c2c54,stroke:#888,color:#fff
    style RAGAS fill:#533483,stroke:#888,color:#fff
    style F fill:#474787,stroke:#888,color:#fff
    style R fill:#474787,stroke:#888,color:#fff
    style C fill:#474787,stroke:#888,color:#fff
    style ML fill:#0f3460,stroke:#888,color:#fff
    style Params fill:#16213e,stroke:#888,color:#fff
    style Arts fill:#16213e,stroke:#888,color:#fff
```

---

## Tech Stack

```mermaid
mindmap
  root["⚖️ Legal RAG Agent"]
    AI
      GPT-4o
      text-embedding-3-small
      Function Calling
    Veri
      ChromaDB Cloud
      PyPDF
      LangChain Splitters
    Backend
      FastAPI
      Uvicorn
      Pydantic
    Frontend
      Streamlit
      Streamlit Cloud
    DevOps
      Docker
      Cloud Build
      Cloud Run
      Artifact Registry
    MLOps
      MLflow
      RAGAS
```
---

## Agentic RAG Akışı (Core Pipeline)

```mermaid
sequenceDiagram
    participant U as 👤 Kullanıcı
    participant S as 🎨 Streamlit
    participant F as ⚡ FastAPI
    participant A as 🧠 Agent
    participant G as 🤖 GPT-4o
    participant C as 💾 ChromaDB

    U->>S: "Aidat ödemezsem ne olur?"
    S->>F: POST /ask
    F->>A: generate_answer()
    
    Note over A,G: 1. PLANLAMA (Router)
    A->>G: Soru + Tool tanımları
    G-->>A: Tool Call: search_kmk("aidat borcu")
    
    Note over A,C: 2. ARAŞTIRMA (Retriever)
    A->>C: Vektör araması (Top-K=6)
    C-->>A: İlgili Madde chunk'ları
    
    Note over A,G: 3. CEVAPLAMA (Generator)
    A->>G: Soru + Bulunan maddeler
    G-->>A: "Kat Mülkiyeti Kanunu uyarınca..."
    
    Note over A: 4. KAYNAK EKLEME (Regex)
    A->>A: Chunk'lardan Madde XX çıkar
    A-->>F: Cevap + "📌 Kaynak: KMK (Madde 20)"
    F-->>S: JSON Response
    S-->>U: Cevap gösterilir
```