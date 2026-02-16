import os
import chromadb
from chromadb.utils import embedding_functions
from src import config


def get_chroma_client():
    """
    ChromaDB İstemcisini Getir
    --------------------------
    Bu fonksiyon, UZAK (Cloud) vektör veritabanı ile bağlantıyı kurar.
    Yerel veritabanı desteği kaldırılmıştır.
    
    Çıktı:
        - chromadb.HttpClient: Veritabanı istemcisi
    
    Not:
        .env dosyasında CHROMA_HOST tanımlı olmalıdır.
    """
    try:
        # Cloud ChromaDB Bağlantısı (Zorunlu)
        settings = chromadb.config.Settings()
        
        # Auth Provider YERİNE Header kullanıyoruz (api.trychroma.com için)
        headers = {}
        if config.CHROMA_API_KEY:
            headers["x-chroma-token"] = config.CHROMA_API_KEY
        
        client = chromadb.HttpClient(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            ssl=config.CHROMA_SSL,
            tenant=config.CHROMA_TENANT,
            database=config.CHROMA_DATABASE,
            headers=headers,
            settings=settings
        )
        print(f"☁️ Cloud ChromaDB Bağlandı (ZORUNLU): {config.CHROMA_HOST} | Tenant: {config.CHROMA_TENANT} | DB: {config.CHROMA_DATABASE}")
        return client
        
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        raise e


def get_embedding_function():
    """
    Embedding (Vektörleştirme) Fonksiyonunu Getir
    ---------------------------------------------
    Metinleri sayısal vektörlere çeviren yapay zeka modelini hazırlar.
    
    Model:
        - text-embedding-3-small (OpenAI)
        
    Kullanım:
        Bu fonksiyon hem veri yüklerken (ingestion) hem de soru sorarken (retrieval)
        aynı standartta olmalıdır.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("HATA: .env dosyasında OPENAI_API_KEY eksik!")

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL
    )
