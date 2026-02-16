import os
from dotenv import load_dotenv

# .env dosyasını yükle (API anahtarları için)
load_dotenv()

"""
KONFİGÜRASYON
-----------------------
Bu dosya projenin tüm ayarlarını tek bir merkezde toplar.
Mümkün olduğunca sabit değerleri (hardcoded strings) burada tutmaya çalışın.
"""

# ==============================================================================
# 1. API VE İSTEMCİ AYARLARI
# ==============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# 2. MODEL AYARLARI
# ==============================================================================
LLM_MODEL = "gpt-4o-mini" # Maliyet/Performans için optimize model
EMBEDDING_MODEL = "text-embedding-3-small" # Vektörleştirme modeli

# ==============================================================================
# 3. MLOPS (TAKİP VE LOGLAMA)
# ==============================================================================
# Yerel bir SQLite veritabanı kullanılır.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "legal-rag-v1"

# ==============================================================================
# 4. HUKUK KAYNAKLARI (VERİ)
# ==============================================================================
# Proje kök dizini ve veri klasörü
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Veritabanı Ayarları (Sadece Cloud Modu)
# Yerel klasör desteği kaldırıldı. Mutlaka bir ChromaDB sunucusu (Docker veya Cloud) gereklidir.
CHROMA_HOST = os.getenv("CHROMA_HOST") 
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY") 
CHROMA_SSL = os.getenv("CHROMA_SSL", "False").lower() == "true" 
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")

if not CHROMA_HOST:
    raise ValueError("HATA: .env dosyasında 'CHROMA_HOST' tanımlı değil! Cloud modu zorunludur.")

# Kullanılabilir Hukuk Kaynakları
# Yeni bir kanun eklemek için bu sözlüğe yeni bir kayıt ekleyin ve "make ingest" çalıştırın.
LEGAL_DOCS = {
    "kmk": {
        "name": "Kat Mülkiyeti Kanunu",
        "description": "Apartman, site yönetimi, aidat, komşuluk ilişkileri, gürültü, koku, yasak işler (randevu evi, klinik vb.), kapıcı ve kat malikleri kurulu hakkında sorular.",
        "path": os.path.join(DATA_DIR, "kat-mulkiyeti.pdf"),
        "collection": "law_kmk"
    },
    "tbk": {
        "name": "Türk Borçlar Kanunu (İlgili Maddeler)",
        "description": "Sadece kira sözleşmeleri, kiracı-ev sahibi ilişkileri ve komşuluktan doğan zararlar.",
        "path": os.path.join(DATA_DIR, "borclar-kanunu.pdf"),
        "collection": "law_tbk"
    },
    "anayasa": {
        "name": "T.C. Anayasası (İlgili Maddeler)",
        "description": "Konut dokunulmazlığı, mülkiyet hakkı ve özel hayatın gizliliği (Komşuluk bağlamında).",
        "path": os.path.join(DATA_DIR, "anayasa.pdf"),
        "collection": "law_anayasa"
    },
    "tmk": {
        "name": "Türk Medeni Kanunu (İlgili Maddeler)",
        "description": "Genel mülkiyet hakları, taşınmaz mülkiyeti ve komşuluk hakları. (Apartman veya site dışındaki, müstakil yapılar veya genel hükümler için kullan).",
        "path": os.path.join(DATA_DIR, "medeni_kanun.pdf"),
        "collection": "law_tmk"
    },
    "asansor": {
        "name": "Asansör İşletme ve Bakım Yönetmeliği",
        "description": "Asansörlerin bakımı, yıllık kontrolleri, kırmızı etiket, mühürleme ve yönetici sorumlulukları.",
        "path": os.path.join(DATA_DIR, "asansor_yonetmeligi.pdf"),
        "collection": "reg_asansor"
    },
    "yangin": {
        "name": "Binaların Yangından Korunması Yönetmeliği",
        "description": "Yangın merdiveni, yangın tüpleri, kaçış yolları ve binadaki teknik yangın güvenlik önlemleri.",
        "path": os.path.join(DATA_DIR, "yangin_yonetmeligi.pdf"),
        "collection": "reg_yangin"
    }
}

# ==============================================================================
# 5. RAG PARAMETRELERİ
# ==============================================================================
CHUNK_SIZE = 2000      # Metin parçalama boyutu (karakter)
CHUNK_OVERLAP = 400    # Parçalar arası örtüşme (bağlam kaybını önlemek için)
TOP_K = 6              # LLM'e gönderilecek en alakalı parça sayısı
TEMPERATURE = 0.0      # Yaratıcılık katsayısı (0.0 = En tutarlı/Deterministik, 1.0 = En yaratıcı)
