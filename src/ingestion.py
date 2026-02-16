import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from src import config, utils

def load_pdf(file_path):
    """
    PDF Dosyasını Okur ve Metne Çevirir.
    
    Girdi:
    - file_path: PDF dosyasının fiziksel yolu.
    
    Çıktı:
    - str: Dosyanın tamamının ham metin hali. Dosya yoksa None döner.
    """
    if not os.path.exists(file_path):
        print(f"UYARI: Dosya bulunamadı -> {file_path}")
        return None
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        # Her sayfanın metnini al ve birleştir
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text):
    """
    Metni Parçalar (Chunking Strategy).
    
    Amaç:
    Hukuki metinler (Kanunlar) belirli bir yapıya sahiptir (Madde, Kısım, Bölüm).
    Bu fonksiyon, metni bu mantıksal ayrımlara göre bölmeye çalışır.
    
    Ayırıcılar (Separators):
    Öncelik sırasına göre:
    1. "KISIM", "BÖLÜM" (Büyük başlıklar)
    2. "Madde", "Ek Madde" (En önemli kanun birimleri)
    3. "\n" (Paragraf sonları)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\nKISIM ", "\nBÖLÜM ", "\nMadde ", "\nEk Madde ", "\nGeçici Madde ",
            "\n", " ", ""
        ],
        is_separator_regex=True
    )
    return text_splitter.split_text(text)

def ingest_all_docs(force_recreate=False):
    """
    ETL SÜRECİ (Extract - Transform - Load)
    ---------------------------------------
    Tüm tanımlı hukuk kaynaklarını (config.LEGAL_DOCS) işler ve Vektör Veritabanına yükler.
    
    Adımlar:
    1. Extract: PDF dosyasını oku.
    2. Transform: Metni anlamlı parçalara (chunks) böl.
    3. Load: Parçaları vektöre çevir ve ChromaDB'ye kaydet.
    
    Parametre:
    - force_recreate (bool): True ise var olan veritabanını silip sıfırdan oluşturur.
    """
    client = utils.get_chroma_client()
    embedding_fn = utils.get_embedding_function()
    
    # Config dosyasındaki her bir kanun tanımı için döngü
    for key, doc_info in config.LEGAL_DOCS.items():
        col_name = doc_info["collection"]
        pdf_path = doc_info["path"]
        
        # 1. TEMİZLİK: Eğer 'force_recreate=True' ise eski veriyi sil
        if force_recreate:
            try:
                client.delete_collection(col_name)
                print(f"🗑️ Koleksiyon silindi: {col_name}")
            except:
                pass # Zaten yoksa hata verme, devam et

        # 2. HAZIRLIK: Koleksiyonu (Tabloyu) getir veya oluştur
        collection = client.get_or_create_collection(
            name=col_name,
            embedding_function=embedding_fn
        )

        # 3. KONTROL: Veri zaten var mı? Varsa tekrar yükleme yapma 
        if collection.count() == 0 or force_recreate:
            print(f"📖 Okunuyor: {doc_info['name']}...")
            raw_text = load_pdf(pdf_path)
            
            if raw_text:
                # Metni parçalara böl (Chunking)
                chunks = chunk_text(raw_text)
                
                # Her parçaya benzersiz ID ve Metadata ver
                ids = [f"{key}_{i}" for i in range(len(chunks))]
                
                # Metadata: Bu parça hangi kitaptan geldi? (Örn: source='kmk')
                metadatas = [{"source": key, "doc_name": doc_info["name"]} for _ in chunks]
                
                # Veritabanına kaydet
                collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                print(f"✅ Yüklendi: {doc_info['name']} ({len(chunks)} parça)")
            else:
                print(f"❌ Dosya okunamadı veya boş: {pdf_path}")
        else:
            print(f"⏭️ Zaten yüklü: {doc_info['name']}")

if __name__ == "__main__":
    # Tek başına çalıştırılırsa verileri tazeler
    ingest_all_docs(force_recreate=True)
