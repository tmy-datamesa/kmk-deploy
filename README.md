# 🏛️ Multi-Law Legal Agent (Hukuk Asistanı)

Bu proje, tek bir kanun yerine **birden fazla hukuk kaynağını (KMK, TBK, Anayasa vb.)** yönetebilen, "Ajan (Agent)" tabanlı bir yapay zeka sistemidir.

## 🌟 Nedir Farkı? (V1 vs V2)

*   **V1 (Eski):** Sadece Kat Mülkiyeti Kanunu'nu bilen, "tek fonksiyonlu" bir araçtı.
*   **V2 (Yeni):** Akıllı bir **Yönlendirici (Router)** içerir. Sorunuza bakar, hangi kanunun uzmanlık alanına girdiğine karar verir. **Ancak dikkat:** Sistem sadece **Apartman, Site ve Komşuluk Hukuku** bağlamında çalışır.

Örneğin:
*   *"Aidat ödemezsem ne olur?"* -> **KMK (Kat Mülkiyeti)**
*   *"Kiracı depozitosu iade edilmedi"* -> **TBK (Borçlar Kanunu)** *(Sadece konut kiraları bağlamında)*

## 📂 Mimari (Agentic RAG)

Sistem **"OpenAI Tools"** teknolojisini kullanarak çalışır:
1.  **Agent**: Kullanıcı sorusunu analiz eder.
2.  **Tools**:
    *   `search_kmk()`: Kat Mülkiyeti Kanunu (Ana Kaynak).
    *   `search_tbk()`: Türk Borçlar Kanunu (Kira/Komşuluk).
    *   `search_tmk()`: Türk Medeni Kanunu (Mülkiyet/Komşuluk Hakları).
    *   `search_asansor()`: Asansör Bakım Yönetmeliği.
    *   `search_yangin()`: Yangın Koruma Yönetmeliği.
    *   `search_anayasa()`: Anayasa (Haklar).
3.  **RAG Engine**: Seçilen alet çalışır, veritabanından bilgi çeker ve ajana verir.


## 🛠️ Kurulum

```bash
# 1. Klasöre girin
cd kmk-multi-law-rag

# 2. Kurulumu yapın
make setup

# 3. .env dosyasını ayarlayın
# (OPENAI_API_KEY ve CHROMA_HOST zorunludur. Yerel veritabanı desteklenmemektedir.)

# 4. Verileri yükleyin (PDF'ler taranır)
make ingest

# 5. Başlatın
make run
```

## 🧹 Temizlik ve Bakım

Projede biriken logları veya önbellek dosyalarını temizlemek için:

```bash
# Önbellek dosyalarını (__pycache__) temizler
make clean

# MLflow loglarını ve veritabanını sıfırlar (DİKKAT: Eski test sonuçları silinir)
make clean-logs
```

## 📊 Değerlendirme (Evaluation)
Projenin performansını **RAGAS** ve **MLflow** ile ölçmek için:

1. Değerlendirme scriptini çalıştırın:
```bash
make eval
```
2. Sonuçları MLflow arayüzünde görüntüleyin:
```bash
mlflow ui
```
Tarayıcınızda `http://127.0.0.1:5000` adresine gidin. Burada:
- Her bir denemeyi (Run) görebilir,
- "Faithfulness" ve "Answer Relevancy" skorlarını karşılaştırabilir,
- Hangi modelin veya parametrenin (Chunk Size, Top K) daha iyi sonuç verdiğini analiz edebilirsiniz.



