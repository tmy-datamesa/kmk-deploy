# 📈 Başarı Metrikleri — Multi-Law Legal RAG Agent

---

## 1. Genel Performans Özeti

| Metrik | Skor | Açıklama |
|--------|------|----------|
| **Faithfulness** | **0.59** | Cevap, kaynaklara ne kadar sadık? (Halüsinasyon kontrolü) |
| **Answer Relevancy** | **0.51** | Cevap, sorulan soruyla ne kadar alakalı? |
| **Answer Correctness** | **0.57** | Cevap, beklenen doğru cevaba ne kadar yakın? |

> **Değerlendirme modeli:** GPT-4o (RAGAS judge)
> **Test seti:** 15 soru-cevap çifti, 6 farklı hukuk kaynağından

---

## 2. Soru Bazlı Detaylı Sonuçlar

| # | Soru | Kaynak | Faithfulness | Relevancy | Correctness |
|---|------|--------|:---:|:---:|:---:|
| 1 | Çatı akıyor, tamir masrafına dükkan katılmak zorunda mı? | KMK | 0.25 | 0.49 | 0.71 |
| 2 | Yönetim planını değiştirmek için kaç oy gerekir? | KMK | 0.67 | 0.63 | 0.42 |
| 3 | Kat malikleri toplantısı ne zaman yapılır? | KMK | **1.00** | 0.63 | 0.36 |
| 4 | Komşu evini randevu evi olarak kullanıyor, ne yapabiliriz? | KMK | 0.50 | 0.56 | 0.52 |
| 5 | Kiracıyım, kirayı en geç ne zaman ödeyeyim? | TBK | **1.00** | 0.65 | **0.82** |
| 6 | Evde sonradan ayıp çıkarsa kiracı hakkım ne? | TBK | **1.00** | 0.63 | 0.77 |
| 7 | Kiracı komşulara saygısızlık yaparsa ev sahibi? | TBK | **1.00** | 0.37 | 0.55 |
| 8 | Asansör bakımı ne sıklıkla yaptırılmalı? | Asansör | 0.80 | 0.00 | 0.58 |
| 9 | Kırmızı etiketli asansör kullanılabilir mi? | Asansör | 0.00 | 0.61 | **0.82** |
| 10 | Binada kaç yangın söndürücü olmalı? | Yangın | **1.00** | 0.60 | 0.74 |
| 11 | Plan değişikliğinde salt çoğunluk yeterli mi? | KMK | 0.60 | 0.53 | 0.62 |
| 12 | Su sızıntısı tamiri — konut dokunulmazlığı iddiası | Anayasa+KMK | 0.40 | 0.38 | 0.38 |
| 13 | Ev sahibi aidat borcu için kiracıya gidin diyebilir mi? | KMK | 0.33 | 0.60 | 0.15 |
| 14 | Komşu gürültü ve koku — kanuni haklar | TMK | 0.29 | **0.67** | 0.33 |
| 15 | Polis veya yönetici izinsiz girebilir mi? | Anayasa | 0.00 | 0.33 | 0.78 |

---

## 3. Kanun Bazlı Performans Karşılaştırması

| Hukuk Kaynağı | Soru Sayısı | Ort. Faithfulness | Ort. Relevancy | Ort. Correctness |
|---------------|:-----------:|:-----------------:|:--------------:|:----------------:|
| **Kat Mülkiyeti Kanunu** | 6 | 0.56 | 0.57 | 0.45 |
| **Türk Borçlar Kanunu** | 3 | **1.00** | 0.55 | 0.71 |
| **Asansör Yönetmeliği** | 2 | 0.40 | 0.31 | 0.70 |
| **Yangın Yönetmeliği** | 1 | **1.00** | 0.60 | 0.74 |
| **Anayasa** | 2 | 0.20 | 0.36 | 0.58 |
| **Türk Medeni Kanunu** | 1 | 0.29 | **0.67** | 0.33 |

---

## 4. Güçlü ve Zayıf Yönler

### ✅ En Başarılı Alanlar

| # | Soru | Neden Başarılı? |
|---|------|-----------------|
| 5 | Kira ödeme zamanı (TBK) | Net madde, açık soru → F=1.00, AC=0.82 |
| 6 | Kiralanan ayıpları (TBK) | İyi chunk eşleşmesi → F=1.00, AC=0.77 |
| 10 | Yangın söndürücü sayısı | Spesifik veri, sayısal cevap → F=1.00, AC=0.74 |

### ⚠️ Geliştirilmesi Gereken Alanlar

| # | Soru | Sorun Analizi |
|---|------|---------------|
| 9 | Kırmızı etiket asansör | Retriever doğru chunk bulamadı → F=0.00 |
| 15 | Konut dokunulmazlığı | Anayasa chunk'ları yetersiz → F=0.00 |
| 13 | Aidat borcu kiracıya devredilebilir mi? | LLM halüsinasyon yaptı → AC=0.15 |

---

## 5. RAG Parametreleri

| Parametre | Değer | Etkisi |
|-----------|-------|--------|
| **LLM** | `gpt-4o` | Geniş Türkçe anlama kapasitesi |
| **Embedding** | `text-embedding-3-small` | Hızlı, maliyet-etkin vektörleştirme |
| **Chunk Size** | 2000 karakter | Daha fazla bağlam korunur |
| **Chunk Overlap** | 400 karakter | Madde bölünmelerini önler |
| **Top-K** | 6 | Her sorgu için 6 chunk değerlendirilir |
| **Temperature** | 0.0 | Deterministik = tutarlı cevaplar |

---
