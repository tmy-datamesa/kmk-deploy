"""
agent.py — Agentic RAG: Planlama + Araştırma + Cevaplama
=========================================================
Sistemin beyni: soruyu analiz eder, doğru kanunu seçer,
ilgili maddeleri bulur ve kaynaklı cevap üretir.

OpenAI Function Calling (tool_use) kullanarak hangi kanunun
aranacağına LLM'in kendisi karar verir — bu yaklaşıma
"Agentic RAG" denir.

"""

from openai import OpenAI
import json
import re
import mlflow
from src import config
from src.rag_engine import LegalRAGTool
from src import utils

class LegalRAG:
    """
    RAG SİSTEMİ (ROUTER + RETRIEVER + GENERATOR)
    ---------------------------------------------
    Bu sınıf sistemin beyni olarak çalışır ve tüm akışı yönetir:
    1. Planlama (Router): Sorunun hangi kanunla ilgili olduğunu belirler.
    2. Araştırma (Retriever): İlgili kanun içinde vektör araması yapar.
    3. Cevaplama (Generator): Bulunan bilgileri kullanarak kullanıcıya cevap üretir.
    """
    
    def __init__(self):
        """
        Sistemi Hazırla:
        - OpenAI ve ChromaDB bağlantılarını kur.
        - MLflow takip sistemini başlat.
        - Tüm hukuk kaynaklarını (Tools) hafızaya yükle.
        """
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.chroma_client = utils.get_chroma_client()
        
        # MLflow Konfigürasyonu
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
        # ARAÇLARI HAZIRLA (Her kanun için RAG motoru)
        self.tools_map = {}
        for key, info in config.LEGAL_DOCS.items():
            self.tools_map[key] = LegalRAGTool(info["collection"], self.chroma_client)
            
    def _get_system_prompt(self):
        """
        SİSTEM PROMPT (KİMLİK VE KURALLAR)
        ----------------------------------
        Ajanın nasıl davranacağını, hangi üslubu kullanacağını ve uyması gereken
        hukuki kuralları (Normlar Hiyerarşisi vb.) burada tanımlıyoruz.
        """
        return """
        Sen apartman, site ve konut hukuku konusunda uzman bir asistansın.
        Görevin: Kat Mülkiyeti Kanunu, Türk Borçlar Kanunu, Türk Medeni Kanunu,
        Anayasa ve ilgili yönetmelikler (asansör, yangın vb.) hakkındaki soruları
        SADECE sana verilen bağlam (context) bilgisine dayanarak cevaplamak.

        CEVAP FORMATI (ZORUNLU):
        - Soruyu doğrudan ve eksiksiz cevapla. Sorulan şeyin cevabını atla geçme.
        - Cevabını tek paragraf halinde yaz. Uzun açıklamalar yapma ama soruyu tam karşıla.
        - Cevabına ilgili kanun veya yönetmeliğin ismiyle başla.
          Örnekler: "Kat Mülkiyeti Kanunu uyarınca, ...", "Türk Borçlar Kanunu uyarınca, ..."
        - Madde numarası YAZMA. Kaynak referansı sistem tarafından otomatik ekleniyor.
        - Başlık, madde işareti, numara listesi KULLANMA. Düz metin yaz.

        KRİTİK KURALLAR:
        1. SADECE sana verilen bağlamdaki bilgiyi kullan. Bağlamda olmayan bilgiyi EKLEME.
        2. Bağlamda bilgi yoksa sadece "Bu konuda verilen metinlerde bilgi bulunmamaktadır." de.
        3. Cevabı Türkçe yaz.
        """

    def _extract_article_refs(self, sources):
        """
        MADDE NUMARASI ÇIKARICI (Hallucination Önleyici)
        ------------------------------------------------
        Ne Yapar: Retrieved chunk'ların ham metninden regex ile
        madde numaralarını çeker ve kaynak adına göre gruplar.

        Girdi: sources — [{'content': '...Madde 20...', 'metadata': {'doc_name': 'Kat Mülkiyeti Kanunu'}}]
        Çıktı: "📌 Kaynak: Kat Mülkiyeti Kanunu (Madde 20, 4)"

        Neden: LLM bazen doğru maddeyi bilse de numarayı yanlış yazabilir
        (hallucination). Bu fonksiyon sadece gerçekten chunk'ta geçen
        madde numaralarını kullanır.
        """
        # Her kaynak dokümanı için bulunan madde numaralarını topla
        doc_articles = {}  # {'Kat Mülkiyeti Kanunu': {20, 4, 25}, ...}

        for src in sources:
            doc_name = src.get("metadata", {}).get("doc_name", "Bilinmiyor")
            content = src.get("content", "")

            # Regex: "Madde 20", "Ek Madde 3", "Geçici Madde 1" gibi kalıpları yakala
            # Negatif lookbehind ile "Ek Madde" ve "Geçici Madde" ayrı yakalanır
            patterns = re.findall(r'(?:Ek Madde|Geçici Madde|Madde)\s+(\d+)', content)

            if doc_name not in doc_articles:
                doc_articles[doc_name] = set()
            # Bulunan numaraları set'e ekle (tekrar önleme)
            doc_articles[doc_name].update(patterns)

        # Hiç madde bulunamadıysa boş döndür
        if not doc_articles or all(len(v) == 0 for v in doc_articles.values()):
            return ""

        # Formatla: "📌 Kaynak: KMK (Madde 4, 20) | TBK (Madde 314)"
        parts = []
        for doc_name, articles in doc_articles.items():
            if articles:
                # Madde numaralarını sayısal sıraya koy
                sorted_articles = sorted(articles, key=int)
                madde_str = ", ".join([f"Madde {a}" for a in sorted_articles])
                parts.append(f"{doc_name} ({madde_str})")
            else:
                parts.append(doc_name)

        return "📌 Kaynak: " + " | ".join(parts)

    def _get_openai_tools(self):
        """
        LLM'e Sunulacak Araçlar (Tools).
        OpenAI Function Calling formatına uygun olarak tanımlanır.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": f"search_{key}",
                    "description": info['description'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Aranacak konu"}
                        },
                        "required": ["query"]
                    }
                }
            }
            for key, info in config.LEGAL_DOCS.items()
        ]

    def generate_answer(self, user_query):
        """
        ANA AKIŞ (Main Flow):
        1. Planlama: Soruyu al ve LLM'e gönder. Hangi aracı çağıracağına karar versin.
        2. Araştırma: Eğer araç çağırdıysa, ilgili kanun içinde arama yap.
        3. Cevaplama: Bulunan bilgileri LLM'e geri gönder ve nihai cevabı ürettir.
        """
        # Her sorgu için bir MLflow Run başlat (Kullanım takibi için)
        # Not: Metin dosyaları (artifacts) yoğunluk yaratmaması için loglanmıyor, sadece parametreler takip ediliyor.
        run_name = user_query[:50] + "..." if len(user_query) > 50 else user_query
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "llm_model": config.LLM_MODEL,
                "top_k": config.TOP_K,
                "embedding_model": config.EMBEDDING_MODEL,
                "temperature": config.TEMPERATURE
            })
            
            # --- 1. ADIM: Planlama ---
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_query}
            ]
            
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=self._get_openai_tools(),
                tool_choice="auto",
                temperature=config.TEMPERATURE
            )
            
            msg = response.choices[0].message
            tool_calls = msg.tool_calls
            used_sources = [] # Artık dict listesi olacak
            
            # --- 2. ADIM: Araç Kullanımı (Varsa) ---
            if tool_calls:
                messages.append(msg)
                
                for tool_call in tool_calls:
                    # Fonksiyon isminden hangi kanunu arayacağını anla (örn: search_kmk -> kmk)
                    doc_key = tool_call.function.name.replace("search_", "")
                    query = json.loads(tool_call.function.arguments).get("query")
                    
                    rag_tool = self.tools_map.get(doc_key)
                    context_str = "Bilgi bulunamadı."
                    
                    if rag_tool:
                        results = rag_tool.get_context(query)
                        if results:
                            # Context stringini oluştur (LLM için)
                            context_str = "\n".join([r['content'] for r in results])
                            
                            # Ham veriyi sakla (UI ve Eval için)
                            # Her bir result zaten {'content': ..., 'metadata': ...} formatında
                            used_sources.extend(results)

                    # Aracı çalıştır ve sonucunu mesaja ekle
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": context_str
                    })

                # --- 3. ADIM: Nihai Cevap ---
                final_response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=config.TEMPERATURE
                )
                answer = final_response.choices[0].message.content
            else:
                # Araç çağırmadıysa doğrudan cevabı döndür
                answer = msg.content

            # --- 4. ADIM: Kaynak Referansını Koddan Ekle ---
            # LLM'in madde numarası uydurmak yerine, chunk'lardan
            # regex ile çekilen gerçek madde numaralarını sona ekle.
            ref_header = self._extract_article_refs(used_sources)
            if ref_header:
                answer = f"{answer}\n\n{ref_header}"

            return answer, used_sources
