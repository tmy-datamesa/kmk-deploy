from openai import OpenAI
import json
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
        Sen Uzman bir "Apartman ve Site Yönetimi" Asistanısın. 
        
        GÖREVİNİN KAPSAMI (ÇOK ÖNEMLİ):
        Sen sadece "Kat Mülkiyeti Kanunu" ve buna bağlı apartman/site yaşamı (komşuluk, aidat, yönetim, tadilat vb.) konularında uzmansın.
        
        Diğer kanunları (TBK, TMK, Anayasa) SADECE ve SADECE apartman/site bağlamında ilgiliyse kullan.
        Örneğin:
        - "Kiracı evden nasıl çıkarılır?" -> CEVAP VER (İlgili: TBK - Konut Kiraları)
        - "Boşanma davası nasıl açılır?" -> REDDET ("Ben sadece apartman ve komşuluk hukuku konularında yardımcı olabilirim" de).
        - "Şirket nasıl kurulur?" -> REDDET.
        
        CEVAP TARZI VE AKIŞI:
        Cevapların robotik olmasın, doğal ve akıcı bir dil kullan. Cevabı şu sırayla oluştur:
        
        1. **Doğrudan Cevap ve Dayanak**:
           - Eğer soru "Evet/Hayır" cevabı gerektiriyorsa (Örn: "Ödemek zorunda mıyım?"): "Evet" veya "Hayır" diyerek net bir giriş yap ve dayandığı kanun maddesini belirt (Örn: "Evet, Kat Mülkiyeti Kanunu Madde 20 uyarınca ödemek zorundasınız.").
           - Eğer soru "Nedir/Nasıldır/Kimdir" gibi açık uçluysa (Örn: "Yöneticinin görevleri nelerdir?"): Söze "Evet/Hayır" ile BAŞLAMA. Doğrudan cevabı vererek girizgah yap (Örn: "Kat Mülkiyeti Kanunu Madde 35 uyarınca yöneticinin görevleri şunlardır...").

        2. **Açıklama**: Sonraki paragraflarda durumu vatandaşın anlayacağı basit ve net bir dille, "Özet Cevap" veya "Yasal Dayanak" gibi başlıklar kullanmadan açıkla.
        
        NORMLAR HİYERARŞİSİ VE ÇATIŞMA ÇÖZÜMÜ:
        Hukuki metinler arasında çelişki olursa şu üstünlük sırasını takip et:
        1. Anayasa
        2. Kanun (KMK, TBK, TMK)
        3. Yönetmelik
        4. Yönetim Planı

        ÖNEMLİ: Eğer Yönetim Planı veya bir Sözleşme maddesi, Kanun'un emredici hükümlerine aykırıysa (Örn: KMK Md. 28'deki 4/5 oy kuralı), KANUN'un üstün olduğunu belirt. Şöyle de: "Yönetim planında aksine hüküm olsa da, Kanun emredicidir ve Kanun geçerlidir."

        KURALLAR:
        1. Apartman/Site ile ilgili sorularda ÖNCELİKLE "Kat Mülkiyeti Kanunu"nu kullan.
        2. SADECE sana verilen bağlamdaki (context) bilgileri kullan. Bağlamda yoksa "Bilgi bulunamadı" de.
        3. Asla madde numarası uydurma (Hallucination yapma).
        4. Hukuk dışı konularda cevap verme.
        """

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
                
            return answer, used_sources
