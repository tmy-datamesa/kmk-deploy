from src import config, utils

class LegalRAGTool:
    """
    TEKİL ARAMA MOTORU (RETRIEVER)
    ------------------------------
    Bu sınıf, belirli bir hukuk kaynağı (kanun, yönetmelik vb.) içerisinde
    vektör tabanlı arama yapmaktan sorumludur.
    
    Kullanım Amacı:
    - Verilen soruyu (query) alır.
    - İlgili ChromaDB koleksiyonunda en yakın eşleşmeleri bulur.
    - Bulunan içerikleri LLM'in anlayacağı formata çevirir.
    """
    
    def __init__(self, collection_name, client=None):
        """
        Araç ilklendirme.
        
        Girdi:
        - collection_name: Aranacak kanunun ChromaDB'deki koleksiyon adı (örn: "law_kmk")
        - client: (Opsiyonel) Var olan bir ChromaDB istemcisi. Yoksa yenisini oluşturur.
        """
        self.client = client or utils.get_chroma_client()
        self.embedding_fn = utils.get_embedding_function()
        
        # Spesifik koleksiyona bağlan
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def retrieve(self, query):
        """
        Vektör Araması Yapar.
        
        Ne Yapar:
        1. Soruyu embedding (sayı vektörü) haline getirir.
        2. Koleksiyondaki en yakın `config.TOP_K` adet parçayı bulur.
        
        Girdi:
        - query (str): Kullanıcı sorusu veya arama metni.
        
        Çıktı:
        - list: Bulunan dokümanların içerik ve metadatalarını içeren sözlük listesi.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=config.TOP_K
        )
        
        # Sonuçları işle ve yapılandır
        structured_results = []
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
            
            for doc, meta in zip(docs, metas):
                structured_results.append({
                    "content": doc,
                    "metadata": meta
                })
                
        return structured_results

    def get_context(self, query):
        """
        Ajan (Agent) tarafından kullanılan standart arayüz.
        Doğrudan retrieve fonksiyonunu çağırır.
        """
        return self.retrieve(query)
