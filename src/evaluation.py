import pandas as pd
import mlflow
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.agent import LegalRAG
from src import config

# Basit bir test veri seti (Sorular ve Beklenen Cevaplar)
# bu veriler harici bir dosyadan (JSON/CSV) okunmalıdır.
import json
import os

# Test veri seti yolu
EVAL_DATA_PATH = os.path.join(config.DATA_DIR, "eval_data.json")

def run_evaluation():
    print("Değerlendirme Başlıyor...")
    rag = LegalRAG()
    
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # Veri setini yükle
    try:
        with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"{len(test_data)} adet test sorusu yüklendi.")
    except FileNotFoundError:
        print(f"HATA: {EVAL_DATA_PATH} bulunamadı!")
        return

    # 1. Soruları RAG Sistemine Sor
    for item in test_data:
        print(f"Soru: {item['question']}")
        
        # Cevap ve Kaynakları Al
        answer, sources = rag.generate_answer(item['question'])
        
        # RAGAS DEĞERLENDİRMESİ İÇİN VERİ HAZIRLIĞI
        # ------------------------------------------
        # Ragas 'contexts' olarak sadece metin listesi (list[str]) bekler.
        # agent.py artık sözlük listesi döndürüyor ({'content':..., 'metadata':...}).
        # Bu yüzden sadece 'content' alanını çekiyoruz.
        context_contents = [s['content'] for s in sources]
        
        results["question"].append(item['question'])
        results["answer"].append(answer)
        results["contexts"].append(context_contents)
        # JSON'daki anahtar 'ground_truth_answer'
        results["ground_truth"].append(item.get('ground_truth_answer', ''))

    # 2. Dataset Oluştur
    dataset = Dataset.from_dict(results)

    # 3. RAGAS ile Değerlendir
    print("RAGAS Metrikleri Hesaplanıyor...")
    
    # LLM ve Embedding modellerini açıkça belirtiyoruz
    eval_llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
    eval_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print(f"Sonuçlar: {scores}")

    # 4. MLflow'a Kaydet
    with mlflow.start_run(run_name="ragas_evaluation"):
        # RAGAS skorlarını sözlüğe çevir (Pandas üzerinden ortalama alarak)
        # scores bir EvaluationResult nesnesidir.
        scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
        
        mlflow.log_metrics(scores_dict)
        
        # Detaylı sonuçları CSV olarak kaydet
        df_result = scores.to_pandas()
        df_result.to_csv("evaluation_results.csv", index=False)
        mlflow.log_artifact("evaluation_results.csv")
    
    print("Değerlendirme tamamlandı ve MLflow'a kaydedildi.")

if __name__ == "__main__":
    run_evaluation()
