PYTHON = python3
PIP = pip

.PHONY: setup ingest run eval clean clean-logs

# Kurulum
setup:
	$(PIP) install -r requirements.txt
	@echo "Kurulum Tamamlandı! .env dosyanızı oluşturmayı unutmayın."

# Veri Yükleme (Tüm kanunları tarar)
ingest:
	@echo "Kütüphane Güncelleniyor..."
	$(PYTHON) -c "from src.ingestion import ingest_all_docs; ingest_all_docs(force_recreate=True)"

# Uygulamayı Başlat
run:
	streamlit run app.py

# Değerlendirme (RAGAS + MLflow)
eval:
	$(PYTHON) src/evaluation.py
	@echo "Değerlendirme tamamlandı. Sonuçları görmek için: mlflow ui"

# Temizlik (Önbellek)
clean:
	rm -rf __pycache__ src/__pycache__
	@echo "Önbellek temizlendi."

# Log Temizliği (MLflow)
clean-logs:
	rm -rf mlruns mlflow.db
	@echo "MLflow logları ve veritabanı temizlendi."
