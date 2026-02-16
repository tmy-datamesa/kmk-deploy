# Dockerfile - Legal RAG API Backend
# Cloud Run üzerinde çalıştırılmak üzere optimize edilmiştir.

FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Önce requirements kopyala (Docker cache için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Cloud Run portu: 8080
EXPOSE 8080

# Uvicorn ile FastAPI başlat
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8080"]
