FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY multi_modal_rag_docker.py .
RUN mv multi_modal_rag_docker.py multi_modal_rag.py

EXPOSE 8081

CMD ["python", "multi_modal_rag.py"]
