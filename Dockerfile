FROM python:3.11-slim

# System deps for Pillow and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + server
COPY model /app/model
COPY app.py /app/app.py

# Cloud Run
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
