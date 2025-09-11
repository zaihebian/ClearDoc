# 📄 ClearDoc – Document Quality Gate

ClearDoc is a lightweight AI service that checks whether a scanned document image is **GOOD** (usable) or **BAD** (blurred, skewed, low-light, compressed, etc.).

It uses a fine-tuned [ConvNeXtV2](https://huggingface.co/facebook/convnextv2-tiny-1k-224) model, wrapped in a FastAPI server and containerized with Docker.  
Perfect for integrating into document processing pipelines (e.g., OCR, RPA, or cloud workflows).

---

## 🚀 Features

- ✅ Classifies document quality as **GOOD** or **BAD**  
- 🔎 Provides heuristic **reason** for rejection (e.g., *blur/motion*, *low-light*, *compression*)  
- ⚡ FastAPI backend for easy REST integration  
- 🐳 Dockerized for cloud deployment (Google Cloud Run, AWS, Azure, etc.)  
- 📊 MLflow experiment tracking supported  

---

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/ClearDoc.git
cd ClearDoc
```

Create a virtual environment & install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Training (optional)

If you want to retrain or fine-tune:

```bash
python train.py
```

Tracked metrics and models will be saved in **MLflow** and `./model/`.

---

## 🌐 Running the API

### Local
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
Visit: [http://localhost:8080](http://localhost:8080)

### Docker
Build and run:
```bash
docker build -t cleardoc .
docker run -p 8080:8080 cleardoc
```

---

## 📡 API Endpoints

### Root
```http
GET /
```
**Response**
```json
{"status":"ok","labels":["GOOD","BAD"]}
```

### Predict
```http
POST /predict
Content-Type: multipart/form-data
file: <your-document-image>
```

**Response**
```json
{
  "verdict": "BAD",
  "prob_bad": 0.8723,
  "reason": "blur/motion"
}
```

---

## ☁️ Deployment on Google Cloud Run

1. Push Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/cleardoc
   ```

2. Deploy:
   ```bash
   gcloud run deploy cleardoc \
       --image gcr.io/<PROJECT_ID>/cleardoc \
       --platform managed \
       --allow-unauthenticated \
       --region <REGION>
   ```

3. Use the Cloud Run URL:
   ```bash
   curl -X POST -F "file=@sample.png" https://<YOUR-SERVICE-URL>/predict
   ```

---

## 🖼️ Example

![demo](docs/demo.png)

---

## ⚖️ License

MIT License © 2025 – Built for research and document automation workflows.
