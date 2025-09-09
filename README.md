Train locally
# (optional) create venv
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
python train.py
# outputs: ./model/

Run API locally
uvicorn app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080/docs

Build&deploy to Google Cloud Run
gcloud config set project YOUR_PROJECT_ID

# Build container (from folder with Dockerfile, app.py, model/, requirements.txt)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/quality-gate

# Deploy (public)
gcloud run deploy quality-gate \
  --image gcr.io/YOUR_PROJECT_ID/quality-gate \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1

