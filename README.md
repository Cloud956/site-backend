# site-backend
📦 Backend (FastAPI + Gunicorn)
Build image
docker build -t site-backend .
Run container
docker run -d \
  --name backend \
  -p 8000:8000 \
  site-backend
