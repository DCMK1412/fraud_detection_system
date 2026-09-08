FROM python:3.11-slim

WORKDIR /app

COPY requirements_docker.txt .

RUN pip install --no-cache-dir --default-timeout=300 fastapi==0.110.0 uvicorn==0.27.1 numpy==1.26.3 pandas==2.2.1 scikit-learn==1.3.2 imbalanced-learn==0.11.0 joblib==1.3.2 pydantic==2.6.3

RUN pip install --no-cache-dir --default-timeout=600 --retries 10 xgboost==2.0.3

COPY app.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]