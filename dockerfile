FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app_updated.py .

# Cloud Run sets PORT automatically
ENV PORT=8080
EXPOSE 8080

# IMPORTANT: run app_updated.py, NOT main.py
CMD ["uvicorn", "app_updated:app", "--host", "0.0.0.0", "--port", "8080"]