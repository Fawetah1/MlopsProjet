FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY app.py .
COPY traffic_model_fixed2.pkl .
COPY processed_features2.csv .

EXPOSE 5000

CMD ["python", "app.py"]