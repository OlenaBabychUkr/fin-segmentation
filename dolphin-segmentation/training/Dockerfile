FROM python:3.11-slim

WORKDIR /app

# Runtime libraries commonly needed by OpenCV/Ultralytics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY promote_challenger.py .

CMD ["python", "train.py"]
