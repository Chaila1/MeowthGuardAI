FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flask torch torchvision pillow

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]