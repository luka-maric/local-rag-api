FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "gradio>=6.0.0" "httpx>=0.27.0"

COPY ui/app.py .

EXPOSE 7860

CMD ["python", "app.py"]
