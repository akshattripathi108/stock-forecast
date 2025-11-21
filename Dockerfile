
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV FLASK_APP=src.app
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["gunicorn", "--chdir", "src", "app:app", "-b", "0.0.0.0:8080", "--workers", "1"]
