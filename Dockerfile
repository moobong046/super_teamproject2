FROM python:3.8-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libmagic-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
EXPOSE 9090
# src 폴더 내부에서 app.py를 실행하기 위해 작업 디렉토리 설정 유의
CMD ["python", "src/app.py"]