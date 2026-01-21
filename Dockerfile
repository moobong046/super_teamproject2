# 1. 베이스 이미지 설정
FROM python:3.8-slim

# 2. 필수 패키지 설치
RUN apt-get update && apt-get install -y libmagic-dev && rm -rf /var/lib/apt/lists/*

# 3. Hugging Face 보안 정책에 따른 사용자(user) 설정
# (중요: USER user 설정 이후의 작업은 모두 이 사용자의 권한으로 진행됩니다)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. 작업 디렉토리 설정 (사용자 홈 디렉토리 하위로)
WORKDIR $HOME/app

# 5. 라이브러리 설치 (캐시 활용을 위해 requirements부터 복사)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 전체 소스 코드 복사 (권한 유지)
COPY --chown=user . .

# 7. 포트 설정 (Hugging Face 기본 포트)
EXPOSE 7860

# 8. 앱 실행
# src 폴더 안에 app.py가 있다면 경로를 정확히 지정해야 합니다.
CMD ["python", "src/app.py"]