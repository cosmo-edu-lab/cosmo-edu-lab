
FROM python:3.10-slim


ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /App



RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install


EXPOSE 7860
ENV MPLCONFIGDIR=/tmp/matplotlib


CMD git clone https://$GITHUB_TOKEN@github.com/Elyon7/Cosmo-Edu_Lab.git . && \
    git lfs pull && \
    mkdir -p App/student_submissions && \
    chmod 777 App/student_submissions && \
    pip install --no-cache-dir -r requirements.txt && \
    cd App && \
    python main.py