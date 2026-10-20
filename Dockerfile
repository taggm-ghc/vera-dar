# VERA-DAR image (p3m3 item #73). The same image runs both Render services:
#   API: default CMD (./start.sh api)
#   UI:  Render "Docker Command" set to ./start.sh ui
# Secrets come from Render environment variables at runtime, never from the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# requirements.txt only; requirements-dev.txt is excluded by .dockerignore.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 vera && chown -R vera /app
USER vera

EXPOSE 8000

CMD ["./start.sh", "api"]
