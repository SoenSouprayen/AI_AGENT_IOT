FROM python:3.13.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && useradd --uid 10001 --no-create-home agent

COPY --chown=agent:agent interface.py ./
COPY --chown=agent:agent src ./src

USER agent

EXPOSE 8501

ENTRYPOINT ["python", "interface.py"]
CMD ["--no-browser", "--host", "0.0.0.0"]
