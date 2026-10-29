FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir .
ENTRYPOINT ["nufield"]
CMD ["demo", "--output-dir", "outputs/demo", "--n-sims", "80", "--grid-size", "8", "--seed", "7"]
