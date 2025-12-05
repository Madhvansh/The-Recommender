# Lightweight CPU image for serving the recommender.
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (smaller, no CUDA), then the package.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .

# Catalog size must be provided at run time (depends on the trained dataset).
ENV RECO_CONFIG=configs/s4rec_amazon_beauty.yaml \
    RECO_CKPT=/app/checkpoints/best.pt \
    RECO_DEVICE=cpu

EXPOSE 8000
# Mount a trained checkpoint at /app/checkpoints and set RECO_NUM_ITEMS.
CMD ["uvicorn", "recommender.serve:app", "--host", "0.0.0.0", "--port", "8000"]
