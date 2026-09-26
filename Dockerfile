FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY steered_finetuner ./steered_finetuner
COPY analysis ./analysis
COPY tests ./tests
COPY pyproject.toml README.md ./

ENTRYPOINT ["python3"]
