# Container image for the SDPO_ReAct search/open/find sidecar.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY browser.py server.py retrieval_adapter.py .

EXPOSE 8421
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8421"]
