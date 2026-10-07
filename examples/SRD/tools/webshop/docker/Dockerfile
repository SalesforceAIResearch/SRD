# Container image for the SDPO_ReAct webshop_step sidecar.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git openjdk-21-jdk-headless curl \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# `python -m spacy download` shells out to a pip URL with an old-style egg
# fragment (`#egg=en_core_web_lg`) that pip>=25 rejects outright -- install
# the model wheels directly instead of going through spacy's own downloader.
# web_agent_site/engine/goal.py hard-loads en_core_web_sm (a SEPARATE, smaller
# model from en_core_web_lg -- both needed, not a duplicate).
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl

# Vendor WebShop verbatim (same discipline as ../../search/docker/browser.py's
# "copied VERBATIM" note) -- this is princeton-nlp/WebShop's own source, not
# reimplemented here.
RUN git clone --depth 1 https://github.com/princeton-nlp/WebShop.git /app/webshop

WORKDIR /app/webshop
# Small (1000-product) dataset + its human-instruction file. WebShop's own
# setup.sh fetches these via gdown from Google Drive, but that link is
# rate-limited/access-blocked in practice (confirmed: fails identically from
# both this container and the host, "Cannot retrieve the public link").
# WooooDyy/AgentGym vendors the IDENTICAL small-dataset files directly in
# their repo (verified: same byte sizes as WebShop's own docs describe,
# real JSON not an LFS pointer) -- use that mirror instead of gdown.
RUN mkdir -p data && cd data && \
    curl -sfL -o items_shuffle_1000.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_shuffle_1000.json" && \
    curl -sfL -o items_ins_v2_1000.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_ins_v2_1000.json" && \
    curl -sfL -o items_human_ins.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_human_ins.json"

# Build the Lucene index the env needs to even instantiate (not just to
# search) -- see requirements.txt's comment. convert_product_file_format.py
# writes into 4 resource-variant dirs without creating any of them first.
RUN cd search_engine && mkdir -p resources resources_100 resources_1k resources_100k && \
    python convert_product_file_format.py && bash run_indexing.sh

WORKDIR /app
COPY run.py router.py worker.py .
ENV PYTHONPATH=/app/webshop

EXPOSE 8422
CMD ["python3", "run.py"]
