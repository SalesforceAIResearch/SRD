# Container image for the SDPO_ReAct alfworld_step sidecar.
FROM python:3.10-slim

# textworld's jericho dependency builds a C++ frotz interpreter (calls `make`
# at pip-install time, not a prebuilt wheel) -- python:3.10-slim has no build
# toolchain by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# alfworld-download's default cache dir; ALFWORLD_DATA must point here at
# runtime too (set below) so config_tw.yaml's $ALFWORLD_DATA-relative paths
# resolve to what was actually downloaded at build time.
ENV ALFWORLD_DATA=/root/.cache/alfworld
RUN alfworld-download -f

COPY run.py router.py worker.py config_tw.yaml .
ENV ALFWORLD_CONFIG=/app/config_tw.yaml

EXPOSE 8423
CMD ["python3", "run.py"]
