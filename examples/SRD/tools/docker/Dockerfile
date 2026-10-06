# Container image for the SDPO_ReAct code_interpreter sandbox.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --chmod=0644: the server is read+executed by the unprivileged `sandbox` user
# (USER below), so it must be world-readable in the image REGARDLESS of the
# source file's mode in the build context -- a checkout that happens to carry a
# restrictive umask/ACL (0600) would otherwise COPY straight through and crash
# the container at import with PermissionError: '/app/sandbox_server.py'.
COPY --chmod=0644 sandbox_server.py .

# Unprivileged user: code submitted by the (untrusted, still-training) policy
# runs as this user inside the container, not root.
RUN useradd -m -u 1000 sandbox
USER sandbox

EXPOSE 8420
CMD ["uvicorn", "sandbox_server:app", "--host", "0.0.0.0", "--port", "8420"]
