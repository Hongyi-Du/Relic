FROM python:3.11-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*

COPY codebase/ /app/repo/
COPY knowledge/ /app/knowledge/
RUN cp -r /app/repo /app/workspace && mkdir -p /logs/verifier
WORKDIR /app/workspace
