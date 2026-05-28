# Prebaked runtime image for the agent's Docker-persistent sandbox.
#
# Bakes the otherwise per-task `container_setup` work (a C toolchain + the pytest
# tooling the success gate depends on) into the image ONCE. When the agent runs
# with `--docker-image crucible-runtime`, it detects this tooling is already
# present and skips setup, dropping per-task container_setup from ~15s to ~1s.
#
# Build:   docker build -t crucible-runtime docker/
#   (or:   ./docker/build.sh)
# Use:     python -m agent '<goal>' --docker --docker-persistent --docker-image crucible-runtime

FROM python:3.12-slim

# The test runner + JSON report plugin the success gate parses. This alone is
# enough for the agent to auto-skip per-task container_setup, and keeps the
# image tiny (~tens of MB over the base).
RUN pip install --no-cache-dir pytest pytest-json-report

# Optional: a C build toolchain, only needed for the occasional source-only
# C-extension package (most popular packages ship wheels). It's large (hundreds
# of MB) — uncomment only if your tasks need to compile from source.
# RUN apt-get update -qq \
#     && apt-get install -y --no-install-recommends build-essential gcc python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# Common runtime deps baked in to eliminate per-task `dependency_install` for
# web tasks. httpx is required by FastAPI's TestClient (the gate uses it).
# Adjust this set to match your typical tasks.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx requests pydantic

WORKDIR /workspace
