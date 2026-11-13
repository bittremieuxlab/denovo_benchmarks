# Streamlit dashboard for the de novo benchmarking results.
#
# Only the dashboard runs here. The benchmarking pipeline itself needs Apptainer
# and GPUs and is excluded from the build context via .dockerignore.

FROM python:3.12-slim

WORKDIR /app

# curl is needed by the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements-dashboard.txt /app/requirements-dashboard.txt

RUN pip3 install --no-cache-dir -r requirements-dashboard.txt

# Brings in dashboard.py, datasets_info.py and results/
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
