# ============================================================
# BASE IMAGE 
#Ubuntu-based minimal Linux filesystem
#Python runtime
#system libraries
# ============================================================
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install dist/*.whl || true
CMD ["python", "main.py"]
